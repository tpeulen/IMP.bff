/**\file InferenceLaplace.cpp
 * Curvature, covariance, exact polish and evidence, and Tierney-Kadane moments of any evaluable
 * factor graph (PRD-156 step 4). Written 2026-11-11 by claude-code/Opus 5.5 for tpeulen; the
 * arithmetic is that of the decay-specific functions it replaces (BayesianDecayPosterior.h:
 * bayesian_decay_hessian, _covariance, _polish_exact, _laplace_evidence_exact, _fit_node_tilted,
 * _tk_mean_rel), on the graph's sums.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/fit/InferenceLaplace.h>
#include <IMP/bff/fit/Optimization.h>
#include <IMP/bff/bayesian/BayesianLaplace.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

IMPBFF_BEGIN_NAMESPACE

namespace {

//: the free block of a full dim x dim matrix
std::vector<double> reduce(const std::vector<double>& M, std::size_t dim, const std::vector<int>& F) {
  const std::size_t n = F.size();
  std::vector<double> out(n * n);
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = 0; b < n; ++b) out[a * n + b] = M[std::size_t(F[a]) * dim + std::size_t(F[b])];
  return out;
}

std::vector<double> reduce_vector(const std::vector<double>& v, const std::vector<int>& F) {
  std::vector<double> out(F.size());
  for (std::size_t a = 0; a < F.size(); ++a) out[a] = v[std::size_t(F[a])];
  return out;
}

//: a factor's flat positions in theta, scope order
std::vector<int> positions_of(const InferenceFactorGraph& g, const std::string& factor) {
  std::vector<int> pos;
  for (const std::string& v : g.variables_of(factor)) {
    const int at = g.index_of(v), size = std::max(1, g.get_variable_size(v));
    for (int k = 0; k < size; ++k) pos.push_back(at + k);
  }
  return pos;
}

}  // namespace

std::vector<int> inference_free_coordinates(const InferenceFactorGraph& graph, const std::vector<std::string>& fixed) {
  const std::size_t dim = graph.get_parameter_dimension();
  std::vector<char> held(dim, 0);
  for (const std::string& key : fixed) {
    const int at = graph.index_of(key);
    if (at < 0) throw std::invalid_argument("inference: no variable '" + key + "' to hold fixed");
    const int size = std::max(1, graph.get_variable_size(key));
    for (int k = 0; k < size; ++k) held[std::size_t(at + k)] = 1;
  }
  std::vector<int> F;
  for (std::size_t i = 0; i < dim; ++i)
    if (!held[i]) F.push_back(int(i));
  return F;
}

std::vector<double> inference_hessian(const InferenceFactorGraph& graph, const std::vector<double>& theta,
                                      const std::vector<std::string>& fixed, double rel) {
  const std::vector<int> F = inference_free_coordinates(graph, fixed);
  const std::size_t n = F.size();
  std::vector<double> H(n * n, 0.0);
  for (std::size_t b = 0; b < n; ++b) {
    const std::size_t j = std::size_t(F[b]);
    const double h = rel * std::max(1.0, std::fabs(theta[j]));
    std::vector<double> tp = theta, tm = theta;
    tp[j] += h;
    tm[j] -= h;
    const InferenceScore gp = graph.eval(tp, true), gm = graph.eval(tm, true);
    for (std::size_t a = 0; a < n; ++a)
      H[a * n + b] = -(gp.gradient[std::size_t(F[a])] - gm.gradient[std::size_t(F[a])]) / (2.0 * h);  // gradient of +log p
  }
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t b = a + 1; b < n; ++b) {
      const double v = 0.5 * (H[a * n + b] + H[b * n + a]);
      H[a * n + b] = H[b * n + a] = v;
    }
  return H;
}

std::vector<double> inference_covariance(const InferenceFactorGraph& graph, const std::vector<double>& theta,
                                         const std::vector<std::string>& fixed, bool exact) {
  const std::vector<int> F = inference_free_coordinates(graph, fixed);
  const std::size_t n = F.size(), dim = graph.get_parameter_dimension();
  const std::vector<double> A =
      exact ? inference_hessian(graph, theta, fixed) : reduce(graph.eval(theta, true).information, dim, F);
  CholeskyFactor ch;
  if (!ch.factor(A.data(), n)) return std::vector<double>();
  std::vector<double> S(n * n, 0.0), e(n, 0.0), col(n);
  for (std::size_t j = 0; j < n; ++j) {
    std::fill(e.begin(), e.end(), 0.0);
    e[j] = 1.0;
    ch.solve(e.data(), col.data());
    for (std::size_t i = 0; i < n; ++i) S[i * n + j] = col[i];
  }
  return S;
}

double inference_laplace_evidence(const InferenceFactorGraph& graph, const std::vector<double>& theta,
                                  const std::vector<std::string>& fixed, bool exact) {
  const std::vector<int> F = inference_free_coordinates(graph, fixed);
  const InferenceScore s = graph.eval(theta, true);
  const std::vector<double> A =
      exact ? inference_hessian(graph, theta, fixed) : reduce(s.information, graph.get_parameter_dimension(), F);
  double ev = 0.0;
  return bayesian_laplace_log_evidence(s.log_density, A.data(), F.size(), &ev) ? ev
                                                                               : -std::numeric_limits<double>::infinity();
}

InferenceFit inference_polish(const InferenceFactorGraph& graph, const InferenceFit& fit, int max_iterations,
                              double tolerance) {
  const std::vector<int> F = inference_free_coordinates(graph, fit.fixed);
  const std::size_t n = F.size();
  InferenceFit R = fit;
  std::vector<double> th = fit.theta, trial = fit.theta, x(n);
  for (std::size_t a = 0; a < n; ++a) x[a] = th[std::size_t(F[a])];
  auto objective = [&](const double* t) {
    for (std::size_t a = 0; a < n; ++a) trial[std::size_t(F[a])] = t[a];
    return -graph.eval(trial, false).log_density;
  };
  DampedNewton<decltype(objective)> stepper;   //: the default schedule, as bayesian_decay_polish_exact
  InferenceScore pt = graph.eval(th, true);
  int it = 0;
  for (; it < max_iterations; ++it) {
    const std::vector<double> H = inference_hessian(graph, th, fit.fixed);
    const std::vector<double> g = reduce_vector(pt.gradient, F);
    const OptimizationStepResult r = stepper.step(H.data(), g.data(), n, -pt.log_density, x.data(), objective);
    if (!r.accepted) break;
    for (std::size_t a = 0; a < n; ++a) th[std::size_t(F[a])] = x[a];
    pt = graph.eval(th, true);
    R.decrement = r.decrement;
    if (r.decrement < tolerance) { ++it; break; }
  }
  R.theta = th;
  R.at_mode = pt;
  R.iterations += it;
  const std::vector<double> H = inference_hessian(graph, th, fit.fixed);
  double ev = 0.0;
  R.laplace_ok = bayesian_laplace_log_evidence(pt.log_density, H.data(), n, &ev);
  R.laplace_evidence = R.laplace_ok ? ev : -std::numeric_limits<double>::infinity();
  return R;
}

InferenceTierneyKadane inference_tierney_kadane(const InferenceFactorGraph& graph, const InferenceFit& mode,
                                                InferenceFactorScore* functional,
                                                const std::vector<std::string>& scope, double eps,
                                                int max_iterations, bool exact_curvature) {
  if (functional == nullptr) throw std::invalid_argument("inference_tierney_kadane: no functional");
  InferenceTierneyKadane out;
  out.shift.assign(2, std::nan(""));
  out.dlogdet.assign(2, std::nan(""));
  out.flat_ratio.assign(2, std::nan(""));
  out.flat_curvature = out.flat_prior_share = std::nan("");
  out.mean = out.sd = std::nan("");
  const std::size_t dim = graph.get_parameter_dimension();
  const std::vector<int> F = inference_free_coordinates(graph, mode.fixed);
  const std::size_t n = F.size();
  std::vector<int> where(dim, -1);   //: theta position -> free index
  for (std::size_t a = 0; a < n; ++a) where[std::size_t(F[a])] = int(a);

  //: the base curvature both integrals share: the mode's information or the exact curvature
  const std::vector<double> A0 =
      exact_curvature ? inference_hessian(graph, mode.theta, mode.fixed) : reduce(mode.at_mode.information, dim, F);
  double logdet0 = 0.0;
  if (!bayesian_log_det_spd(A0.data(), n, &logdet0)) return out;
  std::vector<double> vflat(n, 0.0);
  double lam_min = std::nan("");
  const bool have_flat = bayesian_smallest_eigenpair(A0.data(), n, vflat.data(), &lam_min);
  out.flat_curvature = have_flat ? lam_min : std::nan("");
  if (have_flat && lam_min > 0.0) {
    //: how much of that direction's curvature the prior and hyperprior factors alone supply
    std::vector<double> P(n * n, 0.0);
    for (const std::string& k : graph.get_factor_keys()) {
      const InferenceFactorKind kind = graph.get_factor_kind(k);
      if (kind != INFERENCE_FACTOR_PRIOR && kind != INFERENCE_FACTOR_HYPER) continue;
      InferenceFactorScore* s = graph.get_factor_score(k);
      const std::vector<int> pos = positions_of(graph, k);
      const std::size_t m = pos.size();
      const std::vector<double> r = s->eval(graph.get_factor_values(k, mode.theta), true);
      for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b) {
          const int ia = where[std::size_t(pos[a])], ib = where[std::size_t(pos[b])];
          if (ia >= 0 && ib >= 0) P[std::size_t(ia) * n + std::size_t(ib)] += r[1 + m + a * m + b];
        }
    }
    double q = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
      double s = 0.0;
      for (std::size_t b = 0; b < n; ++b) s += P[a * n + b] * vflat[b];
      q += vflat[a] * s;
    }
    out.flat_prior_share = q / lam_min;
  }

  //: the functional's scope positions, and the tilt power * log(f + eps) as a score over that scope
  InferenceFactorGraph probe = graph;
  const std::string tilt_key = "__tierney_kadane_tilt";
  probe.add_factor(tilt_key, INFERENCE_FACTOR_PRIOR, scope);
  const std::vector<int> spos = positions_of(probe, tilt_key);
  const std::size_t m = spos.size();
  IMP::Pointer<InferenceFactorScore> keep(functional);
  auto tilt = [keep, eps, m](double power, const std::vector<double>& x, bool derivs) {
    const std::vector<double> r = keep->eval(x, derivs);
    const double d = r.at(0) + eps;
    const double value = power * std::log(d);
    if (!derivs) return std::vector<double>{value};
    if (r.size() != 1 + m + m * m)
      throw std::runtime_error("inference_tierney_kadane: the functional returned " + std::to_string(r.size()) +
                               " values, expected [f, df, -d2f] over " + std::to_string(m) + " coordinates");
    const double k1 = power / d, k2 = power / (d * d);
    std::vector<double> out(1 + m + m * m);
    out[0] = value;
    for (std::size_t a = 0; a < m; ++a) out[1 + a] = k1 * r[1 + a];
    //: information of the tilt = -d2 tilt = k1 (-d2f) + k2 df df'
    for (std::size_t a = 0; a < m; ++a)
      for (std::size_t b = 0; b < m; ++b) out[1 + m + a * m + b] = k1 * r[1 + m + a * m + b] + k2 * r[1 + a] * r[1 + b];
    return out;
  };

  const double L0 = mode.at_mode.log_density;
  double lr[2] = {std::nan(""), std::nan("")};
  for (int k = 0; k < 2; ++k) {
    const double power = double(k + 1);
    InferenceFactorGraph tg = graph;
    tg.add_factor(tilt_key, INFERENCE_FACTOR_PRIOR, scope);
    tg.set_factor_score(tilt_key, new InferenceFunctionScore(
                                      [tilt, power](const std::vector<double>& x, bool d) { return tilt(power, x, d); }));
    //: the default step schedule (line search from the first step) and tolerance of bayesian_decay_fit_node_tilted
    const InferenceFit t = inference_fitting(tg, mode.theta, max_iterations, 1e-8,
                                             std::numeric_limits<double>::infinity(), mode.fixed);
    out.iterations += t.iterations;
    if (!t.converged || !std::isfinite(t.at_mode.log_density)) return out;
    const std::vector<double> T = tilt(power, tg.get_factor_values(tilt_key, t.theta), true);
    std::vector<double> H = A0;   //: the MODE's matrix plus the tilt's exact curvature at the tilted mode
    for (std::size_t a = 0; a < m; ++a)
      for (std::size_t b = 0; b < m; ++b) {
        const int ia = where[std::size_t(spos[a])], ib = where[std::size_t(spos[b])];
        if (ia >= 0 && ib >= 0) H[std::size_t(ia) * n + std::size_t(ib)] += T[1 + m + a * m + b];
      }
    double logdet1 = 0.0;
    if (!bayesian_log_det_spd(H.data(), n, &logdet1)) return out;
    //: the tilted fit's log density already is log p + power log(f + eps) at the tilted mode
    lr[k] = t.at_mode.log_density - L0 - 0.5 * (logdet1 - logdet0);
    out.dlogdet[std::size_t(k)] = logdet1 - logdet0;
    double d2 = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
      const double da = t.theta[std::size_t(F[a])] - mode.theta[std::size_t(F[a])];
      if (da == 0.0) continue;
      for (std::size_t b = 0; b < n; ++b) d2 += da * A0[a * n + b] * (t.theta[std::size_t(F[b])] - mode.theta[std::size_t(F[b])]);
    }
    out.shift[std::size_t(k)] = std::sqrt(d2 > 0.0 ? d2 : 0.0);
    if (have_flat) {
      double q = 0.0;
      for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b) {
          const int ia = where[std::size_t(spos[a])], ib = where[std::size_t(spos[b])];
          if (ia >= 0 && ib >= 0) q += vflat[std::size_t(ia)] * T[1 + m + a * m + b] * vflat[std::size_t(ib)];
        }
      out.flat_ratio[std::size_t(k)] = lam_min > 0.0 ? std::fabs(q) / lam_min : std::numeric_limits<double>::infinity();
    }
  }
  const BayesianTierneyKadane r = bayesian_tierney_kadane(lr[0], lr[1], eps);
  out.mean = r.mean;
  out.sd = r.sd;
  out.ok = r.ok;
  return out;
}

IMPBFF_END_NAMESPACE

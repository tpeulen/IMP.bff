/**\file InferenceFitting.cpp
 * One generic fit of an evaluable factor graph (PRD-155). Written 2026-11-10 by
 * claude-code/Opus 5.5 for tpeulen.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/fit/InferenceFitting.h>
#include <IMP/bff/fit/Optimization.h>
#include <IMP/bff/bayesian/BayesianLaplace.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

IMPBFF_BEGIN_NAMESPACE

InferenceFit inference_fitting(const InferenceFactorGraph& graph,
                                     const std::vector<double>& theta0, int max_iterations,
                                     double tolerance, double line_search_below,
                                     const std::vector<std::string>& fixed) {
  if (!graph.get_is_evaluable()) {
    throw std::domain_error("inference_fitting: every factor needs a score");
  }
  if (max_iterations < 1 || !(tolerance > 0.0)) {
    throw std::invalid_argument("inference_fitting: max_iterations >= 1 and tolerance > 0");
  }
  const std::size_t dim = graph.get_parameter_dimension();
  if (theta0.size() != dim) {
    throw std::invalid_argument("inference_fitting: theta0 has " + std::to_string(theta0.size()) +
                                " values, the graph " + std::to_string(dim));
  }
  //: the free coordinates, in theta order; with nothing fixed every coordinate, so the arithmetic
  //: below is the one of the full fit (copies in the same order change no number)
  std::vector<char> held(dim, 0);
  for (const std::string& key : fixed) {
    const int at = graph.index_of(key);
    if (at < 0) throw std::invalid_argument("inference_fitting: no variable '" + key + "' to hold fixed");
    const int size = std::max(1, graph.get_variable_size(key));
    for (int k = 0; k < size; ++k) held[std::size_t(at + k)] = 1;
  }
  std::vector<std::size_t> F;
  for (std::size_t i = 0; i < dim; ++i)
    if (!held[i]) F.push_back(i);
  const std::size_t nf = F.size();
  if (nf == 0) throw std::invalid_argument("inference_fitting: every coordinate is fixed");

  std::vector<double> th = theta0;
  InferenceFit R;
  R.fixed = fixed;
  R.decrement = std::nan("");
  InferenceScore pt = graph.eval(th, true);
  std::vector<double> x(nf), g(nf), A(nf * nf);
  auto reduce = [&]() {
    for (std::size_t a = 0; a < nf; ++a) {
      x[a] = th[F[a]];
      g[a] = pt.gradient[F[a]];
      for (std::size_t b = 0; b < nf; ++b) A[a * nf + b] = pt.information[F[a] * dim + F[b]];
    }
  };
  reduce();
  //: the loop of bayesian_decay_fit_node, on the graph's sums over the free coordinates: A is the
  //: information, g the gradient of the log posterior, the objective minus the log posterior
  std::vector<double> trial = th;
  auto objective = [&](const double* t) {
    for (std::size_t a = 0; a < nf; ++a) trial[F[a]] = t[a];
    return -graph.eval(trial, false).log_density;
  };
  DampedNewton<decltype(objective)> stepper;
  stepper.line_search_below = line_search_below;
  int it = 0;
  for (; it < max_iterations; ++it) {
    const OptimizationStepResult r =
        stepper.step(A.data(), g.data(), nf, -pt.log_density, x.data(), objective);
    if (!r.accepted) { R.converged = std::isfinite(R.decrement) && R.decrement < 1e-3; break; }
    R.n_gradient_fallback += r.gradient_fallback ? 1 : 0;
    R.decrement = r.decrement;
    for (std::size_t a = 0; a < nf; ++a) th[F[a]] = x[a];
    pt = graph.eval(th, true);
    reduce();
    if (R.decrement < tolerance) { R.converged = true; ++it; break; }
  }
  if (!R.converged && std::isfinite(R.decrement) && R.decrement < 1e-3) R.converged = true;
  R.iterations = it;
  double ev = 0.0;
  R.laplace_ok = bayesian_laplace_log_evidence(pt.log_density, A.data(), nf, &ev);
  R.laplace_evidence = R.laplace_ok ? ev : -std::numeric_limits<double>::infinity();
  R.theta = th;
  R.at_mode = pt;
  return R;
}

IMPBFF_END_NAMESPACE

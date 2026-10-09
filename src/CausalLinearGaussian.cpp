/**
 *  \file CausalLinearGaussian.cpp
 *  \brief Linear-Gaussian structural causal model (see CausalLinearGaussian.h).
 *
 *  Every world is an affine map of the noise vector, X = A U + a. The factual
 *  world has A = (I - B)^-1. do(X_k = v) zeroes row k of B and the noise of k
 *  and puts v into the constant; pinning a mediator to its value in another
 *  world puts that world's (row of A, entry of a) in place of the noise. A
 *  world's Gaussian is then (A mu_U + a, A Sigma_U A^T) for whatever
 *  distribution U has -- the prior, or the abducted posterior of one unit.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/CausalLinearGaussian.h>
#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

IMPBFF_BEGIN_NAMESPACE

namespace {

struct AffineWorld {
  Eigen::MatrixXd A;  // X = A U + a
  Eigen::VectorXd a;
};

struct NoiseDistribution {
  Eigen::VectorXd mean;
  Eigen::MatrixXd cov;
};

CausalGaussianWorld to_world(const std::vector<std::string>& names, const AffineWorld& w,
                             const NoiseDistribution& u) {
  CausalGaussianWorld out;
  out.names = names;
  const Eigen::VectorXd m = w.A * u.mean + w.a;
  const Eigen::MatrixXd c = w.A * u.cov * w.A.transpose();
  out.mean.assign(m.data(), m.data() + m.size());
  out.covariance.resize(static_cast<std::size_t>(c.rows() * c.cols()));
  for (Eigen::Index i = 0; i < c.rows(); ++i) {
    for (Eigen::Index j = 0; j < c.cols(); ++j) {
      out.covariance[static_cast<std::size_t>(i * c.cols() + j)] = c(i, j);
    }
  }
  return out;
}

}  // namespace

// ------------------------------------------------------------------ world

double CausalGaussianWorld::get_mean(const std::string& name) const {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) IMP_THROW("no variable '" << name << "' in this world", IMP::ValueException);
  return mean[static_cast<std::size_t>(it - names.begin())];
}

double CausalGaussianWorld::get_sd(const std::string& name) const {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) IMP_THROW("no variable '" << name << "' in this world", IMP::ValueException);
  const std::size_t i = static_cast<std::size_t>(it - names.begin());
  return std::sqrt(std::max(0.0, covariance[i * names.size() + i]));
}

// ------------------------------------------------------------------ model

CausalLinearGaussian::CausalLinearGaussian() = default;

int CausalLinearGaussian::index_of(const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) IMP_THROW("no variable '" << name << "'", IMP::ValueException);
  return it->second;
}

void CausalLinearGaussian::add_variable(const std::string& name, double noise_mean,
                                        double noise_sd) {
  if (index_.count(name)) IMP_THROW("duplicate variable '" << name << "'", IMP::ValueException);
  if (!(noise_sd >= 0.0)) {
    IMP_THROW("variable '" << name << "': the noise width must be >= 0", IMP::ValueException);
  }
  index_[name] = static_cast<int>(names_.size());
  names_.push_back(name);
  noise_mean_.push_back(noise_mean);
  noise_sd_.push_back(noise_sd);
  parents_.push_back({});
}

void CausalLinearGaussian::add_edge(const std::string& parent, const std::string& child,
                                    double weight) {
  const int p = index_of(parent), c = index_of(child);
  if (p == c) IMP_THROW("an arrow from '" << parent << "' to itself is a cycle", IMP::ValueException);
  // A cycle would appear if child already reaches parent.
  std::vector<int> stack{c};
  std::set<int> seen{c};
  std::vector<std::vector<int> > children(names_.size());
  for (std::size_t i = 0; i < parents_.size(); ++i) {
    for (const auto& pw : parents_[i]) children[static_cast<std::size_t>(pw.first)].push_back(static_cast<int>(i));
  }
  while (!stack.empty()) {
    const int v = stack.back();
    stack.pop_back();
    if (v == p) {
      IMP_THROW("the arrow " << parent << " -> " << child << " closes a cycle", IMP::ValueException);
    }
    for (int w : children[static_cast<std::size_t>(v)]) {
      if (seen.insert(w).second) stack.push_back(w);
    }
  }
  for (auto& pw : parents_[static_cast<std::size_t>(c)]) {
    if (pw.first == p) {
      pw.second += weight;
      return;
    }
  }
  parents_[static_cast<std::size_t>(c)].push_back({p, weight});
}

unsigned int CausalLinearGaussian::get_number_of_variables() const {
  return static_cast<unsigned int>(names_.size());
}

std::vector<std::string> CausalLinearGaussian::get_variable_names() const { return names_; }

std::vector<std::string> CausalLinearGaussian::get_parents(const std::string& name) const {
  std::vector<std::string> out;
  for (const auto& pw : parents_[static_cast<std::size_t>(index_of(name))]) {
    out.push_back(names_[static_cast<std::size_t>(pw.first)]);
  }
  return out;
}

namespace {

// The world in which `fixed` variables equal fixed_values and `pinned`
// variables equal pin_rows * U + pin_const (a mediator taken from another
// world); every other variable keeps its equation and its noise.
AffineWorld make_world(std::size_t n,
                       const std::vector<std::vector<std::pair<int, double> > >& parents,
                       const std::vector<int>& fixed, const std::vector<double>& fixed_values,
                       const std::vector<int>& pinned = std::vector<int>(),
                       const Eigen::MatrixXd& pin_rows = Eigen::MatrixXd(),
                       const Eigen::VectorXd& pin_const = Eigen::VectorXd()) {
  const Eigen::Index N = static_cast<Eigen::Index>(n);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, N);
  for (std::size_t i = 0; i < n; ++i) {
    for (const auto& pw : parents[i]) B(static_cast<Eigen::Index>(i), pw.first) += pw.second;
  }
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(N, N);
  Eigen::VectorXd c = Eigen::VectorXd::Zero(N);
  for (std::size_t k = 0; k < fixed.size(); ++k) {
    B.row(fixed[k]).setZero();
    S(fixed[k], fixed[k]) = 0.0;
    c(fixed[k]) = fixed_values[k];
  }
  for (std::size_t k = 0; k < pinned.size(); ++k) {
    B.row(pinned[k]).setZero();
    S.row(pinned[k]) = pin_rows.row(static_cast<Eigen::Index>(k));
    c(pinned[k]) = pin_const(static_cast<Eigen::Index>(k));
  }
  const Eigen::MatrixXd M = (Eigen::MatrixXd::Identity(N, N) - B).inverse();
  return AffineWorld{M * S, M * c};
}

}  // namespace

CausalGaussianWorld CausalLinearGaussian::get_interventional(
    const std::vector<std::string>& do_names, const std::vector<double>& do_values) const {
  return get_counterfactual({}, {}, do_names, do_values);
}

CausalGaussianWorld CausalLinearGaussian::get_conditional(
    const std::vector<std::string>& obs_names, const std::vector<double>& obs_values) const {
  return get_counterfactual(obs_names, obs_values, {}, {});
}

namespace {

std::vector<int> indices_of(const std::vector<std::string>& names,
                            const std::map<std::string, int>& index, const char* what) {
  std::vector<int> out;
  std::set<int> seen;
  for (const auto& n : names) {
    const auto it = index.find(n);
    if (it == index.end()) IMP_THROW(what << ": no variable '" << n << "'", IMP::ValueException);
    if (!seen.insert(it->second).second) {
      IMP_THROW(what << ": '" << n << "' is named twice", IMP::ValueException);
    }
    out.push_back(it->second);
  }
  return out;
}

NoiseDistribution abduct(const AffineWorld& factual, const NoiseDistribution& prior,
                         const std::vector<int>& obs, const std::vector<double>& values) {
  if (obs.empty()) return prior;
  const Eigen::Index m = static_cast<Eigen::Index>(obs.size());
  const Eigen::Index n = prior.mean.size();
  Eigen::MatrixXd G(m, n);
  Eigen::VectorXd r(m);
  for (Eigen::Index k = 0; k < m; ++k) {
    G.row(k) = factual.A.row(obs[static_cast<std::size_t>(k)]);
    r(k) = values[static_cast<std::size_t>(k)] - factual.a(obs[static_cast<std::size_t>(k)]);
  }
  const Eigen::MatrixXd SGt = prior.cov * G.transpose();
  // Pseudo-inverse: redundant but consistent observations, or observations of
  // deterministic (zero-noise) variables, leave G Sigma G^T singular.
  const Eigen::MatrixXd Pinv =
      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(G * SGt).pseudoInverse();
  const Eigen::MatrixXd K = SGt * Pinv;
  NoiseDistribution post;
  post.mean = prior.mean + K * (r - G * prior.mean);
  post.cov = prior.cov - K * G * prior.cov;
  post.cov = 0.5 * (post.cov + post.cov.transpose());
  return post;
}

}  // namespace

CausalGaussianWorld CausalLinearGaussian::get_abducted_noise(
    const std::vector<std::string>& obs_names, const std::vector<double>& obs_values) const {
  if (obs_names.size() != obs_values.size()) {
    IMP_THROW("get_abducted_noise: " << obs_names.size() << " names and " << obs_values.size()
                                     << " values", IMP::ValueException);
  }
  const std::size_t n = names_.size();
  NoiseDistribution prior;
  prior.mean = Eigen::Map<const Eigen::VectorXd>(noise_mean_.data(), static_cast<Eigen::Index>(n));
  prior.cov = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) prior.cov(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = noise_sd_[i] * noise_sd_[i];
  const AffineWorld factual = make_world(n, parents_, {}, {});
  const NoiseDistribution u = abduct(factual, prior, indices_of(obs_names, index_, "observation"), obs_values);
  std::vector<std::string> noise_names;
  for (const auto& v : names_) noise_names.push_back("U[" + v + "]");
  AffineWorld identity{Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)),
                       Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n))};
  return to_world(noise_names, identity, u);
}

CausalGaussianWorld CausalLinearGaussian::get_counterfactual(
    const std::vector<std::string>& obs_names, const std::vector<double>& obs_values,
    const std::vector<std::string>& do_names, const std::vector<double>& do_values) const {
  if (obs_names.size() != obs_values.size() || do_names.size() != do_values.size()) {
    IMP_THROW("get_counterfactual: every named variable needs exactly one value", IMP::ValueException);
  }
  const std::size_t n = names_.size();
  NoiseDistribution prior;
  prior.mean = Eigen::Map<const Eigen::VectorXd>(noise_mean_.data(), static_cast<Eigen::Index>(n));
  prior.cov = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) prior.cov(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = noise_sd_[i] * noise_sd_[i];
  const AffineWorld factual = make_world(n, parents_, {}, {});
  const NoiseDistribution u =
      abduct(factual, prior, indices_of(obs_names, index_, "observation"), obs_values);  // abduction
  const AffineWorld world =
      make_world(n, parents_, indices_of(do_names, index_, "intervention"), do_values);  // action
  return to_world(names_, world, u);                                                     // prediction
}

std::vector<double> CausalLinearGaussian::get_natural_effects(
    const std::string& treatment, const std::vector<std::string>& mediators,
    const std::string& outcome, double t0, double t1,
    const std::vector<std::string>& obs_names, const std::vector<double>& obs_values) const {
  const std::size_t n = names_.size();
  const int t = index_of(treatment), y = index_of(outcome);
  const std::vector<int> med = indices_of(mediators, index_, "mediator");
  if (std::find(med.begin(), med.end(), t) != med.end() ||
      std::find(med.begin(), med.end(), y) != med.end()) {
    IMP_THROW("get_natural_effects: the treatment and the outcome cannot be mediators", IMP::ValueException);
  }
  if (obs_names.size() != obs_values.size()) {
    IMP_THROW("get_natural_effects: every observed variable needs exactly one value", IMP::ValueException);
  }
  NoiseDistribution prior;
  prior.mean = Eigen::Map<const Eigen::VectorXd>(noise_mean_.data(), static_cast<Eigen::Index>(n));
  prior.cov = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) prior.cov(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = noise_sd_[i] * noise_sd_[i];
  const NoiseDistribution u = abduct(make_world(n, parents_, {}, {}), prior,
                                     indices_of(obs_names, index_, "observation"), obs_values);
  const AffineWorld w0 = make_world(n, parents_, {t}, {t0});
  const AffineWorld w1 = make_world(n, parents_, {t}, {t1});
  // Y(t1, M(t0)): the treatment at t1, every mediator at its value in world t0
  // -- the same unit's noise, read through world t0's equations.
  Eigen::MatrixXd pin_rows(static_cast<Eigen::Index>(med.size()), static_cast<Eigen::Index>(n));
  Eigen::VectorXd pin_const(static_cast<Eigen::Index>(med.size()));
  for (std::size_t k = 0; k < med.size(); ++k) {
    pin_rows.row(static_cast<Eigen::Index>(k)) = w0.A.row(med[k]);
    pin_const(static_cast<Eigen::Index>(k)) = w0.a(med[k]);
  }
  const AffineWorld wx = make_world(n, parents_, {t}, {t1}, med, pin_rows, pin_const);
  auto expect = [&](const AffineWorld& w) { return w.A.row(y).dot(u.mean) + w.a(y); };
  const double e0 = expect(w0), e1 = expect(w1), ex = expect(wx);
  return {e1 - e0, ex - e0, e1 - ex};
}

std::vector<double> CausalLinearGaussian::get_samples(unsigned int n_samples, unsigned int seed,
                                                      const std::vector<std::string>& do_names,
                                                      const std::vector<double>& do_values) const {
  if (do_names.size() != do_values.size()) {
    IMP_THROW("get_samples: every intervened variable needs exactly one value", IMP::ValueException);
  }
  const std::size_t n = names_.size();
  const AffineWorld w = make_world(n, parents_, indices_of(do_names, index_, "intervention"), do_values);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<double> out(static_cast<std::size_t>(n_samples) * n);
  Eigen::VectorXd u(static_cast<Eigen::Index>(n));
  for (unsigned int s = 0; s < n_samples; ++s) {
    for (std::size_t i = 0; i < n; ++i) u(static_cast<Eigen::Index>(i)) = noise_mean_[i] + noise_sd_[i] * z(rng);
    const Eigen::VectorXd x = w.A * u + w.a;
    for (std::size_t i = 0; i < n; ++i) out[s * n + i] = x(static_cast<Eigen::Index>(i));
  }
  return out;
}

IMPBFF_END_NAMESPACE

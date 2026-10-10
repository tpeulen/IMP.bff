/**
 *  \file CounterfactualDistanceNetwork.cpp
 *  \brief Site-bias abduction, interventions and replay for a distance network
 *         (see CounterfactualDistanceNetwork.h).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/CounterfactualDistanceNetwork.h>
#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

IMPBFF_BEGIN_NAMESPACE

namespace {

std::vector<double> normalised_from_log(const std::vector<double>& lp) {
  const double top = *std::max_element(lp.begin(), lp.end());
  std::vector<double> w(lp.size());
  double total = 0.0;
  for (std::size_t i = 0; i < lp.size(); ++i) {
    w[i] = std::isfinite(lp[i]) ? std::exp(lp[i] - top) : 0.0;
    total += w[i];
  }
  for (auto& v : w) v /= total;
  return w;
}

int argmax(const std::vector<double>& w) {
  return static_cast<int>(std::max_element(w.begin(), w.end()) - w.begin());
}

}  // namespace

CounterfactualDistanceNetwork::CounterfactualDistanceNetwork(const std::vector<double>& model,
                                                             int n_candidates,
                                                             const std::vector<int>& pair_sites,
                                                             int n_sites)
    : n_candidates_(n_candidates), n_pairs_(0), n_sites_(n_sites), model_(model),
      pair_sites_(pair_sites) {
  if (n_candidates < 1 || n_sites < 1) {
    IMP_THROW("CounterfactualDistanceNetwork: need at least one candidate and one site",
              IMP::ValueException);
  }
  if (pair_sites.size() % 2 != 0 || pair_sites.empty()) {
    IMP_THROW("CounterfactualDistanceNetwork: pair_sites holds two site indices per pair",
              IMP::ValueException);
  }
  n_pairs_ = static_cast<int>(pair_sites.size() / 2);
  if (model.size() != static_cast<std::size_t>(n_candidates) * static_cast<std::size_t>(n_pairs_)) {
    IMP_THROW("CounterfactualDistanceNetwork: the model table has " << model.size()
                  << " entries, expected " << n_candidates << " x " << n_pairs_,
              IMP::ValueException);
  }
  for (int s : pair_sites) {
    if (s < 0 || s >= n_sites) {
      IMP_THROW("CounterfactualDistanceNetwork: site index " << s << " outside [0, " << n_sites << ")",
                IMP::ValueException);
    }
  }
  tau_.assign(static_cast<std::size_t>(n_sites), 0.0);
  prior_.assign(static_cast<std::size_t>(n_candidates), 1.0 / n_candidates);
}

int CounterfactualDistanceNetwork::get_number_of_candidates() const { return n_candidates_; }
int CounterfactualDistanceNetwork::get_number_of_pairs() const { return n_pairs_; }
int CounterfactualDistanceNetwork::get_number_of_sites() const { return n_sites_; }

void CounterfactualDistanceNetwork::set_measurement(const std::vector<double>& values,
                                                    const std::vector<double>& sigma) {
  if (values.size() != static_cast<std::size_t>(n_pairs_) || sigma.size() != values.size()) {
    IMP_THROW("set_measurement: need one value and one error bar per pair (" << n_pairs_ << ")",
              IMP::ValueException);
  }
  for (double s : sigma) {
    if (!(s > 0.0)) IMP_THROW("set_measurement: error bars must be > 0", IMP::ValueException);
  }
  y_ = values;
  sigma_ = sigma;
}

void CounterfactualDistanceNetwork::set_site_bias_prior(const std::vector<double>& tau) {
  if (tau.size() != static_cast<std::size_t>(n_sites_)) {
    IMP_THROW("set_site_bias_prior: need one width per site (" << n_sites_ << ")", IMP::ValueException);
  }
  for (double t : tau) {
    if (!(t >= 0.0)) IMP_THROW("set_site_bias_prior: widths must be >= 0", IMP::ValueException);
  }
  tau_ = tau;
}

void CounterfactualDistanceNetwork::set_candidate_prior(const std::vector<double>& prior) {
  if (prior.size() != static_cast<std::size_t>(n_candidates_)) {
    IMP_THROW("set_candidate_prior: need one probability per candidate", IMP::ValueException);
  }
  double total = 0.0;
  for (double p : prior) {
    if (!(p >= 0.0)) IMP_THROW("set_candidate_prior: probabilities must be >= 0", IMP::ValueException);
    total += p;
  }
  if (!(total > 0.0)) IMP_THROW("set_candidate_prior: all zero", IMP::ValueException);
  prior_ = prior;
  for (auto& p : prior_) p /= total;
}

void CounterfactualDistanceNetwork::check_ready(const char* what) const {
  if (y_.empty()) IMP_THROW(what << ": call set_measurement first", IMP::ValueException);
}

// ------------------------------------------------------------ the usual analysis

std::vector<double> CounterfactualDistanceNetwork::get_standard_posterior(
    const std::vector<double>& values) const {
  if (values.size() != static_cast<std::size_t>(n_pairs_)) {
    IMP_THROW("get_standard_posterior: need one value per pair", IMP::ValueException);
  }
  if (sigma_.empty()) IMP_THROW("get_standard_posterior: call set_measurement first", IMP::ValueException);
  std::vector<double> lp(static_cast<std::size_t>(n_candidates_));
  for (int f = 0; f < n_candidates_; ++f) {
    double chi2 = 0.0;
    for (int p = 0; p < n_pairs_; ++p) {
      const double r = (values[static_cast<std::size_t>(p)] -
                        model_[static_cast<std::size_t>(f) * n_pairs_ + p]) /
                       sigma_[static_cast<std::size_t>(p)];
      chi2 += r * r;
    }
    lp[static_cast<std::size_t>(f)] = std::log(prior_[static_cast<std::size_t>(f)]) - 0.5 * chi2;
  }
  return normalised_from_log(lp);
}

std::vector<double> CounterfactualDistanceNetwork::get_chi2() const {
  check_ready("get_chi2");
  std::vector<double> out(static_cast<std::size_t>(n_candidates_));
  for (int f = 0; f < n_candidates_; ++f) {
    double chi2 = 0.0;
    for (int p = 0; p < n_pairs_; ++p) {
      const double r = (y_[static_cast<std::size_t>(p)] - model_[static_cast<std::size_t>(f) * n_pairs_ + p]) /
                       sigma_[static_cast<std::size_t>(p)];
      chi2 += r * r;
    }
    out[static_cast<std::size_t>(f)] = chi2;
  }
  return out;
}

// ------------------------------------------------------------ biases integrated out

namespace {

struct BiasAlgebra {
  Eigen::MatrixXd A;     // (pairs, sites) incidence
  Eigen::MatrixXd Sinv;  // (diag sigma^2 + A T A^T)^-1
  Eigen::MatrixXd TAt;   // T A^T, (sites, pairs)
  Eigen::MatrixXd post_cov;  // Var[b | y, f], the same for every f
};

BiasAlgebra bias_algebra(int n_pairs, int n_sites, const std::vector<int>& pair_sites,
                         const std::vector<double>& sigma, const std::vector<double>& tau) {
  BiasAlgebra out;
  out.A = Eigen::MatrixXd::Zero(n_pairs, n_sites);
  for (int p = 0; p < n_pairs; ++p) {
    out.A(p, pair_sites[static_cast<std::size_t>(2 * p)]) += 1.0;
    out.A(p, pair_sites[static_cast<std::size_t>(2 * p + 1)]) += 1.0;
  }
  Eigen::VectorXd t2(n_sites);
  for (int s = 0; s < n_sites; ++s) t2(s) = tau[static_cast<std::size_t>(s)] * tau[static_cast<std::size_t>(s)];
  Eigen::MatrixXd S = out.A * t2.asDiagonal() * out.A.transpose();
  for (int p = 0; p < n_pairs; ++p) S(p, p) += sigma[static_cast<std::size_t>(p)] * sigma[static_cast<std::size_t>(p)];
  out.Sinv = S.ldlt().solve(Eigen::MatrixXd::Identity(n_pairs, n_pairs));
  out.TAt = t2.asDiagonal() * out.A.transpose();
  Eigen::MatrixXd T = t2.asDiagonal();
  out.post_cov = T - out.TAt * out.Sinv * out.TAt.transpose();
  return out;
}

}  // namespace

std::vector<double> CounterfactualDistanceNetwork::get_candidate_posterior(bool bias_aware) const {
  check_ready("get_candidate_posterior");
  if (!bias_aware) return get_standard_posterior(y_);
  const BiasAlgebra alg = bias_algebra(n_pairs_, n_sites_, pair_sites_, sigma_, tau_);
  const Eigen::Map<const Eigen::VectorXd> y(y_.data(), n_pairs_);
  std::vector<double> lp(static_cast<std::size_t>(n_candidates_));
  for (int f = 0; f < n_candidates_; ++f) {
    const Eigen::Map<const Eigen::VectorXd> m(model_.data() + static_cast<std::size_t>(f) * n_pairs_, n_pairs_);
    const Eigen::VectorXd r = y - m;
    // log|S| is the same for every candidate and cancels.
    lp[static_cast<std::size_t>(f)] = std::log(prior_[static_cast<std::size_t>(f)]) - 0.5 * r.dot(alg.Sinv * r);
  }
  return normalised_from_log(lp);
}

std::vector<double> CounterfactualDistanceNetwork::get_site_biases_given_candidates() const {
  check_ready("get_site_biases_given_candidates");
  const BiasAlgebra alg = bias_algebra(n_pairs_, n_sites_, pair_sites_, sigma_, tau_);
  const Eigen::Map<const Eigen::VectorXd> y(y_.data(), n_pairs_);
  std::vector<double> out(static_cast<std::size_t>(n_candidates_) * n_sites_);
  for (int f = 0; f < n_candidates_; ++f) {
    const Eigen::Map<const Eigen::VectorXd> m(model_.data() + static_cast<std::size_t>(f) * n_pairs_, n_pairs_);
    const Eigen::VectorXd b = alg.TAt * (alg.Sinv * (y - m));
    for (int s = 0; s < n_sites_; ++s) out[static_cast<std::size_t>(f) * n_sites_ + s] = b(s);
  }
  return out;
}

std::vector<double> CounterfactualDistanceNetwork::get_abducted_site_biases() const {
  const std::vector<double> w = get_candidate_posterior(true);
  const std::vector<double> bf = get_site_biases_given_candidates();
  std::vector<double> b(static_cast<std::size_t>(n_sites_), 0.0);
  for (int f = 0; f < n_candidates_; ++f) {
    for (int s = 0; s < n_sites_; ++s) {
      b[static_cast<std::size_t>(s)] += w[static_cast<std::size_t>(f)] * bf[static_cast<std::size_t>(f) * n_sites_ + s];
    }
  }
  return b;
}

std::vector<double> CounterfactualDistanceNetwork::get_abducted_site_bias_sd() const {
  const std::vector<double> w = get_candidate_posterior(true);
  const std::vector<double> bf = get_site_biases_given_candidates();
  const std::vector<double> mean = get_abducted_site_biases();
  const BiasAlgebra alg = bias_algebra(n_pairs_, n_sites_, pair_sites_, sigma_, tau_);
  std::vector<double> sd(static_cast<std::size_t>(n_sites_));
  for (int s = 0; s < n_sites_; ++s) {
    double var = alg.post_cov(s, s);
    for (int f = 0; f < n_candidates_; ++f) {
      const double d = bf[static_cast<std::size_t>(f) * n_sites_ + s] - mean[static_cast<std::size_t>(s)];
      var += w[static_cast<std::size_t>(f)] * d * d;
    }
    sd[static_cast<std::size_t>(s)] = std::sqrt(std::max(0.0, var));
  }
  return sd;
}

std::vector<double> CounterfactualDistanceNetwork::get_abducted_noise() const {
  const int f = argmax(get_candidate_posterior(true));
  const std::vector<double> bf = get_site_biases_given_candidates();
  std::vector<double> eps(static_cast<std::size_t>(n_pairs_));
  for (int p = 0; p < n_pairs_; ++p) {
    const int i = pair_sites_[static_cast<std::size_t>(2 * p)], j = pair_sites_[static_cast<std::size_t>(2 * p + 1)];
    eps[static_cast<std::size_t>(p)] = y_[static_cast<std::size_t>(p)] - model_[static_cast<std::size_t>(f) * n_pairs_ + p] -
                                       bf[static_cast<std::size_t>(f) * n_sites_ + i] -
                                       bf[static_cast<std::size_t>(f) * n_sites_ + j];
  }
  return eps;
}

// ------------------------------------------------------------ interventions

std::vector<double> CounterfactualDistanceNetwork::get_ideal_measurement() const {
  const std::vector<double> b = get_abducted_site_biases();
  std::vector<double> out = y_;
  for (int p = 0; p < n_pairs_; ++p) {
    out[static_cast<std::size_t>(p)] -= b[static_cast<std::size_t>(pair_sites_[static_cast<std::size_t>(2 * p)])] +
                                        b[static_cast<std::size_t>(pair_sites_[static_cast<std::size_t>(2 * p + 1)])];
  }
  return out;
}

std::vector<double> CounterfactualDistanceNetwork::get_hinge_posteriors() const {
  const std::vector<double> b = get_abducted_site_biases();
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(n_sites_) * n_candidates_);
  for (int s = 0; s < n_sites_; ++s) {
    std::vector<double> ys = y_;
    for (int p = 0; p < n_pairs_; ++p) {
      const int uses = (pair_sites_[static_cast<std::size_t>(2 * p)] == s) +
                       (pair_sites_[static_cast<std::size_t>(2 * p + 1)] == s);
      ys[static_cast<std::size_t>(p)] -= uses * b[static_cast<std::size_t>(s)];
    }
    const std::vector<double> w = get_standard_posterior(ys);
    out.insert(out.end(), w.begin(), w.end());
  }
  return out;
}

std::vector<int> CounterfactualDistanceNetwork::get_counterfactual_replay() const {
  const int f_hat = argmax(get_candidate_posterior(true));
  const std::vector<double> bf = get_site_biases_given_candidates();
  const std::vector<double> eps = get_abducted_noise();
  std::vector<int> out(static_cast<std::size_t>(n_candidates_));
  std::vector<double> yv(static_cast<std::size_t>(n_pairs_));
  for (int f = 0; f < n_candidates_; ++f) {
    for (int p = 0; p < n_pairs_; ++p) {
      const int i = pair_sites_[static_cast<std::size_t>(2 * p)], j = pair_sites_[static_cast<std::size_t>(2 * p + 1)];
      yv[static_cast<std::size_t>(p)] = model_[static_cast<std::size_t>(f) * n_pairs_ + p] +
                                        bf[static_cast<std::size_t>(f_hat) * n_sites_ + i] +
                                        bf[static_cast<std::size_t>(f_hat) * n_sites_ + j] +
                                        eps[static_cast<std::size_t>(p)];
    }
    out[static_cast<std::size_t>(f)] = argmax(get_standard_posterior(yv));
  }
  return out;
}

std::vector<int> CounterfactualDistanceNetwork::get_fresh_noise_replay(unsigned int seed) const {
  check_ready("get_fresh_noise_replay");
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> z(0.0, 1.0);
  std::vector<int> out(static_cast<std::size_t>(n_candidates_));
  std::vector<double> yv(static_cast<std::size_t>(n_pairs_));
  for (int f = 0; f < n_candidates_; ++f) {
    for (int p = 0; p < n_pairs_; ++p) {
      yv[static_cast<std::size_t>(p)] = model_[static_cast<std::size_t>(f) * n_pairs_ + p] +
                                        sigma_[static_cast<std::size_t>(p)] * z(rng);
    }
    out[static_cast<std::size_t>(f)] = argmax(get_standard_posterior(yv));
  }
  return out;
}

IMPBFF_END_NAMESPACE

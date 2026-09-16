/**
 * \file IMP/bff/BayesianLaplace.h
 * \brief The Gaussian approximation at a posterior mode, its evidence, and a
 *        grid of them mixed by that evidence.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_BAYESIANLAPLACE_H
#define IMPBFF_BAYESIANLAPLACE_H

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/Optimization.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name The Laplace approximation
//! @{

/**
 * \brief `log det A` for a symmetric positive-definite `A`, by Cholesky.
 *
 * Returns false when `A` is not positive definite, which at a mode means the
 * point is not one. Computing the determinant as a sum of logarithms rather
 * than the logarithm of a product is not fastidiousness: the determinant of a
 * hundred-dimensional information matrix overflows a double long before its
 * logarithm is large.
 */
inline bool bayesian_log_det_spd(const double* A, std::size_t n, double* out) {
  return log_det_spd(A, n, out);   // tttrlib's, through Optimization.h: the one Cholesky
}

/**
 * \brief `log p(y)` from a mode and the curvature there.
 *
 * Expand `log p(y, theta)` to second order about its maximum and integrate the
 * Gaussian that results:
 *
 *     log p(y) ~ log p(y, theta_hat) + (d/2) log(2 pi) - (1/2) log det A
 *
 * with `A = -d2 log p / dtheta2` at the mode. This is the marginal likelihood
 * -- the number that compares models, or, here, the nodes of a hyperparameter
 * grid -- and the `-(1/2) log det A` is the whole content of Occam's razor in
 * this approximation: a model whose posterior is sharply peaked in many
 * directions pays for each of them.
 *
 * \param log_post_at_mode `log p(y, theta_hat)`, unnormalised in `theta`
 * \param A minus the curvature at the mode, `n x n` row-major
 */
inline bool bayesian_laplace_log_evidence(double log_post_at_mode, const double* A,
                                 std::size_t n, double* out) {
  double ld = 0.0;
  if (!bayesian_log_det_spd(A, n, &ld)) return false;
  *out = log_post_at_mode + 0.5 * double(n) * std::log(2.0 * M_PI) - 0.5 * ld;
  return true;
}

/**
 * \brief The smallest eigenvalue of a symmetric positive-definite `A` and its direction, by inverse
 *        iteration on the Cholesky factor (Golub & Van Loan, *Matrix Computations* 4th ed., 8.2.2).
 *
 * **Why a posterior's flattest direction is worth a function of its own.** A Laplace approximation is a
 * Gaussian fitted at a mode, and where the precision has a near-zero eigenvalue the Gaussian's width in
 * that direction is set by whatever holds the coefficient there -- a smoothing prior, say -- and not by the
 * data. Any quantity that depends on the approximation's VOLUME (an evidence, a ratio of two Laplace
 * integrals) is then dominated by a direction in which the posterior is nowhere near Gaussian. Measuring
 * that eigenvalue is how such a quantity can say so instead of being quoted.
 *
 * Returns false when `A` is not positive definite. `v` (length `n`) receives the unit eigenvector.
 */
inline bool bayesian_smallest_eigenpair(const double* A, std::size_t n, double* v, double* lambda_min,
                                        int iterations = 60) {
  CholeskyFactor ch;
  if (!ch.factor(A, n)) return false;
  std::vector<double> x(n, 0.0), y(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) x[i] = 1.0 / std::sqrt(double(n));
  double lam = 0.0;
  for (int it = 0; it < iterations; ++it) {
    if (!ch.solve(x.data(), y.data())) return false;      //: y = A^-1 x, so y grows along the flattest axis
    double nrm = 0.0;
    for (std::size_t i = 0; i < n; ++i) nrm += y[i] * y[i];
    nrm = std::sqrt(nrm);
    if (!(nrm > 0.0) || !std::isfinite(nrm)) return false;
    for (std::size_t i = 0; i < n; ++i) x[i] = y[i] / nrm;
    //: the Rayleigh quotient x' A x, which is the eigenvalue once x has settled
    lam = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::size_t j = 0; j < n; ++j) s += A[i * n + j] * x[j];
      lam += x[i] * s;
    }
  }
  for (std::size_t i = 0; i < n; ++i) v[i] = x[i];
  *lambda_min = lam;
  return true;
}

//! The mean and standard deviation of a positive summary by Tierney & Kadane's ratio.
struct BayesianTierneyKadane {
  double mean = std::numeric_limits<double>::quiet_NaN();
  double sd = std::numeric_limits<double>::quiet_NaN();
  bool ok = false;
};

/**
 * \brief Tierney & Kadane (1986) moments of a positive summary, from two tilted Laplace ratios.
 *
 * The posterior mean of a positive `f(theta)` is a ratio of two integrals,
 * `E[f] = int f(theta) p(y, theta) dtheta / int p(y, theta) dtheta`, and both can be
 * approximated by Laplace -- the numerator at the mode of `log p + log f`, the denominator at
 * the mode of `log p`. Taking the ratio cancels the leading error terms, which is why the
 * result is accurate to `O(n^-2)` where the delta method is `O(n^-1)` (Tierney & Kadane,
 * *J. Amer. Statist. Assoc.* 81:82, 1986, the "fully exponential" approximation).
 *
 * The caller does the two tilted fits with the tilt `power * log(f + eps)` (`eps` keeps the
 * logarithm finite where `f` can reach zero) and passes, for `power` 1 and 2,
 *
 *     log_ratio = [log p(y, theta_tilted) + power * log(f(theta_tilted) + eps)]
 *                 - log p(y, theta_mode) - 0.5 * (log det H_tilted - log det H_mode),
 *
 * so that `exp(log_ratio_1) = E[f + eps]` and `exp(log_ratio_2) = E[(f + eps)^2]`. The
 * `eps` cancels out of the variance exactly: `Var[f] = m2 - m1^2`.
 *
 * **The two curvatures have to be consistent.** The ratio needs the log-determinants to
 * about `1e-4` nats, and two independent scoring runs do not deliver that -- in ucfret's
 * prototype (`s88_laplace_posterior.py`, 2026-09-06) a tilted mode that sat exactly on the
 * untilted one still differed by 0.026 nats beyond `log f`, which moved a mean by 2.6 %,
 * three of its standard deviations. So `H_tilted` is built as the mode's own matrix plus the
 * exact Hessian of the tilt at the tilted mode, never as a second independent estimate of
 * the whole curvature.
 */
inline BayesianTierneyKadane bayesian_tierney_kadane(double log_ratio1, double log_ratio2, double eps) {
  BayesianTierneyKadane r;
  if (!std::isfinite(log_ratio1) || !std::isfinite(log_ratio2)) return r;
  const double m1 = std::exp(log_ratio1), m2 = std::exp(log_ratio2);
  const double var = m2 - m1 * m1;
  if (!std::isfinite(m1) || !std::isfinite(m2)) return r;
  r.mean = m1 - eps;
  r.sd = std::sqrt(var > 0.0 ? var : 0.0);
  r.ok = true;
  return r;
}

/**
 * \brief A hyperparameter grid, its nodes weighted by their evidence.
 *
 * **Why a grid and not a joint mode.** A smoothing parameter cannot simply be
 * maximised over jointly with what it smooths: the penalty's own normaliser
 * rewards a large weight whenever the coefficients are smooth, so the joint
 * maximum sits at the smoothest possible answer -- no structure at all. The
 * remedy is to treat it as what it is, a parameter to be integrated out:
 * approximate the posterior at each node of a grid, weight the nodes by their
 * evidence, and report the mixture. That is the integrated-nested-Laplace
 * construction (Rue, Martino & Chopin, *J. R. Statist. Soc. B* 71:319, 2009;
 * Wood, *Biometrika* 98:53, 2011, for smoothing parameters specifically), and
 * it needs no sampler: the weights are evidences and the components are
 * Gaussians.
 *
 * **What this class does with them.** For any scalar summary the caller can
 * evaluate at each node -- a distribution's mean, one of its bins, an
 * amplitude -- it combines the per-node mean and variance by the law of total
 * variance:
 *
 *     E[g]   = sum_i w_i m_i
 *     Var[g] = sum_i w_i (v_i + m_i^2) - E[g]^2
 *
 * The second term is what a single node cannot give: the spread BETWEEN nodes,
 * which is the uncertainty in the hyperparameter itself. Reporting the best
 * node's interval instead is the common mistake, and it is too narrow by
 * exactly that term.
 *
 * Weights are formed by subtracting the largest log evidence before
 * exponentiating, because these are log evidences of real data sets and
 * differences of tens of thousands of nats are ordinary.
 */
class BayesianEvidenceMixture {
 public:
  //! Add one node: its log evidence and a summary's mean and variance there.
  void add(double log_evidence, double mean, double variance) {
    lz_.push_back(log_evidence);
    m_.push_back(mean);
    v_.push_back(variance);
  }
  //! Add one node whose summary is a vector; all nodes must agree in length.
  void add_vector(double log_evidence, const std::vector<double>& mean,
                  const std::vector<double>& variance) {
    lz_.push_back(log_evidence);
    mv_.push_back(mean);
    vv_.push_back(variance);
  }

  std::size_t size() const { return lz_.size(); }

  //! Normalised weights, `exp(lz - max)` renormalised.
  std::vector<double> weights() const {
    std::vector<double> w(lz_.size(), 0.0);
    if (lz_.empty()) return w;
    double mx = -std::numeric_limits<double>::infinity();
    for (double z : lz_) if (std::isfinite(z) && z > mx) mx = z;
    double s = 0.0;
    for (std::size_t i = 0; i < lz_.size(); ++i) {
      w[i] = std::isfinite(lz_[i]) ? std::exp(lz_[i] - mx) : 0.0;
      s += w[i];
    }
    if (s > 0.0) for (double& x : w) x /= s;
    return w;
  }

  //! The mixture's mean and standard deviation of a scalar summary.
  void moments(double* mean, double* sd) const {
    const auto w = weights();
    double m = 0.0, s2 = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) m += w[i] * m_[i];
    for (std::size_t i = 0; i < w.size(); ++i) s2 += w[i] * (v_[i] + m_[i] * m_[i]);
    s2 -= m * m;
    *mean = m;
    *sd = std::sqrt(s2 > 0.0 ? s2 : 0.0);
  }

  //! The same, elementwise, for a vector summary.
  void moments_vector(std::vector<double>* mean, std::vector<double>* sd) const {
    const auto w = weights();
    const std::size_t n = mv_.empty() ? 0 : mv_[0].size();
    mean->assign(n, 0.0);
    sd->assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      double m = 0.0, s2 = 0.0;
      for (std::size_t i = 0; i < w.size(); ++i) m += w[i] * mv_[i][j];
      for (std::size_t i = 0; i < w.size(); ++i) s2 += w[i] * (vv_[i][j] + mv_[i][j] * mv_[i][j]);
      s2 -= m * m;
      (*mean)[j] = m;
      (*sd)[j] = std::sqrt(s2 > 0.0 ? s2 : 0.0);
    }
  }

  /**
   * \brief The mixture's CDF of a scalar summary: `F(t) = sum_i w_i Phi((t - mu_i) / sd_i)`.
   *
   * The nodes' Laplace approximations are normals, so the mixture over the hyperparameter is a mixture
   * of normals and its CDF is analytic -- no draws. A node of zero variance contributes a step.
   */
  double cdf(double t) const {
    const auto w = weights();
    double F = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
      if (w[i] <= 0.0) continue;
      const double sd = std::sqrt(v_[i] > 0.0 ? v_[i] : 0.0);
      F += w[i] * (sd > 0.0 ? 0.5 * std::erfc(-(t - m_[i]) / (sd * 1.4142135623730951)) : (t >= m_[i] ? 1.0 : 0.0));
    }
    return F;
  }

  /**
   * \brief The quantile of that mixture: `F(t) = p`, by bisection.
   *
   * A mixture's quantile is not the mixture of the nodes' quantiles, which is why a credible interval
   * over a hyperparameter grid needs this rather than the moments alone. The bracket starts at the
   * extreme nodes' means +- 40 sd and halves 200 times (about 1e-58 of the bracket).
   */
  double quantile(double p) const {
    const auto w = weights();
    double lo = 0.0, hi = 0.0;
    bool any = false;
    for (std::size_t i = 0; i < w.size(); ++i) {
      if (w[i] <= 0.0) continue;
      const double sd = std::sqrt(v_[i] > 0.0 ? v_[i] : 0.0);
      lo = any ? std::min(lo, m_[i] - 40.0 * sd) : m_[i] - 40.0 * sd;
      hi = any ? std::max(hi, m_[i] + 40.0 * sd) : m_[i] + 40.0 * sd;
      any = true;
    }
    if (!any) return std::numeric_limits<double>::quiet_NaN();
    if (!(p > 0.0)) return lo;
    if (!(p < 1.0)) return hi;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      (cdf(mid) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
  }

  //! The same quantile elementwise for a vector summary (p(R/R0) bin by bin).
  std::vector<double> quantile_vector(double p) const {
    const std::size_t n = mv_.empty() ? 0 : mv_[0].size();
    const auto w = weights();
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      BayesianEvidenceMixture one;
      for (std::size_t i = 0; i < w.size(); ++i) one.add(lz_[i], mv_[i][j], vv_[i][j]);
      out[j] = one.quantile(p);
    }
    return out;
  }

 private:
  std::vector<double> lz_, m_, v_;
  std::vector<std::vector<double>> mv_, vv_;
};

//! @}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_BAYESIANLAPLACE_H

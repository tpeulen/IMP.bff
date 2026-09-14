/**
 * \file IMP/bff/FitStatistics.h
 * \brief Whether a fit is good: the deviance a count model earns, and whether
 *        its residuals are structured.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FITSTATISTICS_H
#define IMPBFF_FITSTATISTICS_H

#include <IMP/bff/IMPCompatibility.h>
#include <cmath>
#include <random>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name Fit statistics
//! @{

/**
 * \brief Poisson deviance, `2 sum [ m - y + y log(y/m) ]`.
 *
 * **Why not chi-square.** For counting data the natural goodness-of-fit
 * statistic is the likelihood-ratio one, and dividing by an estimated variance
 * is a Gaussian approximation that fails where it matters -- in the tail,
 * where the counts are few and the long lifetimes live. Weighting by the
 * OBSERVED counts (`sigma = sqrt(y)`, Neyman) biases the fit low there;
 * weighting by the model (Pearson) does not, but is not the likelihood. The
 * deviance is the likelihood ratio against a model that fits every bin
 * exactly, so it needs no weights at all, and it is what "chi-square" should
 * mean for photon counting.
 *
 * A bin with `y = 0` contributes `2m`, which is the limit of `y log(y/m)` as
 * `y -> 0` and not a special case to be skipped.
 */
inline double poisson_deviance(const double* y, const double* m, std::size_t n) {
  double d = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double mi = (m[i] > 1e-300) ? m[i] : 1e-300;
    d += 2.0 * (mi - y[i]);
    if (y[i] > 0.0) d += 2.0 * y[i] * std::log(y[i] / mi);
  }
  return d;
}

//! Signed square roots of the per-bin deviance -- residuals whose sum of
//! squares IS the deviance, unlike `(y - m)/sqrt(y)`.
inline void deviance_residuals(const double* y, const double* m, std::size_t n, double* r) {
  for (std::size_t i = 0; i < n; ++i) {
    const double mi = (m[i] > 1e-300) ? m[i] : 1e-300;
    double d = 2.0 * (mi - y[i]);
    if (y[i] > 0.0) d += 2.0 * y[i] * std::log(y[i] / mi);
    if (d < 0.0) d = 0.0;
    r[i] = (y[i] >= mi ? 1.0 : -1.0) * std::sqrt(d);
  }
}

//! The outcome of a Wald-Wolfowitz runs test.
struct RunsTest {
  std::size_t n_runs = 0;      //!< runs of equal sign
  std::size_t n_above = 0;     //!< values above the cutoff
  std::size_t n_below = 0;     //!< values below it
  double z = 0.0;              //!< standardised deviation from the expected count
  double p_value = 1.0;        //!< two-sided
};

/**
 * \brief Are the residuals arranged in runs, or do their signs alternate like
 *        coin flips?
 *
 * **What it catches that a fit statistic does not.** A deviance per degree of
 * freedom near one says the residuals are the right SIZE. It says nothing
 * about their ORDER, and a model that is systematically low over one stretch
 * of the axis and high over another can have an excellent deviance. The runs
 * test asks whether the number of sign changes is what independent noise would
 * give: too few runs means the residuals are correlated along the axis, which
 * is what a missing component or a wrong instrument response looks like.
 *
 * Under the null the number of runs `R` among `n1` values above and `n2` below
 * has mean `2 n1 n2 / (n1 + n2) + 1` and variance
 * `2 n1 n2 (2 n1 n2 - n1 - n2) / ((n1+n2)^2 (n1+n2-1))`, and `R` is
 * asymptotically normal (Wald & Wolfowitz, *Ann. Math. Statist.* 11:147,
 * 1940).
 *
 * Values exactly at the cutoff are counted as above, which is the convention
 * `statsmodels.sandbox.stats.runs.runstest_1samp` uses; pass the residuals
 * themselves with `cutoff = 0` for the usual sign test.
 */
inline RunsTest runs_test(const double* x, std::size_t n, double cutoff = 0.0,
                          bool continuity_correction = false) {
  RunsTest r;
  if (n == 0) return r;
  std::vector<bool> above(n);
  for (std::size_t i = 0; i < n; ++i) {
    above[i] = (x[i] >= cutoff);
    if (above[i]) ++r.n_above; else ++r.n_below;
  }
  r.n_runs = 1;
  for (std::size_t i = 1; i < n; ++i) if (above[i] != above[i - 1]) ++r.n_runs;
  const double n1 = double(r.n_above), n2 = double(r.n_below), N = n1 + n2;
  if (n1 == 0.0 || n2 == 0.0 || N < 2.0) { r.z = 0.0; r.p_value = 1.0; return r; }
  const double mean = 2.0 * n1 * n2 / N + 1.0;
  const double var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - N) / (N * N * (N - 1.0));
  if (!(var > 0.0)) { r.z = 0.0; r.p_value = 1.0; return r; }
  double diff = double(r.n_runs) - mean;
  if (continuity_correction) diff -= (diff > 0.0 ? 0.5 : -0.5);
  r.z = diff / std::sqrt(var);
  r.p_value = std::erfc(std::fabs(r.z) / std::sqrt(2.0));
  return r;
}

//! One histogram's goodness of fit (`poisson_goodness_of_fit`).
struct PoissonHistogramFit {
  double deviance = 0.0;         //!< `poisson_deviance` over the selected bins
  double dof = 0.0;              //!< selected bins minus the histogram's share of the parameters
  double deviance_per_dof = 0.0;
  double runs_p = 1.0;           //!< `runs_test` on the residual signs, cutoff at their mean
  double reference_mean = 0.0;   //!< deviance per dof of Poisson data drawn at the fitted means
  double reference_sd = 0.0;
  double z = 0.0;                //!< (deviance_per_dof - reference_mean) / reference_sd
  std::vector<double> weighted_residuals;   //!< (y - m) / sqrt(m) on the selected bins
};

//! Several histograms, each and together.
struct PoissonGoodnessOfFit {
  std::vector<PoissonHistogramFit> histograms;
  double deviance = 0.0, dof = 0.0, deviance_per_dof = 0.0, reference_mean = 0.0, reference_sd = 0.0, z = 0.0;
};

/**
 * \brief Deviance per degree of freedom against a reference measured from the
 *        model, and the runs test, for `n_hist` histograms.
 *
 * `y`, `m`, `mask` are `n_hist x n_bins` row-major; bins with `mask > 0` count.
 * Degrees of freedom: per histogram its selected bins minus `n_params / n_hist`,
 * overall all selected bins minus `n_params`.
 *
 * **Why a measured reference and not 1.** The Poisson deviance per bin is not one
 * at low counts, and a histogram's share of the parameters is a convention. So
 * `n_draws` Poisson data sets are drawn at the fitted means and scored against
 * them: `z` says how many sd of THAT the data sit above what the model would give
 * if it were true. The generator is `std::mt19937_64(seed)`.
 *
 * The runs test is `runs_test` on the signs of the weighted residuals with the
 * cutoff at their mean -- `statsmodels runstest_1samp(cutoff='mean',
 * correction=False)` on the signs, as a residual-structure test is usually run.
 */
inline PoissonGoodnessOfFit poisson_goodness_of_fit(const std::vector<double>& y, const std::vector<double>& m,
                                                    const std::vector<double>& mask, std::size_t n_hist,
                                                    double n_params, int n_draws = 150, std::uint64_t seed = 0) {
  const std::size_t n = n_hist ? y.size() / n_hist : 0;
  PoissonGoodnessOfFit G;
  G.histograms.resize(n_hist);
  std::vector<std::vector<std::size_t>> sel(n_hist);
  double nsel_all = 0.0;
  for (std::size_t k = 0; k < n_hist; ++k) {
    PoissonHistogramFit& h = G.histograms[k];
    std::vector<double> ys, ms, signs;
    for (std::size_t i = 0; i < n; ++i) {
      if (!(mask[k * n + i] > 0.0)) continue;
      sel[k].push_back(i);
      const double l = std::max(m[k * n + i], 1e-12), yy = y[k * n + i];
      ys.push_back(yy); ms.push_back(l);
      const double w = (yy - l) / std::sqrt(l);
      h.weighted_residuals.push_back(w);
      signs.push_back(w > 0.0 ? 1.0 : (w < 0.0 ? -1.0 : 0.0));
    }
    h.deviance = poisson_deviance(ys.data(), ms.data(), ys.size());
    h.dof = std::max(double(sel[k].size()) - n_params / double(std::max<std::size_t>(n_hist, 1)), 1.0);
    h.deviance_per_dof = h.deviance / h.dof;
    double mean_sign = 0.0;
    for (double t : signs) mean_sign += t;
    if (!signs.empty()) mean_sign /= double(signs.size());
    h.runs_p = runs_test(signs.data(), signs.size(), mean_sign).p_value;
    G.deviance += h.deviance;
    nsel_all += double(sel[k].size());
  }
  G.dof = std::max(nsel_all - n_params, 1.0);
  G.deviance_per_dof = G.deviance / G.dof;
  std::mt19937_64 rng(seed);
  std::vector<std::vector<double>> per(n_hist);
  std::vector<double> tot, yd, md;
  for (int d = 0; d < n_draws; ++d) {
    double all = 0.0;
    for (std::size_t k = 0; k < n_hist; ++k) {
      yd.clear(); md.clear();
      for (std::size_t i : sel[k]) {
        const double mu = std::max(m[k * n + i], 1e-12);
        yd.push_back(double(std::poisson_distribution<long long>(mu)(rng)));
        md.push_back(mu);
      }
      const double dk = poisson_deviance(yd.data(), md.data(), yd.size());
      per[k].push_back(dk / G.histograms[k].dof);
      all += dk;
    }
    tot.push_back(all / G.dof);
  }
  auto moments = [](const std::vector<double>& v, double& mean, double& sd) {
    mean = 0.0; sd = 0.0;
    if (v.empty()) return;
    for (double t : v) mean += t;
    mean /= double(v.size());
    for (double t : v) sd += (t - mean) * (t - mean);
    sd = std::sqrt(sd / double(v.size()));
  };
  for (std::size_t k = 0; k < n_hist; ++k) {
    PoissonHistogramFit& h = G.histograms[k];
    moments(per[k], h.reference_mean, h.reference_sd);
    h.z = (h.deviance_per_dof - h.reference_mean) / std::max(h.reference_sd, 1e-12);
  }
  moments(tot, G.reference_mean, G.reference_sd);
  G.z = (G.deviance_per_dof - G.reference_mean) / std::max(G.reference_sd, 1e-12);
  return G;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FITSTATISTICS_H

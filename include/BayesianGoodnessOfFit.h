/**
 * \file IMP/bff/BayesianGoodnessOfFit.h
 * \brief Whether a fitted count model describes its histograms: the Poisson
 *        deviance per degree of freedom against a reference measured from the
 *        model itself, and the runs test on the residual signs.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * Ported 2026-09-14 from ucfret's `s89_cpp/cbm56_fit.h` (PRD-142 step 2), where
 * it reproduced the Python (`s85_wres.fit_statistics`, statsmodels) runs-test p
 * to 1e-16 and the per-histogram deviance to 1e-9 on the CBM56 measurement.
 */

#ifndef IMPBFF_BAYESIANGOODNESSOFFIT_H
#define IMPBFF_BAYESIANGOODNESSOFFIT_H

#include <IMP/bff/IMPCompatibility.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name Goodness of fit of a count model
//! @{

/**
 * \brief The runs test on a sequence, two-sided normal p-value.
 *
 * `statsmodels.sandbox.stats.runs.runstest_1samp(x, cutoff='mean',
 * correction=False)`, line for line: the indicator `x >= mean(x)`, the number of
 * runs (maximal stretches of one indicator value), and
 *
 *     z = (runs - (2 n+ n- / n + 1)) / sqrt(2 n+ n- (2 n+ n- - n) / (n^2 (n - 1))),
 *     p = 2 sf(|z|).
 *
 * Applied to the SIGNS of weighted residuals it asks whether the residuals
 * alternate like noise or come in stretches, which a deviance per dof cannot see:
 * a residual wave of a few tenths of a sigma passes the deviance and fails
 * this. NaN when the sequence is shorter than two or has one indicator value.
 */
inline double bayesian_runs_test_p(const std::vector<double>& x) {
  const std::size_t n = x.size();
  if (n < 2) return std::nan("");
  double mean = 0.0;
  for (double t : x) mean += t;
  mean /= double(n);
  double npo = 0.0, runs = 1.0;
  int prev = -1;
  for (std::size_t i = 0; i < n; ++i) {
    const int ind = x[i] >= mean ? 1 : 0;
    npo += ind;
    if (i > 0 && ind != prev) runs += 1.0;
    prev = ind;
  }
  const double N = double(n), nne = N - npo, npn = npo * nne;
  const double rvar = 2.0 * npn * (2.0 * npn - N) / (N * N) / (N - 1.0);
  if (!(rvar > 0.0)) return std::nan("");
  const double z = (runs - (2.0 * npn / N + 1.0)) / std::sqrt(rvar);
  return std::erfc(std::fabs(z) / std::sqrt(2.0));
}

//! One histogram's statistics.
struct BayesianHistogramFit {
  double deviance = 0.0;       //!< 2 sum (m - y + y log(y/m)) over the selected bins
  double dof = 0.0;            //!< selected bins minus its share of the parameters
  double deviance_per_dof = 0.0;
  double runs_p = 0.0;         //!< runs test on the signs of the weighted residuals
  double reference_mean = 0.0; //!< deviance per dof of Poisson data drawn at the fitted means
  double reference_sd = 0.0;
  double z = 0.0;              //!< (deviance_per_dof - reference_mean) / reference_sd
  std::vector<double> weighted_residuals;   //!< (y - m) / sqrt(m) on the selected bins, in order
};

//! All histograms together.
struct BayesianGoodnessOfFit {
  std::vector<BayesianHistogramFit> histograms;
  double deviance = 0.0, dof = 0.0, deviance_per_dof = 0.0, reference_mean = 0.0, reference_sd = 0.0, z = 0.0;
};

/**
 * \brief Deviance, runs test and the measured reference for `n_hist` histograms.
 *
 * `y`, `m` and `mask` are `n_hist x n_bins`, row-major: counts, fitted means and
 * a selection (bins with `mask > 0` count). The degrees of freedom are, per
 * histogram, its selected bins minus `n_params / n_hist`; overall, all selected
 * bins minus `n_params`.
 *
 * **Why a measured reference and not 1.** The expected Poisson deviance per bin
 * is not one at low counts, and a histogram's dof share of a hundred parameters
 * is a guess. So the reference is measured: `n_draws` Poisson data sets drawn at
 * the fitted means, each one's deviance against those means, mean and sd per
 * histogram and overall. `z` then says how many reference sd the data sit above
 * what the model would produce if it were true. The generator is
 * `std::mt19937_64(seed)`: the reference agrees with another implementation's in
 * distribution, not draw by draw.
 */
inline BayesianGoodnessOfFit bayesian_goodness_of_fit(const std::vector<double>& y, const std::vector<double>& m,
                                                      const std::vector<double>& mask, std::size_t n_hist,
                                                      double n_params, int n_draws = 150, std::uint64_t seed = 0) {
  const std::size_t n = n_hist ? y.size() / n_hist : 0;
  BayesianGoodnessOfFit G;
  G.histograms.resize(n_hist);
  std::vector<std::vector<std::size_t>> sel(n_hist);
  double nsel_all = 0.0;
  for (std::size_t k = 0; k < n_hist; ++k) {
    BayesianHistogramFit& h = G.histograms[k];
    std::vector<double> signs;
    for (std::size_t i = 0; i < n; ++i) {
      if (!(mask[k * n + i] > 0.0)) continue;
      sel[k].push_back(i);
      const double l = std::max(m[k * n + i], 1e-12), yy = y[k * n + i];
      h.deviance += 2.0 * (l - yy + yy * std::log(std::max(yy, 1e-12) / l));
      const double w = (yy - l) / std::sqrt(l);
      h.weighted_residuals.push_back(w);
      signs.push_back(w > 0.0 ? 1.0 : (w < 0.0 ? -1.0 : 0.0));
    }
    h.dof = std::max(double(sel[k].size()) - n_params / double(std::max<std::size_t>(n_hist, 1)), 1.0);
    h.deviance_per_dof = h.deviance / h.dof;
    h.runs_p = bayesian_runs_test_p(signs);
    G.deviance += h.deviance;
    nsel_all += double(sel[k].size());
  }
  G.dof = std::max(nsel_all - n_params, 1.0);
  G.deviance_per_dof = G.deviance / G.dof;
  std::mt19937_64 rng(seed);
  std::vector<std::vector<double>> per(n_hist);
  std::vector<double> tot;
  for (int d = 0; d < n_draws; ++d) {
    double all = 0.0;
    for (std::size_t k = 0; k < n_hist; ++k) {
      double dk = 0.0;
      for (std::size_t i : sel[k]) {
        const double mu = std::max(m[k * n + i], 1e-12);
        const double yk = double(std::poisson_distribution<long long>(mu)(rng));
        dk += 2.0 * ((yk > 0.0 ? yk * std::log(yk / mu) : 0.0) - (yk - mu));
      }
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
    BayesianHistogramFit& h = G.histograms[k];
    moments(per[k], h.reference_mean, h.reference_sd);
    h.z = (h.deviance_per_dof - h.reference_mean) / std::max(h.reference_sd, 1e-12);
  }
  moments(tot, G.reference_mean, G.reference_sd);
  G.z = (G.deviance_per_dof - G.reference_mean) / std::max(G.reference_sd, 1e-12);
  return G;
}

//! @}

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_BAYESIANGOODNESSOFFIT_H */

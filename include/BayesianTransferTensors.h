/**
 * \file IMP/bff/BayesianTransferTensors.h
 * \brief The transfer tensors of a Bayesian FRET decay fit: the lifetime basis
 *        on an instrument axis, and the maps that carry a distance, a rotational
 *        time or an acceptor lifetime onto it.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * PRD-143 #18, written 2026-09-15 (ucfret prompt 448), replacing ucfret's Python
 * builders (`s80_analytic_stage2.build`, `s79_fret_stage2`, `s53_phase1_pseudolik.Basis`)
 * step by step. Step 18a: the basis. 18b: the ridge projection onto it. 18c: the
 * donor quenching and rotational maps.
 */

#ifndef IMPBFF_BAYESIANTRANSFERTENSORS_H
#define IMPBFF_BAYESIANTRANSFERTENSORS_H

#include <IMP/bff/BayesianMeasuredResponse.h>
#include <IMP/bff/internal/DampedNewton.h>

#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name Transfer tensors for a Bayesian FRET decay fit
//! @{

//! The lifetimes of the basis: `per_decade` points per decade in log10, from one
//! step above `log10_lo` to one step below `log10_hi` (both ends excluded, so a
//! lifetime at the edge of the grid is never an interior column).
inline std::vector<double> bayesian_log_lifetime_grid(double log10_lo, double log10_hi, int per_decade) {
  const double step = 1.0 / double(per_decade);
  std::vector<double> tau;
  for (int k = 1;; ++k) {
    const double u = log10_lo + k * step;
    if (u > log10_hi - step + 1e-9) break;
    tau.push_back(std::pow(10.0, u));
  }
  return tau;
}

/**
 * \brief A sampled Gaussian instrument response of unit sum.
 *
 * Channel `i` is `exp(-((i dt - position) / sigma)^2 / 2)`, sampled at the
 * channel's left edge (not integrated over it) and normalised to sum to one.
 * This is the response the transfer maps are built against: the maps are
 * meant not to depend on it, and a narrow analytic response keeps their
 * projection well conditioned.
 */
inline std::vector<double> bayesian_gaussian_response(std::size_t n, double dt, double position, double sigma) {
  std::vector<double> h(n);
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double u = (double(i) * dt - position) / sigma;
    h[i] = std::exp(-0.5 * u * u);
    s += h[i];
  }
  for (double& v : h) v /= s;
  return h;
}

//! What the transfer basis is built from (the defaults are ucfret's `s80.build`).
struct BayesianTransferBasisSpec {
  BayesianDecayAxis axis;                 //!< the instrument's axis and period
  double response_position = -1.0;        //!< ns; negative: 0.10 of the window n dt
  double response_sigma = 0.075;          //!< ns
  double log10_tau_lo = std::log10(0.05); //!< ns
  double log10_tau_hi = 0.0;              //!< ns; 0: log10 of three windows
  int per_decade = 10;
};

/**
 * \brief The lifetime basis the transfer maps are projected onto.
 *
 * `basis.B` is `n x K`: the response, one exact periodic column per grid
 * lifetime (`bayesian_response_basis`, no background removed, no shift), and a
 * flat column; each sums to one. `kernel` is kept because the maps reuse its
 * lifetimes and FFTs.
 */
struct BayesianTransferBasis {
  std::vector<double> response;
  std::shared_ptr<const BayesianPeriodicKernel> kernel;
  BayesianResponseBasis basis;
};

inline BayesianTransferBasis bayesian_transfer_basis(const BayesianTransferBasisSpec& spec) {
  const BayesianDecayAxis& ax = spec.axis;
  const double window = double(ax.n) * ax.dt;
  const double position = spec.response_position < 0.0 ? 0.10 * window : spec.response_position;
  const double hi = spec.log10_tau_hi == 0.0 ? std::log10(3.0 * window) : spec.log10_tau_hi;
  BayesianTransferBasis out;
  out.response = bayesian_gaussian_response(ax.n, ax.dt, position, spec.response_sigma);
  out.kernel = std::make_shared<const BayesianPeriodicKernel>(ax, bayesian_log_lifetime_grid(spec.log10_tau_lo, hi, spec.per_decade));
  BayesianResponseOptions opt;
  opt.remove_background = false;
  out.basis = bayesian_response_basis(*out.kernel, out.response, 0.0, 0.0, false, opt);
  return out;
}

//! The ridge strength of the map projection, relative to the mean diagonal of `B^T B`.
constexpr double BAYESIAN_TRANSFER_RIDGE = 1e-8;

/**
 * \brief The projection of a decay onto the transfer basis: ridge least squares,
 *        factored once for all targets (tttrlib's `RidgeProjector`).
 *
 * A map's coefficients are signed and unconstrained; the ridge keeps them bounded
 * where neighbouring lifetime columns are nearly collinear -- a plain least squares
 * gives large cancelling coefficients, harmless until a rotational map multiplies
 * them. `lambda = BAYESIAN_TRANSFER_RIDGE * trace(B^T B) / K`, as ucfret's `s79._ridge`.
 */
inline ::tttrlib::RidgeProjector bayesian_transfer_projector(const BayesianResponseBasis& basis,
                                                             double relative = BAYESIAN_TRANSFER_RIDGE) {
  ::tttrlib::RidgeProjector p;
  if (!p.factor(basis.B.data(), basis.n, basis.K, relative, true))
    throw std::runtime_error("bayesian_transfer_projector: B^T B + lambda I is not positive definite");
  return p;
}

/**
 * \brief Maps of an added decay rate: what each basis lifetime becomes when a
 *        second channel drains the same excited state.
 *
 * For grid point `j` with added rate `k_j` (1/ns), every basis lifetime `tau_c`
 * shortens to `tau_s = 1 / (1/tau_c + k_j)`. Its column is built the way the basis
 * builds its own (the exact periodic kernel at `tau_s`, unit sum) and scaled by
 * `tau_s / tau_c` -- the light a unit of `tau_c` emits once the added channel
 * takes its share -- then projected onto the basis. The result is
 * `n_rates x K x n_tau`, row-major: `S[j, :, c]` are the basis coefficients of
 * lifetime column `c` under rate `j`.
 *
 * Two uses, and the rate is the caller's:
 * - **donor quenching by FRET**: `k = (1/tau_ref) (R0/R)^6`. `R0` and `tau_ref`
 *   are a pair -- `R0` is determined at the donor lifetime `tau_ref`, so one
 *   `tau_ref` quenches every component alike (the homogeneous approximation; see
 *   `FRETSpectrumNode.h`, `PhotophysicsTransferKinetics.h`);
 * - **rotational depolarisation**: `k = 1/rho`, the anisotropy decay's share of a
 *   polarised channel.
 */
inline std::vector<double> bayesian_transfer_rate_maps(const BayesianTransferBasis& tb,
                                                       const ::tttrlib::RidgeProjector& projector,
                                                       const std::vector<double>& rates) {
  const BayesianResponseBasis& B = tb.basis;
  const std::size_t n = B.n, K = B.K;
  const std::vector<double>& tau = tb.kernel->tau();
  const std::size_t nt = tau.size();
  BayesianResponseOptions opt;
  opt.remove_background = false;
  std::vector<double> S(rates.size() * K * nt), y(n), x(K), tau_s(nt);
  for (std::size_t j = 0; j < rates.size(); ++j) {
    for (std::size_t c = 0; c < nt; ++c) tau_s[c] = 1.0 / (1.0 / tau[c] + rates[j]);
    const BayesianPeriodicKernel kernel_s(tb.kernel->axis(), tau_s);
    const BayesianResponseBasis cols = bayesian_response_basis(kernel_s, tb.response, 0.0, 0.0, false, opt);
    for (std::size_t c = 0; c < nt; ++c) {
      const double area = tau_s[c] / tau[c];
      for (std::size_t i = 0; i < n; ++i) y[i] = cols.B[i * K + 1 + c] * area;
      if (!projector.project(y.data(), x.data()))
        throw std::runtime_error("bayesian_transfer_rate_maps: projection failed");
      for (std::size_t k = 0; k < K; ++k) S[(j * K + k) * nt + c] = x[k];
    }
  }
  return S;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANTRANSFERTENSORS_H */

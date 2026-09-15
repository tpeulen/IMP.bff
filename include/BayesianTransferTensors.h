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
 * donor quenching and rotational maps. 18d: the sensitised and direct acceptor maps.
 */

#ifndef IMPBFF_BAYESIANTRANSFERTENSORS_H
#define IMPBFF_BAYESIANTRANSFERTENSORS_H

#include <IMP/bff/BayesianMeasuredResponse.h>
#include <IMP/bff/internal/DampedNewton.h>
#include <IMP/bff/internal/BayesianParallel.h>

#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <cmath>
#include <cstddef>
#include <complex>
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
 * Solved by Householder QR of `[B; sqrt(lambda) I]` (PRD-143 A3): on the CBM56 basis the
 * normal equations carry 3e-7 of error in the coefficients, QR 4e-13 (50-digit reference).
 */
inline ::tttrlib::RidgeProjector bayesian_transfer_projector(const BayesianResponseBasis& basis,
                                                             double relative = BAYESIAN_TRANSFER_RIDGE,
                                                             ::tttrlib::RidgeSolver solver = ::tttrlib::RidgeSolver::qr) {
  ::tttrlib::RidgeProjector p;
  if (!p.factor(basis.B.data(), basis.n, basis.K, relative, true, solver))
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
  std::vector<double> S(rates.size() * K * nt);
  internal::bayesian_parallel_for(rates.size(), [&](std::size_t j) {
    std::vector<double> y(n), x(K), tau_s(nt);
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
  });
  return S;
}

namespace internal {
/**
 * \brief `dt (sum_{j<=i} x[j] h[i-j] - h[i] x[0] / 2)`, clamped at zero, for one
 *        response `h` and many `x`: the causal convolution with the trapezoid's
 *        half weight on the first sample -- ucfret's `SpectrumDecoder._conv`.
 *
 * By FFT as `_conv` does it: `x` and `h` zero-padded to `L >= 2n` (a power of two),
 * so the full convolution (length `2n - 1`) does not wrap and its first `n` values
 * are the causal sum. `h`'s transform is kept, so each `x` costs one forward and
 * one inverse FFT of length `L` instead of `n^2 / 2` products. Equal to the direct
 * causal sum to round-off (PRD-143 A4.2: 1e-13 of the peak on every CBM56 acceptor
 * map, test `test_bayesian_transfer_basis.py`) and several times faster.
 */
class BayesianTrapezoidConvolver {
 public:
  BayesianTrapezoidConvolver(const double* h, std::size_t n, double dt) : n_(n), dt_(dt), h_(h, h + n) {
    L_ = 1;
    while (L_ < 2 * n) L_ *= 2;
    std::vector<double> hp(L_, 0.0);
    std::copy(h, h + n, hp.begin());
    H_.resize(L_ / 2 + 1);
    bayesian_rfft(hp.data(), L_, H_.data());
  }
  void operator()(const double* x, double* out) const {
    std::vector<double> xp(L_, 0.0), y(L_);
    std::copy(x, x + n_, xp.begin());
    std::vector<std::complex<double>> X(L_ / 2 + 1);
    bayesian_rfft(xp.data(), L_, X.data());
    for (std::size_t k = 0; k < X.size(); ++k) X[k] *= H_[k];
    bayesian_irfft(X.data(), L_, y.data());
    for (std::size_t i = 0; i < n_; ++i) out[i] = std::max(dt_ * (y[i] - 0.5 * h_[i] * x[0]), 0.0);
  }

 private:
  std::size_t n_, L_ = 1;
  double dt_;
  std::vector<double> h_;
  std::vector<std::complex<double>> H_;
};
//! What a channel collects of an exponential against its value at the channel start:
//! `tau (1 - exp(-dt/tau)) / dt`.
inline double bayesian_bin_factor(double tau, double dt) { return tau * -std::expm1(-dt / tau) / dt; }
}  // namespace internal

/**
 * \brief The acceptor's maps: sensitised emission after transfer, and direct excitation.
 *
 * Built as ucfret's `s79_fret_stage2.sens_maps_grid` and `direct_maps` build them
 * (PRD-143 #18d, a faithful port): point-sampled exponentials times the bin factor,
 * convolved by `internal::BayesianTrapezoidConvolver` -- NOT the exact periodic
 * kernel the basis and the donor maps use. That inconsistency is the prototype's and
 * is kept here so that the port can be gated; an exact construction is a separate,
 * named decision (PRD-143 #18f).
 *
 * With `t_i = i dt`, for FRET rate `k_j` and basis lifetime `tau_c`:
 * `tau_q = 1/(1/tau_c + k_j)`, efficiency `E = k_j tau_q`, the quenched donor
 * `d_i = exp(-t_i/tau_q) b(tau_q)`. For acceptor lifetime `tau_a` with kernel
 * `a_i = exp(-t_i/tau_a) / sum`, the sensitised decay is `conv(conv(d, a), response)`
 * scaled to total `E`, projected. Its rotational partner at `rho_a` is
 * `conv(conv(d e^{-t/rho}, a) e^{-t/rho}, response)` with the SAME scale -- the
 * depolarisation factor applied before and after the acceptor's kernel, as in the
 * prototype. The direct decay `conv(exp(-t/tau_a) b(tau_a), response)` is projected
 * and scaled so the basis decay sums to one; its partner
 * `conv(exp(-t/tau_a) e^{-t/rho} b(tau_a), response)` takes the same scale.
 *
 * Shapes, row-major: `sensitised` `n_rates x n_ta x K x n_tau`; `sensitised_rot`
 * `n_rho x n_rates x n_ta x K x n_tau`; `direct` `n_ta x K`; `direct_rot`
 * `n_rho x n_ta x K`.
 */
struct BayesianAcceptorMaps {
  std::size_t n_rates = 0, n_ta = 0, n_rho = 0, K = 0, n_tau = 0;
  std::vector<double> sensitised, sensitised_rot, direct, direct_rot;
};

inline BayesianAcceptorMaps bayesian_transfer_acceptor_maps(const BayesianTransferBasis& tb,
                                                            const ::tttrlib::RidgeProjector& projector,
                                                            const std::vector<double>& fret_rates,
                                                            const std::vector<double>& tau_a,
                                                            const std::vector<double>& rho_a) {
  const BayesianResponseBasis& B = tb.basis;
  const std::size_t n = B.n, K = B.K;
  const double dt = tb.kernel->axis().dt;
  const std::vector<double>& tau = tb.kernel->tau();
  const std::size_t nt = tau.size(), nr = fret_rates.size(), na = tau_a.size(), nh = rho_a.size();
  const double* h = tb.response.data();
  BayesianAcceptorMaps out;
  out.n_rates = nr; out.n_ta = na; out.n_rho = nh; out.K = K; out.n_tau = nt;
  out.sensitised.assign(nr * na * K * nt, 0.0);
  out.sensitised_rot.assign(nh * nr * na * K * nt, 0.0);
  out.direct.assign(na * K, 0.0);
  out.direct_rot.assign(nh * na * K, 0.0);
  auto project = [&](const std::vector<double>& y, double* x) {
    if (!projector.project(y.data(), x)) throw std::runtime_error("bayesian_transfer_acceptor_maps: projection failed");
  };
  std::vector<std::vector<double>> kern(na, std::vector<double>(n)), fall(nh, std::vector<double>(n));
  std::vector<internal::BayesianTrapezoidConvolver> by_kern;
  for (std::size_t l = 0; l < na; ++l) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) { kern[l][i] = std::exp(-double(i) * dt / tau_a[l]); s += kern[l][i]; }
    for (double& v : kern[l]) v /= s;
    by_kern.emplace_back(kern[l].data(), n, dt);
  }
  const internal::BayesianTrapezoidConvolver by_response(h, n, dt);
  for (std::size_t r = 0; r < nh; ++r)
    for (std::size_t i = 0; i < n; ++i) fall[r][i] = std::exp(-double(i) * dt / rho_a[r]);
  internal::bayesian_parallel_for(nr, [&](std::size_t j) {
    std::vector<double> d(n), w(n), acc(n), col(n), x(K);
    for (std::size_t c = 0; c < nt; ++c) {
      const double tq = 1.0 / (1.0 / tau[c] + fret_rates[j]), eff = fret_rates[j] * tq, bf = internal::bayesian_bin_factor(tq, dt);
      for (std::size_t i = 0; i < n; ++i) d[i] = std::exp(-double(i) * dt / tq) * bf;
      for (std::size_t l = 0; l < na; ++l) {
        by_kern[l](d.data(), acc.data());
        by_response(acc.data(), col.data());
        double tot = 0.0;
        for (double v : col) tot += v;
        const double scale = tot > 1e-300 ? eff / tot : 0.0;
        for (double& v : col) v *= scale;
        project(col, x.data());
        for (std::size_t k = 0; k < K; ++k) out.sensitised[((j * na + l) * K + k) * nt + c] = x[k];
        for (std::size_t r = 0; r < nh; ++r) {
          for (std::size_t i = 0; i < n; ++i) w[i] = d[i] * fall[r][i];
          by_kern[l](w.data(), acc.data());
          for (std::size_t i = 0; i < n; ++i) acc[i] *= fall[r][i];
          by_response(acc.data(), col.data());
          for (double& v : col) v *= scale;
          project(col, x.data());
          for (std::size_t k = 0; k < K; ++k) out.sensitised_rot[(((r * nr + j) * na + l) * K + k) * nt + c] = x[k];
        }
      }
    }
  });
  std::vector<double> d(n), w(n), col(n), x(K);
  for (std::size_t l = 0; l < na; ++l) {
    const double bf = internal::bayesian_bin_factor(tau_a[l], dt);
    for (std::size_t i = 0; i < n; ++i) d[i] = std::exp(-double(i) * dt / tau_a[l]) * bf;
    by_response(d.data(), col.data());
    project(col, x.data());
    double area = 0.0;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) area += B.B[i * K + k] * x[k];
    area = std::max(area, 1e-300);
    for (std::size_t k = 0; k < K; ++k) out.direct[l * K + k] = x[k] / area;
    for (std::size_t r = 0; r < nh; ++r) {
      for (std::size_t i = 0; i < n; ++i) w[i] = d[i] * fall[r][i];
      by_response(w.data(), col.data());
      project(col, x.data());
      for (std::size_t k = 0; k < K; ++k) out.direct_rot[(r * na + l) * K + k] = x[k] / area;
    }
  }
  return out;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANTRANSFERTENSORS_H */

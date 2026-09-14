/**
 * \file IMP/bff/BayesianMeasuredResponse.h
 * \brief A measured instrument response prepared for a Bayesian decay fit: its
 *        background, its shift, and the periodic lifetime basis it defines,
 *        with the derivatives a scoring step needs.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * Ported 2026-09-14 from ucfret's `investigation/pinn_pR_anisotropy/s89_cpp`
 * (PRD-142 step 1), where it was gated against the Python prototype
 * (`s88_laplace_posterior.InstrumentModel`, torch) at 5e-15 per basis column on
 * the CBM56 measurement.
 */

#ifndef IMPBFF_BAYESIANMEASUREDRESPONSE_H
#define IMPBFF_BAYESIANMEASUREDRESPONSE_H

#include <IMP/bff/IMPCompatibility.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "pocketfft/pocketfft_hdronly.h"

IMPBFF_BEGIN_NAMESPACE

//! \name A measured response for a Bayesian decay fit
//! @{

//! The time axis of a TCSPC histogram and the excitation period on it.
struct BayesianDecayAxis {
  std::size_t n = 0;          //!< channels in the histogram
  double dt = 0.0;            //!< channel width, ns
  std::size_t n_period = 0;   //!< channels per excitation period (the periodic kernel's length)
};

/**
 * \brief `log(1 + exp(x))`, and `x` itself above 20.
 *
 * The threshold is torch's `softplus` default, kept so that a model written in
 * torch and this one agree to rounding, which is what their gates compare. Its
 * derivative switches at the same point (`bayesian_softplus_derivative`).
 */
inline double bayesian_softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
//! d softplus / dx with the same threshold: 1 above 20, the logistic below.
inline double bayesian_softplus_derivative(double x) { return x > 20.0 ? 1.0 : 1.0 / (1.0 + std::exp(-x)); }
//! `soft * softplus(x / soft)`: equal to `x` above a few `soft`, never below zero, smooth.
inline double bayesian_soft_positive(double x, double soft) { return soft * bayesian_softplus(x / soft); }

//! Real FFT of length n (unscaled), bins 0..n/2 -- numpy's `rfft`.
inline void bayesian_rfft(const double* x, std::size_t n, std::complex<double>* X) {
  pocketfft::r2c(pocketfft::shape_t{n}, pocketfft::stride_t{std::ptrdiff_t(sizeof(double))},
                 pocketfft::stride_t{std::ptrdiff_t(sizeof(std::complex<double>))}, pocketfft::shape_t{0}, true, x, X, 1.0);
}
//! Its inverse, divided by n -- numpy's `irfft` (the "backward" normalisation).
inline void bayesian_irfft(const std::complex<double>* X, std::size_t n, double* x) {
  pocketfft::c2r(pocketfft::shape_t{n}, pocketfft::stride_t{std::ptrdiff_t(sizeof(std::complex<double>))},
                 pocketfft::stride_t{std::ptrdiff_t(sizeof(double))}, pocketfft::shape_t{0}, false, X, x, 1.0 / double(n));
}

/**
 * \brief The exact bin-integrated periodic decay kernel on a lifetime grid.
 *
 * For light absorbed uniformly within channel 0 and a single exponential of
 * lifetime `tau` under excitation every `n_period` channels, the fraction
 * detected in channel `k` is, with `s = tau / dt` and `q = exp(-dt / tau)`,
 *
 *     K(0) = 1 - s (1 - q),     K(k >= 1) = s (1 - q)^2 q^(k-1),
 *
 * periodised (the tail of every earlier pulse folds back, a geometric sum). It
 * sums to one by construction. The trapezoid kernel it replaces was off by
 * 1-1.6 % of the peak at rotational times of tens of ps, four to six Poisson sd
 * in the early channels of a 1e5-count histogram -- enough to pin a fitted
 * fundamental anisotropy against its bound (CBM56, 2026-09-14).
 *
 * The object keeps each column's FFT: it does not depend on the parameters
 * being fitted, and computing it per basis was a quarter of a fit's cost. It
 * is immutable after construction and safe to share between threads.
 */
class BayesianPeriodicKernel {
 public:
  BayesianPeriodicKernel(const BayesianDecayAxis& axis, const std::vector<double>& tau)
      : axis_(axis), tau_(tau), fft_(tau.size(), std::vector<std::complex<double>>(axis.n_period / 2 + 1)) {
    const std::size_t np = axis.n_period;
    std::vector<double> k(np);
    for (std::size_t c = 0; c < tau.size(); ++c) {
      kernel(c, k.data());
      bayesian_rfft(k.data(), np, fft_[c].data());
    }
  }
  const BayesianDecayAxis& axis() const { return axis_; }
  const std::vector<double>& tau() const { return tau_; }
  const std::vector<std::complex<double>>& fft(std::size_t column) const { return fft_[column]; }
  //! The kernel of grid lifetime `column` over one period, into `out` (n_period values).
  void kernel(std::size_t column, double* out) const {
    const std::size_t np = axis_.n_period;
    const double dt = axis_.dt, t = tau_[column];
    const double one_q = -std::expm1(-dt / t), s = t / dt;
    const double wrap = 1.0 / (-std::expm1(-double(np) * dt / t));
    for (std::size_t k = 1; k < np; ++k) out[k] = s * one_q * one_q * std::exp(-(double(k) - 1.0) * dt / t) * wrap;
    out[0] = (1.0 - s * one_q) + s * one_q * one_q * std::exp(-(double(np) - 1.0) * dt / t) * wrap;
  }

 private:
  BayesianDecayAxis axis_;
  std::vector<double> tau_;
  std::vector<std::vector<std::complex<double>>> fft_;
};

/**
 * \brief A measured response and the basis it defines, with its tangents.
 *
 * `B` is `n x K`, row-major, `K = n_tau + 2`: column 0 the prepared response
 * itself (light that reaches the detector without being fluorescence), columns
 * 1..n_tau the response convolved with each kernel column and renormalised,
 * column K-1 flat (`1/n`, an uncorrelated background). Every column sums to
 * one, so the amplitudes a model puts on them are counts.
 *
 * `dB_background` and `dB_shift` are `d B / d background_fraction` and
 * `d B / d shift_bins` (empty unless asked for). A caller fitting the shift in
 * ns multiplies by `1/dt`; a caller fitting a transformed coordinate by its
 * chain factor.
 */
struct BayesianResponseBasis {
  std::size_t n = 0, K = 0;
  std::vector<double> response;        //!< the prepared response (column 0), n
  std::vector<double> B, dB_background, dB_shift;
};

//! What `bayesian_response_basis` does besides the defaults.
struct BayesianResponseOptions {
  double soft = 0.05;                   //!< the soft floor of the background removal, in units of `1e-4 * max`
  bool clamp = true;                    //!< clamp the shifted response at zero (the forward model)
  bool clamp_in_tangent = true;         //!< mask the tangents where the clamp bites; false only to test that it matters
};

/**
 * \brief Prepare a measured response and build its periodic basis.
 *
 * **1. The background.** A fraction `b` of the response's counts, spread over
 * the channels the response occupies (`> 0`), is removed under a soft floor at
 * `1e-4` of its maximum; channels that were zero stay zero. Over the support
 * and not over the whole window: a response isolated to one pulse is exactly
 * zero elsewhere, and spreading the background there removes it where there
 * was none.
 *
 * **2. The shift.** A fractional circular shift by a phase ramp,
 * `irfft(rfft(u) exp(-2 pi i f s))`, CLAMPED at zero. The ramp rings where the
 * response is zero; unclamped, the negative lobes enter every convolved column,
 * and the likelihood saw it -- 204 nats on the CBM56 global fit.
 *
 * **3. Unit sum**, then the kernel columns by one FFT product each, clamped at
 * zero and renormalised.
 *
 * The tangents are forward-mode by hand through the same four steps.
 * **One place they are not a derivative:** where the shift is exactly zero the
 * zeros outside the support come back from the FFT round trip as +-1e-20, and
 * which of them the clamp keeps is round-off; a gradient there depends on the
 * FFT library. Off zero it agrees with central differences to 2e-9.
 */
inline BayesianResponseBasis bayesian_response_basis(const BayesianPeriodicKernel& kernel, const std::vector<double>& measured,
                                                     double background_fraction, double shift_bins, bool tangents,
                                                     const BayesianResponseOptions& opt = BayesianResponseOptions()) {
  using cd = std::complex<double>;
  const BayesianDecayAxis& ax = kernel.axis();
  const std::size_t n = measured.size(), np = ax.n_period, h = np / 2 + 1, nt = kernel.tau().size(), K = nt + 2;
  const double two_pi = 2.0 * M_PI;
  BayesianResponseBasis out;
  out.n = n; out.K = K;

  // 1. background, and d/d b
  double sum = 0.0, mx = -1e300, n_sup = 0.0;
  for (double t : measured) { sum += t; mx = std::max(mx, t); if (t > 0.0) n_sup += 1.0; }
  n_sup = std::max(n_sup, 1.0);
  const double level = background_fraction * sum / n_sup, sc = 1e-4 * mx;
  std::vector<double> u(n, 0.0), du_b(tangents ? n : 0, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (!(measured[i] > 0.0)) continue;
    const double x = (measured[i] - level) / sc;
    u[i] = bayesian_soft_positive(x, opt.soft) * sc;
    if (tangents) du_b[i] = bayesian_softplus_derivative(x / opt.soft) * (-sum / n_sup);
  }
  // 2. the shift, clamped; tangents in b (through the ramp) and in the shift
  std::vector<cd> U(n / 2 + 1), T(n / 2 + 1), ramp(n / 2 + 1);
  bayesian_rfft(u.data(), n, U.data());
  for (std::size_t k = 0; k <= n / 2; ++k) ramp[k] = std::exp(cd(0.0, -two_pi * double(k) / double(n) * shift_bins));
  for (std::size_t k = 0; k <= n / 2; ++k) T[k] = U[k] * ramp[k];
  std::vector<double> y(n), dy_b(tangents ? n : 0), dy_s(tangents ? n : 0);
  bayesian_irfft(T.data(), n, y.data());
  if (tangents) {
    bayesian_rfft(du_b.data(), n, T.data());
    for (std::size_t k = 0; k <= n / 2; ++k) T[k] *= ramp[k];
    bayesian_irfft(T.data(), n, dy_b.data());
    for (std::size_t k = 0; k <= n / 2; ++k) T[k] = U[k] * ramp[k] * cd(0.0, -two_pi * double(k) / double(n));
    bayesian_irfft(T.data(), n, dy_s.data());
    if (opt.clamp && opt.clamp_in_tangent)
      for (std::size_t i = 0; i < n; ++i) if (!(y[i] >= 0.0)) { dy_b[i] = 0.0; dy_s[i] = 0.0; }
  }
  if (opt.clamp) for (double& t : y) t = std::max(t, 0.0);
  // 3. unit sum
  double S = 0.0; for (double t : y) S += t;
  auto quotient = [&](std::vector<double>& d) {
    double sd = 0.0; for (double q : d) sd += q;
    for (std::size_t i = 0; i < n; ++i) d[i] = d[i] / S - y[i] * sd / (S * S);
  };
  if (tangents) { quotient(dy_b); quotient(dy_s); }
  for (double& t : y) t /= S;
  out.response = y;
  // 4. the basis and its tangents
  out.B.assign(n * K, 0.0);
  if (tangents) { out.dB_background.assign(n * K, 0.0); out.dB_shift.assign(n * K, 0.0); }
  for (std::size_t i = 0; i < n; ++i) {
    out.B[i * K] = y[i]; out.B[i * K + K - 1] = 1.0 / double(n);
    if (tangents) { out.dB_background[i * K] = dy_b[i]; out.dB_shift[i * K] = dy_s[i]; }
  }
  std::vector<double> buf(np, 0.0), col(np), dcol(np);
  std::vector<cd> Y(h), Db(tangents ? h : 0), Ds(tangents ? h : 0), tmp(h);
  auto to_period = [&](const std::vector<double>& v, std::vector<cd>& V) {
    std::fill(buf.begin(), buf.end(), 0.0);
    for (std::size_t i = 0; i < std::min(np, n); ++i) buf[i] = v[i];
    bayesian_rfft(buf.data(), np, V.data());
  };
  to_period(y, Y);
  if (tangents) { to_period(dy_b, Db); to_period(dy_s, Ds); }
  for (std::size_t c = 0; c < nt; ++c) {
    const std::vector<cd>& KF = kernel.fft(c);
    for (std::size_t k = 0; k < h; ++k) tmp[k] = KF[k] * Y[k];
    bayesian_irfft(tmp.data(), np, col.data());
    double csum = 0.0;
    for (std::size_t i = 0; i < n; ++i) csum += std::max(col[i], 0.0);
    csum = std::max(csum, 1e-300);
    for (std::size_t i = 0; i < n; ++i) out.B[i * K + 1 + c] = std::max(col[i], 0.0) / csum;
    if (!tangents) continue;
    for (int which = 0; which < 2; ++which) {
      const std::vector<cd>& D = which == 0 ? Db : Ds;
      std::vector<double>& dB = which == 0 ? out.dB_background : out.dB_shift;
      for (std::size_t k = 0; k < h; ++k) tmp[k] = KF[k] * D[k];
      bayesian_irfft(tmp.data(), np, dcol.data());
      double dsum = 0.0;
      for (std::size_t i = 0; i < n; ++i) { if (!(col[i] >= 0.0)) dcol[i] = 0.0; dsum += dcol[i]; }
      for (std::size_t i = 0; i < n; ++i) dB[i * K + 1 + c] = dcol[i] / csum - std::max(col[i], 0.0) * dsum / (csum * csum);
    }
  }
  return out;
}

//! @}

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_BAYESIANMEASUREDRESPONSE_H */

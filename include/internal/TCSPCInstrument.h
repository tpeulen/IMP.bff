/**
 *  \file IMP/bff/internal/TCSPCInstrument.h
 *  \brief The TCSPC instrument stage, once: scale, scatter, background pattern,
 *         pile-up, constant background and differential non-linearity, applied
 *         to a convolved decay.
 *
 *  One implementation for every model that turns a convolved decay into expected
 *  counts -- `TCSPCDecay` (ChiSurf's descriptions) and the Bayesian decay model
 *  (PRD-143 #22, tpeulen 2026-09-15: "bayesian way better, but pileup missing,
 *  chisurf changes"). Only what a fit evaluates is here; deriving a linearisation
 *  table, autoscaling and reporting stay with the callers.
 *
 *  **The parameterisation: fractions of the fluorescence total.** With `F` the
 *  convolved decay and `total = sum(F)`:
 *
 *      components = scale (F + scatter total response + pattern total pattern_shape)
 *      background = background scale total flat_shape
 *
 *  `response`, `pattern_shape` and `flat_shape` each sum to one, so `scatter`,
 *  `pattern` and `background` are the fractions of the fluorescence counts that
 *  scattered light, the measured background pattern and uncorrelated background
 *  carry, and `scale` the counts per unit of `F`. The shapes are given, not built:
 *  in channel space they are the prepared response, the (shifted, normalised)
 *  pattern and `1/n`; in the amplitude space of a basis whose columns each sum to
 *  one they are unit vectors -- the same model, which is why one function serves
 *  both. ChiSurf's absolute scatter and constant background convert with
 *  `tcspc_instrument_fractions_from_absolute`.
 *
 *  **The order** is ChiSurf's (its parity tests pin it): the components; Coates
 *  pile-up (`add_pile_up_to_model_ad`, tttrlib, vendored) on them -- a
 *  data-dependent factor per channel, so it commutes with `scale`; the background
 *  added after pile-up; the DNL table multiplying the finished curve.
 *
 *  Templated on the number type, so a dual-number caller gets exact derivatives;
 *  `tcspc_instrument_components_jacobian` gives them in closed form for `double`.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved. Written 2026-09-15.
 */

#ifndef IMPBFF_INTERNAL_TCSPC_INSTRUMENT_H
#define IMPBFF_INTERNAL_TCSPC_INSTRUMENT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/DecayConvolution.h>

#include <algorithm>
#include <cstddef>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! The fitted quantities of the stage.
template <typename T = double>
struct TCSPCInstrumentParameters {
  T scale = T(1.0);        //!< counts per unit of the convolved decay
  T scatter = T(0.0);      //!< scattered light, as a fraction of the fluorescence total
  T pattern = T(0.0);      //!< the background pattern, as a fraction of the fluorescence total
  T background = T(0.0);   //!< uncorrelated background, as a fraction of the fluorescence total
};

//! What the stage is applied with, and does not fit. Null pointers switch a part off.
struct TCSPCInstrumentSettings {
  const double* response = nullptr;       //!< unit sum, n (scatter's shape); null: no scatter
  const double* pattern = nullptr;        //!< unit sum, n (the pattern's shape); null: none
  const double* flat = nullptr;           //!< unit sum, n (the background's shape); null: none
  //! Coates pile-up (all needed): the data it is computed from, the rate, dead time, time.
  const double* pile_up_data = nullptr;
  int pile_up_n_data = 0;
  double repetition_rate_mhz = 0.0, dead_time_ns = 0.0, measurement_time_s = 0.0;
  int pile_up_start = 0, pile_up_stop = -1;
  const double* linearization = nullptr;  //!< n; multiplies the finished curve; null: none
};

//! The fluorescence total `sum(curve)`.
template <typename T>
inline T tcspc_instrument_total(const T* curve, std::size_t n) {
  T total = T(0.0);
  for (std::size_t i = 0; i < n; ++i) total += curve[i];
  return total;
}

/**
 * \brief Scale, scatter and pattern onto \p curve in place; the background into
 *        \p background_out (overwritten, `n`), to be added after pile-up.
 *
 * Pass the same array for both when nothing comes between (no pile-up): the
 * background is then added at once. Returns the fluorescence total.
 */
template <typename T>
inline T tcspc_instrument_components(T* curve, std::size_t n, const TCSPCInstrumentSettings& s,
                                     const TCSPCInstrumentParameters<T>& p, T* background_out) {
  const T total = tcspc_instrument_total(curve, n);
  if (background_out != curve) for (std::size_t i = 0; i < n; ++i) background_out[i] = T(0.0);
  for (std::size_t i = 0; i < n; ++i) {
    T v = curve[i];
    if (s.response) v += p.scatter * total * s.response[i];
    if (s.pattern) v += p.pattern * total * s.pattern[i];
    curve[i] = p.scale * v;
  }
  if (s.flat) {
    const T b = p.background * p.scale * total;
    for (std::size_t i = 0; i < n; ++i) background_out[i] += b * s.flat[i];
  }
  return total;
}

/**
 * \brief Pile-up on \p curve, then \p background added, then the DNL multiply.
 *
 * `background` may be null (already added, or none).
 */
template <typename T>
inline void tcspc_instrument_detection(T* curve, std::size_t n, const TCSPCInstrumentSettings& s,
                                       const T* background) {
  if (s.pile_up_data)
    add_pile_up_to_model_ad<T>(curve, int(n), s.pile_up_data, s.pile_up_n_data, s.repetition_rate_mhz,
                               s.dead_time_ns, s.measurement_time_s, s.pile_up_start, s.pile_up_stop);
  if (background) for (std::size_t i = 0; i < n; ++i) curve[i] += background[i];
  if (s.linearization) for (std::size_t i = 0; i < n; ++i) curve[i] *= s.linearization[i];
}

//! The whole stage on a convolved decay in channel space: components, pile-up, background, DNL.
template <typename T>
inline void tcspc_instrument(T* curve, std::size_t n, const TCSPCInstrumentSettings& s,
                             const TCSPCInstrumentParameters<T>& p) {
  std::vector<T> background(n, T(0.0));
  tcspc_instrument_components(curve, n, s, p, background.data());
  tcspc_instrument_detection(curve, n, s, background.data());
}

/**
 * \brief Derivatives of `tcspc_instrument_components` (background added in place).
 *
 * \param F the curve BEFORE the stage, `n`
 * \param J_F `(n, m)` row-major, `dF/dtheta` for the caller's `m` coordinates
 * \param J_out `(n, m)` row-major out: the stage applied to each column of `J_F`
 *        (it is affine in `F`, so this is the chain rule)
 * \param d_parameters `(n, 4)` row-major out: d/d `scale`, `scatter`, `pattern`,
 *        `background`, in that order
 * Either output may be null.
 */
inline void tcspc_instrument_components_jacobian(const double* F, std::size_t n, const TCSPCInstrumentSettings& s,
                                                 const TCSPCInstrumentParameters<double>& p, const double* J_F,
                                                 std::size_t m, double* J_out, double* d_parameters) {
  const double total = tcspc_instrument_total(F, n);
  if (J_out) {
    std::vector<double> col_total(m, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t c = 0; c < m; ++c) col_total[c] += J_F[i * m + c];
    for (std::size_t i = 0; i < n; ++i) {
      const double shape = (s.response ? p.scatter * s.response[i] : 0.0) + (s.pattern ? p.pattern * s.pattern[i] : 0.0)
                           + (s.flat ? p.background * s.flat[i] : 0.0);
      for (std::size_t c = 0; c < m; ++c)
        J_out[i * m + c] = p.scale * (J_F[i * m + c] + shape * col_total[c]);
    }
  }
  if (d_parameters) {
    for (std::size_t i = 0; i < n; ++i) {
      const double r = s.response ? s.response[i] : 0.0, q = s.pattern ? s.pattern[i] : 0.0, f = s.flat ? s.flat[i] : 0.0;
      double* row = d_parameters + i * 4;
      row[0] = F[i] + (p.scatter * r + p.pattern * q + p.background * f) * total;
      row[1] = p.scale * total * r;
      row[2] = p.scale * total * q;
      row[3] = p.scale * total * f;
    }
  }
}

/**
 * \brief ChiSurf's absolute scatter and constant background as the stage's fractions.
 *
 * ChiSurf adds `scatter_absolute * response` (response of unit sum) to the
 * convolved decay before scaling by `n0`, and `background_absolute` counts to every
 * one of `n` channels after. With `total = sum(F)`: `scatter = scatter_absolute /
 * total`, `background = background_absolute * n / (n0 * total)`, `scale = n0`,
 * `flat = 1/n`. The conversion depends on `total`, so a description migrated to
 * fractions reproduces a frozen absolute fit only at that fit's own curve.
 */
inline TCSPCInstrumentParameters<double> tcspc_instrument_fractions_from_absolute(
    double n0, double scatter_absolute, double background_absolute, double fluorescence_total, std::size_t n) {
  TCSPCInstrumentParameters<double> p;
  p.scale = n0;
  p.scatter = fluorescence_total != 0.0 ? scatter_absolute / fluorescence_total : 0.0;
  p.background = (n0 * fluorescence_total) != 0.0 ? background_absolute * double(n) / (n0 * fluorescence_total) : 0.0;
  return p;
}

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_TCSPC_INSTRUMENT_H

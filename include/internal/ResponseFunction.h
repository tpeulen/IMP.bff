/**
 *  \file IMP/bff/internal/ResponseFunction.h
 *  \brief A measured response, shifted and normalised, the one way the graph does it.
 *
 *  Every node that convolves with an instrument response prepares it the same
 *  way -- the shift first, then the unit-sum normalisation, because the shift
 *  zeroes what moves past either end and normalising first would leave the
 *  model's area depending on the timeshift. One implementation, so two nodes
 *  over the same response and timeshift cannot disagree about it.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_RESPONSE_FUNCTION_H
#define IMPBFF_INTERNAL_RESPONSE_FUNCTION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/DecayConvolution.h>
#include <IMP/bff/BayesianFisherScoring.h>

#include <algorithm>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! How a measured response is cleaned before it is shifted and normalised.
/*! Inactive, the response is used as measured. Active -- which a node is when
    its description sets a response window or wires a response background --
    it is ChiSurf's IRF preparation, in its order: the constant background
    (a lamp or dark count level) is subtracted, what goes negative is clipped
    to zero, and everything outside `[start, stop)` is zeroed. */
//! What `ResponsePreparation::background` means.
/*! `constant` (ChiSurf): a level in counts per channel, subtracted and clipped at
    zero. `fraction_of_support` (the Bayesian decay model, PRD-142/143): a fraction
    `b` of the response's counts, spread evenly over the channels the response
    occupies (`> 0`), removed under a soft floor at `1e-4` of the response's maximum
    (`bayesian_soft_positive` with torch's threshold); channels that were zero stay
    zero. Over the support, not the window, because a response isolated to one
    pulse is exactly zero elsewhere. */
enum class ResponseBackground { constant, fraction_of_support };

struct ResponsePreparation {
  bool active = false;
  double background = 0.0;
  int start = 0;
  //! Exclusive; negative means the end of the response.
  int stop = -1;
  ResponseBackground background_mode = ResponseBackground::constant;
  //! The soft floor's width for `fraction_of_support`, in units of `1e-4 * max`.
  double soft = 0.05;

  bool operator==(const ResponsePreparation& other) const {
    return active == other.active && background == other.background &&
           start == other.start && stop == other.stop &&
           background_mode == other.background_mode && soft == other.soft;
  }
  bool operator!=(const ResponsePreparation& other) const {
    return !(*this == other);
  }
};

//! The response after its #ResponsePreparation, or the response itself.
/*! \return a pointer into \p cleaned when the preparation is active, else
    into \p response; either way `response.size()` values. */
inline const double* clean_response(const std::vector<double>& response,
                                    const ResponsePreparation& preparation,
                                    std::vector<double>& cleaned) {
  if (!preparation.active) return response.data();
  const int n_points = static_cast<int>(response.size());
  cleaned.resize(response.size());
  const int stop =
      preparation.stop < 0 ? n_points : std::min(preparation.stop, n_points);
  if (preparation.background_mode == ResponseBackground::fraction_of_support) {
    double sum = 0.0, mx = -1e300, n_sup = 0.0;
    for (double t : response) { sum += t; mx = std::max(mx, t); if (t > 0.0) n_sup += 1.0; }
    n_sup = std::max(n_sup, 1.0);
    const double level = preparation.background * sum / n_sup, sc = 1e-4 * mx;
    for (int i = 0; i < n_points; ++i) {
      const double r = response[static_cast<std::size_t>(i)];
      const bool inside = i >= preparation.start && i < stop;
      cleaned[static_cast<std::size_t>(i)] =
          (inside && r > 0.0) ? bayesian_soft_positive((r - level) / sc, preparation.soft,
                                                       BAYESIAN_TORCH_SOFTPLUS_THRESHOLD) * sc
                              : 0.0;
    }
    return cleaned.data();
  }
  for (int i = 0; i < n_points; ++i) {
    const bool inside = i >= preparation.start && i < stop;
    const double value =
        response[static_cast<std::size_t>(i)] - preparation.background;
    cleaned[static_cast<std::size_t>(i)] = (inside && value > 0.0) ? value : 0.0;
  }
  return cleaned.data();
}

//! Clean \p response, shift it by \p timeshift channels and scale it to unit sum.
/*! The cleaning comes first because the shift interpolates between
    neighbouring channels, and a background or a stray count outside the
    window would otherwise be shifted into it. Sign of the shift: ChiSurf's
    `shift_array(v, s)` is tttrlib's `shift_lamp(v, -s)`; at `s == 0` the
    response is copied, not shifted.
    \param[in] response the measured response
    \param[in] preparation how it is cleaned first
    \param[in] timeshift the shift, in channels
    \param[in,out] cleaned scratch for the cleaned copy, reused across calls
    \param[in,out] shifted scratch for the shifted copy, reused across calls
    \param[out] out the prepared response, as long as \p response
*/
inline void prepare_response(const std::vector<double>& response,
                             const ResponsePreparation& preparation,
                             double timeshift, std::vector<double>& cleaned,
                             std::vector<double>& shifted,
                             std::vector<double>& out) {
  const int n_points = static_cast<int>(response.size());
  const double* source = clean_response(response, preparation, cleaned);
  if (timeshift != 0.0) {
    shifted.resize(response.size());
    shift_lamp_ad<double>(shifted.data(), source, -timeshift, n_points, 0.0);
    source = shifted.data();
  }
  // The sum is taken from the source and folded into the copy, so the
  // response is walked twice rather than three times.
  out.resize(response.size());
  double total = 0.0;
  for (int i = 0; i < n_points; ++i) total += source[i];
  if (total > 0.0) {
    for (int i = 0; i < n_points; ++i) {
      out[static_cast<std::size_t>(i)] = source[i] / total;
    }
  } else {
    std::copy(source, source + n_points, out.begin());
  }
}

//! `d cleaned[i] / d preparation.background`, into \p out (`response.size()`).
/*! `constant`: -1 where the channel is inside the window and above the level, 0
    elsewhere (the clip's derivative). `fraction_of_support`: the soft floor's
    derivative times `-sum / n_support`. Inactive: zeros. */
inline void clean_response_background_derivative(const std::vector<double>& response,
                                                 const ResponsePreparation& preparation, double* out) {
  const int n_points = static_cast<int>(response.size());
  for (int i = 0; i < n_points; ++i) out[i] = 0.0;
  if (!preparation.active) return;
  const int stop =
      preparation.stop < 0 ? n_points : std::min(preparation.stop, n_points);
  if (preparation.background_mode == ResponseBackground::fraction_of_support) {
    double sum = 0.0, mx = -1e300, n_sup = 0.0;
    for (double t : response) { sum += t; mx = std::max(mx, t); if (t > 0.0) n_sup += 1.0; }
    n_sup = std::max(n_sup, 1.0);
    const double level = preparation.background * sum / n_sup, sc = 1e-4 * mx;
    for (int i = 0; i < n_points; ++i) {
      const double r = response[static_cast<std::size_t>(i)];
      if (i < preparation.start || i >= stop || !(r > 0.0)) continue;
      out[i] = bayesian_soft_positive_derivative((r - level) / sc, preparation.soft, BAYESIAN_TORCH_SOFTPLUS_THRESHOLD)
               * (-sum / n_sup);
    }
    return;
  }
  for (int i = 0; i < n_points; ++i) {
    const bool inside = i >= preparation.start && i < stop;
    out[i] = (inside && response[static_cast<std::size_t>(i)] - preparation.background > 0.0) ? -1.0 : 0.0;
  }
}

//! Scale \p values to unit sum, carrying tangents through the quotient rule.
/*! Each tangent `d` becomes `d / S - values * sum(d) / S^2` with `S = sum(values)`
    (floored at \p floor), then `values /= S`. Returns `S`. The normalisation a
    response and every convolved column go through, and its derivative, once. */
inline double normalize_unit_sum(double* values, std::size_t n, const std::vector<double*>& tangents,
                                 double floor = 1e-300) {
  double S = 0.0;
  for (std::size_t i = 0; i < n; ++i) S += values[i];
  S = std::max(S, floor);
  for (double* d : tangents) {
    double sd = 0.0;
    for (std::size_t i = 0; i < n; ++i) sd += d[i];
    for (std::size_t i = 0; i < n; ++i) d[i] = d[i] / S - values[i] * sd / (S * S);
  }
  for (std::size_t i = 0; i < n; ++i) values[i] /= S;
  return S;
}

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_RESPONSE_FUNCTION_H

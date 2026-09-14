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

#include <algorithm>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! How a measured response is cleaned before it is shifted and normalised.
/*! Inactive, the response is used as measured. Active -- which a node is when
    its description sets a response window or wires a response background --
    it is ChiSurf's IRF preparation, in its order: the constant background
    (a lamp or dark count level) is subtracted, what goes negative is clipped
    to zero, and everything outside `[start, stop)` is zeroed. */
struct ResponsePreparation {
  bool active = false;
  double background = 0.0;
  int start = 0;
  //! Exclusive; negative means the end of the response.
  int stop = -1;

  bool operator==(const ResponsePreparation& other) const {
    return active == other.active && background == other.background &&
           start == other.start && stop == other.stop;
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

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_RESPONSE_FUNCTION_H

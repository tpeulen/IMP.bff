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

//! Shift \p response by \p timeshift channels and scale it to unit sum.
/*! Sign: ChiSurf's `shift_array(v, s)` is tttrlib's `shift_lamp(v, -s)`. The
    two index in opposite directions *and* interpolate toward opposite
    neighbours, and the two flips cancel exactly. At `s == 0` they differ
    (`shift_lamp` drops the last sample), which is why a zero shift copies.
    \param[in] response the measured response
    \param[in] timeshift the shift, in channels
    \param[in,out] shifted scratch for the shifted copy, reused across calls
    \param[out] out the prepared response, as long as \p response
*/
inline void prepare_response(const std::vector<double>& response,
                             double timeshift, std::vector<double>& shifted,
                             std::vector<double>& out) {
  const int n_points = static_cast<int>(response.size());
  const double* source = response.data();
  if (timeshift != 0.0) {
    shifted.resize(response.size());
    shift_lamp_ad<double>(shifted.data(), response.data(), -timeshift,
                          n_points, 0.0);
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

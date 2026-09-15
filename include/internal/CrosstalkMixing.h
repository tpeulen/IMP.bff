/**
 * \file IMP/bff/internal/CrosstalkMixing.h
 * \brief The arithmetic of crosstalk mixing, header-only and templated: the one
 *        place a source-by-detector matrix is applied to source signals.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * `out[d, item] = sum_i sources[i, item] * M[i, d]` -- a chromophore's light (or a
 * pulse's excitation) distributed over detectors (or chromophores) by the rows of
 * `M`. `PhotophysicsCrosstalkMatrix.h`'s `crosstalk_apply_mixing` and the Bayesian
 * decay model's emission step both evaluate it here (PRD-143 #13, A4.5, 2026-09-15;
 * tpeulen: "the crosstalk mdl for excitation and emission should be only there once").
 * Templated on the number type so dual numbers pass through.
 */
#ifndef IMPBFF_INTERNAL_CROSSTALK_MIXING_H
#define IMPBFF_INTERNAL_CROSSTALK_MIXING_H

#include <IMP/bff/bff_config.h>
#include <cstddef>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! `out[d * n_items + item] = sum_i sources[i * n_items + item] * matrix[i * n_detectors + d]`.
//! `matrix` `n_sources x n_detectors`, `sources` `n_sources x n_items`, `out`
//! `n_detectors x n_items`, all row-major; `out` must not alias `sources`.
template <typename T, typename M>
inline void crosstalk_mix(const M* matrix, std::size_t n_sources, std::size_t n_detectors, const T* sources,
                          std::size_t n_items, T* out) {
  for (std::size_t d = 0; d < n_detectors; ++d)
    for (std::size_t item = 0; item < n_items; ++item) {
      T s = T(0.0);
      for (std::size_t i = 0; i < n_sources; ++i) s += sources[i * n_items + item] * matrix[i * n_detectors + d];
      out[d * n_items + item] = s;
    }
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INTERNAL_CROSSTALK_MIXING_H */

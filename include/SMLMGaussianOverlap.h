/** \file IMP/bff/SMLMGaussianOverlap.h
 *  \brief Persistent spatial-tree overlap of measured and model Gaussians.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLMGAUSSIANOVERLAP_H
#define IMPBFF_SMLMGAUSSIANOVERLAP_H

#include <IMP/bff/SMLM.h>

IMPBFF_BEGIN_NAMESPACE

//! Normalized mixture cross overlap and negative-log-overlap derivative.
struct IMPBFFEXPORT SMLMGaussianOverlapResult {
  double overlap = 0.0;
  double score = 0.0;
  //! One data-normalized overlap per model component, before model weighting.
  std::vector<double> component_overlaps;
  //! Flat M x 3 derivative of SCORE with respect to each model centre.
  std::vector<double> gradient;
  IMP_SHOWABLE_INLINE(SMLMGaussianOverlapResult,
      out << "SMLMGaussianOverlapResult(" << overlap << ")");
};

//! Own one copy of an existing localization index, including its spatial tree.
/** The input index may be destroyed after construction. No reference, callback,
    tree rebuild or data marshalling is needed during repeated evaluate() calls.
    This evaluates Gaussian PAIRS exactly inside a Mahalanobis cutoff; nodes
    are conservatively pruned, never replaced by moment-matched Gaussians.
    No self-overlap/cosine normalization or mixture compression is performed. */
class IMPBFFEXPORT SMLMGaussianOverlap {
 public:
  explicit SMLMGaussianOverlap(const SMLMIndex& index);
  int get_number_of_localizations() const;

  //! Batch model-Gaussian overlap with the fixed measured Gaussian mixture.
  /** Coordinates are flat M x 3. Covariances are flat M x 9, row-major FULL
      covariance matrices, in squared coordinate units. They must be finite,
      symmetric and positive semidefinite; zero covariance is a point model.
      Symmetry/PSD tolerance is 1e-12 times the matrix's maximum absolute entry.
      Tolerated asymmetry is symmetrized and tolerated negative eigenvalues
      are clamped to zero. Measured covariance is diag(index.sigmas^2).

      Both mixtures' finite nonnegative weights are normalized independently;
      omitted model weights are one. Overlap is sum(a_j b_i N(m_j-d_i;
      C_j+diag(sigma_i^2))). The cutoff is inclusive and removes pair tails
      without redistribution. Zero total weight or an empty mixture gives
      overlap zero, score -log(background), and zero gradient.

      score = -log(overlap + background); lower is better. The positive finite
      additive floor and overlap have inverse-volume units, so absolute scores
      depend on the coordinate unit. Gradient is analytic away from cutoff
      boundaries and includes both weight normalization and the global log.
      Unrepresentable pair covariance, overlap or score gradient is rejected. */
  SMLMGaussianOverlapResult evaluate(
      const std::vector<double>& model_coordinates,
      const std::vector<double>& model_covariances,
      const std::vector<double>& model_weights = std::vector<double>(),
      double cutoff_sigma = 6.0, double background = 1e-12) const;
  IMP_SHOWABLE_INLINE(SMLMGaussianOverlap,
      out << "SMLMGaussianOverlap(" << get_number_of_localizations()
          << " localizations)");

 private:
  SMLMIndex index_;
  std::vector<double> normalized_data_weights_;
  bool has_data_weight_ = false;
};

IMPBFF_END_NAMESPACE
#endif  // IMPBFF_SMLMGAUSSIANOVERLAP_H

/** \file IMP/bff/SMLM.h
 *  \brief Owning 3D localization index, likelihoods and selective averaging.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLM_H
#define IMPBFF_SMLM_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <map>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Weighted mean negative log likelihood and its model-coordinate derivative.
struct IMPBFFEXPORT SMLMScoreResult {
  double score = 0.0;
  std::vector<double> densities;
  //! Flat M x 3; derivative of score, including query-weight normalization.
  std::vector<double> gradient;
  IMP_SHOWABLE_INLINE(SMLMScoreResult, out << "SMLMScoreResult(" << score << ")");
};

//! Immutable owning balanced KD-tree over three-dimensional localizations.
/** All positions, sigmas, grid spacings and translations use the SAME physical
    unit, chosen by the caller. Coordinates are flat N x 3; 2D is not supported.
    Sigmas are flat N x 3 standard deviations along the source axes, or N
    isotropic standard deviations (expanded to N x 3). They must be finite and
    positive. Weights default to one and must be finite and nonnegative.
    Particle IDs default to -1 (unassigned); negative IDs cannot be averaged.
    Input buffers are copied once; subsequent searches/scoring remain native.

    KDE uses the normalized 3D Gaussian coefficient and an ellipsoidal
    Mahalanobis cutoff. Truncation removes the tail without renormalizing it.
    Thus integral approaches one with increasing cutoff when weights are
    normalized. Derivatives are analytic away from the cutoff boundary. */
class IMPBFFEXPORT SMLMIndex {
 public:
  SMLMIndex(const std::vector<double>& coordinates,
            const std::vector<double>& sigmas,
            const std::vector<double>& weights = std::vector<double>(),
            const std::vector<int>& particle_ids = std::vector<int>(),
            int leaf_size = 16);
  int get_number_of_localizations() const;
  const std::vector<double>& get_coordinates() const { return coordinates_; }
  const std::vector<double>& get_sigmas() const { return sigmas_; }
  const std::vector<double>& get_weights() const { return weights_; }
  const std::vector<int>& get_particle_ids() const { return particle_ids_; }
  std::vector<int> get_particle_localizations(int particle_id) const;
  //! Original input indices, sorted by index; radius is inclusive, Euclidean.
  std::vector<int> radius_search(const std::vector<double>& point,
                               double radius) const;
  //! Sorted by distance then original index. Negative max_distance is unlimited.
  std::vector<int> nearest_search(const std::vector<double>& point, int k = 1,
                                double max_distance = -1.0) const;
  std::vector<double> evaluate_density(
      const std::vector<double>& query_coordinates, double cutoff_sigma = 4.0,
      bool normalize_weights = true) const;
  //! score = -sum(qweight * log(density + background)) / sum(qweight).
  /*! Positive finite background is an additive density floor, in inverse
      volume units. Zero total localization weight gives background-only
      scores; empty queries or zero total query weight give score/gradient 0. */
  SMLMScoreResult evaluate_score(
      const std::vector<double>& query_coordinates,
      const std::vector<double>& query_weights = std::vector<double>(),
      double cutoff_sigma = 4.0, double background = 1e-12,
      bool normalize_weights = true) const;
  IMP_SHOWABLE_INLINE(SMLMIndex,
      out << "SMLMIndex(" << get_number_of_localizations() << " localizations)");

 private:
  friend class SMLMGaussianOverlap;
  struct Node {
    double lower[3], upper[3], max_sigma[3];
    int begin = 0, end = 0, left = -1, right = -1;
  };
  std::vector<double> coordinates_, sigmas_, weights_, log_normalizers_;
  std::vector<int> particle_ids_, order_;
  std::vector<Node> nodes_;
  std::map<int, std::vector<int> > particles_;
  double total_weight_ = 0.0;
  int leaf_size_;
  int build_node(int begin, int end);
  double evaluate_point(const double* point, double cutoff_sigma,
                        bool normalize_weights, double* gradient) const;
};

//! Particle quality gates and map normalization.
struct IMPBFFEXPORT SMLMAverageOptions {
  double cutoff_sigma = 4.0;
  //! Each accepted particle has unit total localization weight before averaging.
  bool normalize_particles = true;
  //! Divide by accepted particle count (above) or accepted total weight (below).
  /*! Half maps use their OWN denominators. False returns the weighted sum.
      Maps are continuous densities, not voxel probabilities; integrate with
      spacing[0]*spacing[1]*spacing[2]. Clipped tails are not redistributed. */
  bool normalize_density = true;
  int min_localizations = 1;
  //! Zero means no upper count gate. Counts include zero-weight localizations.
  int max_localizations = 0;
  bool use_score_gate = false;
  double min_score = 0.0;
  double max_score = 1.0;
  //! Optional one finite quality score per REQUESTED particle, in request order.
  std::vector<double> particle_scores;
  IMP_SHOWABLE_INLINE(SMLMAverageOptions,
      out << "SMLMAverageOptions(cutoff=" << cutoff_sigma << ")");
};

//! Selective average and independent particle half maps; no imposed symmetry.
struct IMPBFFEXPORT SMLMAverageResult {
  //! Flat x-fast grid: x + nx*(y + ny*z). Origin is voxel (0,0,0)'s centre.
  std::vector<double> values, half1, half2;
  std::vector<int> grid_shape;
  std::vector<double> origin, spacing;
  //! Accepted IDs and their row-major source -> reference 3 x 4 transforms.
  std::vector<int> selected_ids, rejected_ids, localization_counts;
  std::vector<double> transforms;
  //! Alternating accepted particles (even/odd accepted ordinal), never points.
  std::vector<int> half1_particle_ids, half2_particle_ids;
  int selected_localizations = 0;
  int half1_localizations = 0;
  int half2_localizations = 0;
  IMP_SHOWABLE_INLINE(SMLMAverageResult,
      out << "SMLMAverageResult(" << selected_ids.size() << " particles)");
};

//! Deposit only selected particles, rotating the FULL anisotropic covariance.
/** Selection must be nonempty, unique, nonnegative, and present in the index.
    Transforms contain 12 numbers per requested ID, row-major [R | t]. R must
    be a proper orthonormal rotation (tolerance 1e-6). Rejected particles keep
    their diagnostic ID but do not enter any normalization or half map.
    Empty transforms means identity for every requested particle. Grid shape,
    origin and spacing each have three entries; spacing must be positive.
    Accumulation visits only voxels inside each rotated kernel's support box. */
IMPBFFEXPORT SMLMAverageResult average_smlm_particles(
    const SMLMIndex& index, const std::vector<int>& particle_ids,
    const std::vector<double>& transforms, const std::vector<int>& grid_shape,
    const std::vector<double>& origin, const std::vector<double>& spacing,
    const SMLMAverageOptions& options = SMLMAverageOptions());

//! Bounded native translation refinement against a fixed localization KDE.
struct IMPBFFEXPORT SMLMRegistrationOptions {
  int max_iterations = 100;
  double max_translation_step = 1.0;
  double tolerance = 1e-6;
  double cutoff_sigma = 4.0;
  double background = 1e-12;
  IMP_SHOWABLE_INLINE(SMLMRegistrationOptions,
      out << "SMLMRegistrationOptions(" << max_iterations << " iterations)");
};

struct IMPBFFEXPORT SMLMRegistrationResult {
  std::vector<double> transform;
  double initial_score = 0.0, score = 0.0;
  int iterations = 0;
  bool converged = false;
  IMP_SHOWABLE_INLINE(SMLMRegistrationResult,
      out << "SMLMRegistrationResult(score=" << score << ")");
};

//! Backtracking gradient descent; preserves the supplied rotation exactly.
/*! Initial transform is empty (identity) or a validated source -> reference
    3 x 4 rigid transform. No callbacks or iterative Python marshalling.
    This local optimizer needs initial overlap with the truncated KDE. */
IMPBFFEXPORT SMLMRegistrationResult refine_smlm_translation(
    const SMLMIndex& reference, const std::vector<double>& moving_coordinates,
    const std::vector<double>& initial_transform = std::vector<double>(),
    const std::vector<double>& query_weights = std::vector<double>(),
    const SMLMRegistrationOptions& options = SMLMRegistrationOptions());

IMPBFF_END_NAMESPACE
#endif  // IMPBFF_SMLM_H

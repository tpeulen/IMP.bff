/** \file IMP/bff/SMLMLikelihood.h
 *  \brief Forward observation likelihood for a discrete expected emitter model.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLMLIKELIHOOD_H
#define IMPBFF_SMLMLIKELIHOOD_H
#include <IMP/bff/SMLM.h>

IMPBFF_BEGIN_NAMESPACE

//! Single-channel discrete LocMoFit likelihood settings.
/** Implements Wu et al., Nat Methods (2023), DOI 10.1038/s41592-022-01676-z:
    discrete observation model Eq. 3, normalized contamination mixture Eq. 4,
    and independent additional uncertainty Eq. 16. Eq. 5 describes multi-color
    channel weighting; observation weights here support weighted log likelihoods.
    No model-point precision is used as observation noise.

    All coordinates, measured sigmas, intrinsic_sigma and ROI bounds have the
    same caller-chosen physical unit. ROI bounds MUST explicitly contain all
    observations, including zero-weight ones; no automatic position filtering.
    Uniform background density is 1 / prod(roi_max - roi_min). Gaussian support
    is not renormalized to the ROI or after Mahalanobis truncation at cutoff_sigma
    (default 6). This is the infinite-space Gaussian approximation used by the
    paper, with a numerical tail cutoff; fitting is conditional on fixed ROI. */
struct IMPBFFEXPORT SMLMLikelihoodOptions {
  std::vector<double> roi_min, roi_max;
  double background_fraction = 0.01;
  //! Independent isotropic standard deviation; variance is sigma_i^2 + eps^2.
  double intrinsic_sigma = 0.0;
  double cutoff_sigma = 6.0;
  bool compute_model_gradient = true;
  double get_roi_volume() const;
  IMP_SHOWABLE_INLINE(SMLMLikelihoodOptions,
      out << "SMLMLikelihoodOptions(background=" << background_fraction << ")");
};

//! Observation PDF and both unweighted and weighted log likelihoods.
struct IMPBFFEXPORT SMLMLikelihoodResult {
  //! -sum(observation_weight * log(pdf)) / sum(observation_weight).
  double mean_nll = 0.0;
  //! Actual sum_i log(pdf_i), independent of observation weights, for AIC/AICc.
  double log_likelihood = 0.0;
  double weighted_log_likelihood = 0.0;
  double total_observation_weight = 0.0;
  int localization_count = 0;
  //! Positive-weight observations with at least one supported model emitter.
  int supported_localizations = 0;
  std::vector<double> pdf, log_pdf, model_pdf, signal_fraction;
  //! Derivative of mean_nll: [tx,ty,tz,omega_x,omega_y,omega_z].
  /*! Rotation derivatives are for R_new = exp([omega]x)*R, t fixed. */
  std::vector<double> pose_gradient;
  //! Row-major 6x6 positive Gauss-Newton approximation, used by the native fit.
  /*! Includes signal responsibilities; not the exact mixture NLL Hessian. */
  std::vector<double> pose_information;
  //! Flat J x 3 derivative of mean_nll in original model-coordinate axes.
  std::vector<double> model_gradient;
  IMP_SHOWABLE_INLINE(SMLMLikelihoodResult,
      out << "SMLMLikelihoodResult(nll=" << mean_nll << ")");
};

//! Persistent owning expected emitter positions and normalized model weights.
/** A point model has no measurement precision: its private balanced SMLMIndex
    is used ONLY for geometric neighborhood searches. At each localization the
    observation's own anisotropic sigma (plus intrinsic variance) determines the
    blur. A rigid model -> observation pose inverse-transforms search centres,
    so the model KD-tree is reusable across all fitting iterations.
    Model coordinates are flat J x 3, J > 0. Weights default to uniform and must
    be finite, nonnegative, with positive total. They are normalized once. */
class IMPBFFEXPORT SMLMPointModel {
 public:
  SMLMPointModel(const std::vector<double>& coordinates,
                const std::vector<double>& weights = std::vector<double>(),
                int leaf_size = 16);
  int get_number_of_points() const { return model_index_.get_number_of_localizations(); }
  const std::vector<double>& get_coordinates() const { return model_index_.get_coordinates(); }
  const std::vector<double>& get_weights() const { return model_index_.get_weights(); }
  //! Empty pose is identity; otherwise row-major rigid model -> observation 3x4.
  /*! No arbitrary additive floor: zero likelihood gives log_pdf=-infinity.
      Mean NLL is infinity when a positive-weight observation has zero PDF;
      gradients are defined as zero there, and fitting reports no convergence.
      An empty observation cloud or zero total weight gives mean NLL/gradients 0.
      The unweighted log likelihood still includes every supplied observation. */
  SMLMLikelihoodResult evaluate(
      const SMLMIndex& observations, const SMLMLikelihoodOptions& options,
      const std::vector<double>& pose = std::vector<double>()) const;
  IMP_SHOWABLE_INLINE(SMLMPointModel,
      out << "SMLMPointModel(" << get_number_of_points() << " emitters)");
 private:
  SMLMIndex model_index_;
};

//! Native rigid optimization limits; fixed model shape, blur and contamination.
struct IMPBFFEXPORT SMLMLikelihoodFitOptions {
  int max_iterations = 200;
  double max_translation_step = 1.0;
  double max_rotation_step = 0.05;
  double tolerance = 1e-7;
  int max_backtracks = 40;
  IMP_SHOWABLE_INLINE(SMLMLikelihoodFitOptions,
      out << "SMLMLikelihoodFitOptions(" << max_iterations << " iterations)");
};

struct IMPBFFEXPORT SMLMLikelihoodFitResult {
  //! Rigid model -> observation transform; no scale optimization.
  std::vector<double> transform;
  double initial_mean_nll = 0.0;
  int iterations = 0;
  bool converged = false;
  SMLMLikelihoodResult likelihood;
  IMP_SHOWABLE_INLINE(SMLMLikelihoodFitResult,
      out << "SMLMLikelihoodFitResult(nll=" << likelihood.mean_nll << ")");
};

//! Local 6D backtracking descent; requires nonzero model/observation overlap.
/*! A background-only plateau never counts as converged. All model and
    observation arrays and the model tree stay native throughout the fit. */
IMPBFFEXPORT SMLMLikelihoodFitResult fit_smlm_likelihood_rigid(
    const SMLMIndex& observations, const SMLMPointModel& model,
    const SMLMLikelihoodOptions& likelihood_options,
    const std::vector<double>& initial_pose = std::vector<double>(),
    const SMLMLikelihoodFitOptions& fit_options = SMLMLikelihoodFitOptions());

//! Use the actual summed log likelihood, never a mean NLL, as the input.
IMPBFFEXPORT double smlm_aic(double log_likelihood, int free_parameters);
//! Actual localization count n; infinity when n <= free_parameters + 1.
IMPBFFEXPORT double smlm_aicc(double log_likelihood, int localization_count,
                            int free_parameters);

IMPBFF_END_NAMESPACE
#endif  // IMPBFF_SMLMLIKELIHOOD_H

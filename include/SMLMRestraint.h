/** \file IMP/bff/SMLMRestraint.h
 *  \brief Native structural-model restraint against fixed SMLM observations.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLMRESTRAINT_H
#define IMPBFF_SMLMRESTRAINT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/SMLMLikelihood.h>
#include <IMP/bff/SMLMGaussianOverlap.h>
#include <IMP/Restraint.h>
#include <IMP/Particle.h>
#include <IMP/particle_index.h>
#include <memory>
#include <string>

IMPBFF_BEGIN_NAMESPACE

//! Expected emitter particles scored against an owned measured SMLM cloud.
/** IMP-only connection layer. All emitter particles must have XYZ coordinates
    in the supplied model and use the same physical unit as the observations.
    Observations/options/weights are copied at construction; their originating
    Python or C++ objects may be destroyed afterwards. The measured cloud stays
    native. Each evaluation reads live emitter coordinates and rebuilds only
    their small model KD-tree, then adds analytic derivatives through XYZ and
    the caller's DerivativeAccumulator. No Python objective callbacks.

    Default score is the observation-weighted mean negative log likelihood,
    in nats per weighted localization. It includes normalized model weights,
    observation-specific precision, independent intrinsic variance and the
    explicit ROI contamination mixture from SMLMLikelihoodOptions. Optional
    sum mode returns -weighted_log_likelihood, with derivatives multiplied by
    the fixed total observation weight. Neither is a physical kcal/mol energy;
    absolute density log scores depend on the coordinate unit.

    Use IMP's own restraint/RestraintSet weight to scale this score: no second
    multiplicative restraint weight is introduced. XYZ radii and atom masses
    never supply measurement precision or emitter weights. Finite-support
    derivatives inherit the core likelihood's cutoff-boundary limitation. */
class IMPBFFEXPORT SMLMRestraint : public IMP::Restraint {
 public:
  SMLMRestraint(IMP::Model* model, const IMP::ParticlesTemp& emitter_particles,
               const SMLMIndex& observations,
               const SMLMLikelihoodOptions& options,
               const std::vector<double>& model_weights = std::vector<double>(),
               bool sum_negative_log_likelihood = false, int leaf_size = 16,
               std::string name = "SMLMRestraint%1%");

  //! Optional Gaussian-mixture cross-overlap instead of forward observation MLE.
  /** Covariances are fixed M x 9 FULL matrices in the observation frame,
      in squared coordinate units, finite symmetric PSD as validated by the
      native overlap core. They are independent of observed precision, masses
      and XYZ radii. They do not rotate automatically with a rigid-body member:
      use isotropic/zero covariance or supply a restraint matching that frame.
      Score is -log(normalized_cross_overlap + background), in nats, with a
      persistent fixed measured Gaussian tree. No sum/mean option in this mode. */
  SMLMRestraint(IMP::Model* model, const IMP::ParticlesTemp& emitter_particles,
               const SMLMIndex& observations,
               const std::vector<double>& model_covariances,
               const std::vector<double>& model_weights = std::vector<double>(),
               double cutoff_sigma = 6.0, double background = 1e-12,
               std::string name = "SMLMGaussianOverlapRestraint%1%");

  IMP::ParticleIndexes get_particle_indexes() const { return emitter_indexes_; }
  const std::vector<double>& get_model_weights() const { return model_weights_; }
  int get_number_of_localizations() const;
  std::string get_score_mode() const;
  bool get_uses_sum_score() const { return sum_negative_log_likelihood_; }

  double unprotected_evaluate(IMP::DerivativeAccumulator* accumulator) const override;
  IMP::ModelObjectsTemp do_get_inputs() const override;
  IMP_OBJECT_METHODS(SMLMRestraint);

 private:
  IMP::ParticleIndexes emitter_indexes_;
  //! Exactly one of these owns the measured cloud, according to scoring mode.
  std::unique_ptr<SMLMIndex> observations_;
  std::unique_ptr<SMLMGaussianOverlap> overlap_;
  SMLMLikelihoodOptions options_;
  std::vector<double> model_weights_, model_covariances_;
  bool sum_negative_log_likelihood_ = false;
  int leaf_size_ = 16;
  double cutoff_sigma_ = 6.0, background_ = 1e-12;
  void initialize_emitters(const IMP::ParticlesTemp& particles);
  std::vector<double> get_emitter_coordinates() const;
};
IMP_OBJECTS(SMLMRestraint, SMLMRestraints);

IMPBFF_END_NAMESPACE
#endif  // IMPBFF_SMLMRESTRAINT_H

/** \file IMP/bff/SMLMParticleRegistration.h
 *  \brief Native reference-based rotational registration of SMLM particles.
 */
#ifndef IMPBFF_SMLMPARTICLEREGISTRATION_H
#define IMPBFF_SMLMPARTICLEREGISTRATION_H
#include <IMP/bff/SMLMParticles.h>
#include <IMP/bff/SMLMLikelihood.h>
IMPBFF_BEGIN_NAMESPACE

struct IMPBFFEXPORT SMLMParticleRegistrationOptions {
  int angular_samples = 24;
  //! Full circle by default. A smaller interval requires known MODEL symmetry.
  double angular_period = 6.283185307179586;
  double intrinsic_sigma = 0.0;
  double background_fraction = 0.05;
  //! Source-space ROI is each group's bounding box plus this fixed padding.
  double roi_padding = 100.0;
  double min_signal_fraction = 0.1;
  int max_iterations = 80;
  double max_translation_step = 30.0;
  double max_rotation_step = 0.05;
  IMP_SHOWABLE_INLINE(SMLMParticleRegistrationOptions,
      out << "SMLMParticleRegistrationOptions(" << angular_samples << " angles)");
};

struct IMPBFFEXPORT SMLMParticleRegistrationResult {
  std::vector<int> particle_ids, rejected_ids, iterations, converged;
  //! Proper source -> reference transforms, 12 per accepted particle.
  std::vector<double> transforms, initial_nll, final_nll, signal_fraction;
  IMP_SHOWABLE_INLINE(SMLMParticleRegistrationResult,
      out << "SMLMParticleRegistrationResult(" << particle_ids.size() << " particles)");
};

//! Coarse azimuth search followed by full 6D observation-likelihood refinement.
/*! Each source particle is fitted independently to a fixed external emitter
    model. Initial frames may be PCA or caller-supplied source -> reference
    rigid transforms. Full measured per-observation anisotropic noise is used
    in source axes; only the model is transformed during fitting. The best two
    angular starts are refined. No particle coordinates are snapped, scaled,
    duplicated or replaced by reference model positions. This is reference-
    based averaging; model symmetry/label/linker assumptions must be recorded. */
IMPBFFEXPORT SMLMParticleRegistrationResult register_smlm_particles(
    const SMLMIndex& index, const SMLMParticleFrames& initial_frames,
    const SMLMPointModel& model,
    const SMLMParticleRegistrationOptions& options = SMLMParticleRegistrationOptions());
IMPBFF_END_NAMESPACE
#endif

/** \file IMP/bff/SMLMParticles.h
 *  \brief Native particle initialization, rigid refinement and map export.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLMPARTICLES_H
#define IMPBFF_SMLMPARTICLES_H
#include <IMP/bff/SMLM.h>
#include <string>
IMPBFF_BEGIN_NAMESPACE

//! Per-particle source-to-reference frames and measured shape diagnostics.
struct IMPBFFEXPORT SMLMParticleFrames {
  std::vector<int> particle_ids, localization_counts;
  std::vector<double> transforms, centers, eigenvalues;
  IMP_SHOWABLE_INLINE(SMLMParticleFrames,
      out << "SMLMParticleFrames(" << particle_ids.size() << " particles)");
};

//! Center each requested particle and align its least-variance axis to +z.
/*! Independent particle initialization, without a structural template or
    rotational symmetry. The normal's sign is chosen against source +z and
    the minimal rotation preserves the unobserved in-plane orientation.
    Huber weights cap the influence of distant points at robust_radius
    (same units as coordinates); zero disables robust weighting. At least
    three positive-weight, noncollinear points are required per particle.
    The original measurements and covariance remain unchanged in the index. */
IMPBFFEXPORT SMLMParticleFrames align_smlm_particles(
    const SMLMIndex& index, const std::vector<int>& particle_ids,
    double robust_radius = 0.0, int min_localizations = 3,
    int max_localizations = 0);

//! Keep positive-weight rows whose measured precision passes each axis limit.
/*! Empty limits disables the precision gate. Unassigned rows remain available
    for scoring but are excluded from automatic particle frame selection.
    No position/radius mask or replacement display precision is applied. */
IMPBFFEXPORT SMLMIndex select_smlm_precision(
    const SMLMIndex& index,
    const std::vector<double>& max_sigma = std::vector<double>());

//! Apply a proper rigid row-major 3x4 transform to a flat coordinate array.
IMPBFFEXPORT std::vector<double> transform_smlm_points(
    const std::vector<double>& coordinates, const std::vector<double>& transform);

//! Fixed assembly axis and landmark-centered origin from proper rigid operators.
/*! Operators are flat N x 12. Solve (I-R)c=t for the common rotational axis;
    center its free axial coordinate on the mean landmark projection. Returns
    centre[3], unit axis[3], maximum fixed-axis residual. No scale fit. */
IMPBFFEXPORT std::vector<double> get_smlm_assembly_frame(
    const std::vector<double>& operators, const std::vector<double>& landmarks);

//! Refine translation AND rotation using native KDE likelihood gradients.
/*! Local backtracking descent; max_rotation_step is radians per iteration.
    Requires initial overlap. No scaling, symmetry, or shape constraints are
    introduced. The options' max_translation_step uses the coordinate unit. */
IMPBFFEXPORT SMLMRegistrationResult refine_smlm_rigid(
    const SMLMIndex& reference, const std::vector<double>& moving_coordinates,
    const std::vector<double>& initial_transform,
    const std::vector<double>& query_weights = std::vector<double>(),
    const SMLMRegistrationOptions& options = SMLMRegistrationOptions(),
    double max_rotation_step = 0.05);

//! Export the average or a particle half map directly from native storage.
/*! map=0 average, 1 first half, 2 second half. All physical units MUST be
    Angstrom for MRC. Isotropic spacing required by the existing MRC writer.
    Transposes x-fast SMLM storage to the writer's C-order contract in C++. */
IMPBFFEXPORT void write_smlm_average_mrc(
    const SMLMAverageResult& average, const std::string& path, int map = 0);

IMPBFF_END_NAMESPACE
#endif

/**
 *  \file IMP/bff/ProbeTrajectoryDensity.h
 *  \brief Where a probe went over a trajectory: region densities, distance
 *         profiles and its orientation against a fixed component.
 *
 *  What `imp_bff analyze-trajectories` computes (formerly
 *  `IMP.bff.analysis`, then a command of the Python program `bin/imp_bff`).
 *  A simulation writes an RMF whose root holds a fixed component (a host) and
 *  a mobile one (a probe); for every frame the fixed component's
 *  minimal-variance axis is the frame of reference, and every atom of every
 *  region of the probe -- regions are the coloured features of its component
 *  template -- is recorded relative to it. What comes out, per probe:
 *
 *  - `occupancy_<region>.mrc`, the region's atoms as a density map;
 *  - `radial_<region>.csv`, their distances from the fixed centre;
 *  - `axis_z_profile_regions.csv` and `axis_xy_profile_regions.csv`, their
 *    positions along and distances from the axis;
 *  - `axis_mobile_vs_fixed_orientation.csv`, the angle between the probe's
 *    long axis (its non-linker atoms) and the fixed axis, per frame;
 *  - `fixed_axis_definition.json`, the axis and the angle statistics.
 *
 *  The array kernels are TrajectoryAnalysis.h; this is the IMP half.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_PROBETRAJECTORYDENSITY_H
#define IMPBFF_PROBETRAJECTORYDENSITY_H

#include <IMP/bff/bff_config.h>

#include <IMP/Particle.h>
#include <IMP/base_types.h>

#include <map>
#include <ostream>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The name an analysis gives an atom particle.
/*!
    With a lookup (site serial -> name, from the system CIF): the name of the
    atom's input index, or of the index plus one. Otherwise the atom type
    without its `HET:` prefix, or the particle name for a particle that is not
    an atom. A slash-separated name keeps its last part.
*/
IMPBFFEXPORT std::string probe_atom_name(
        IMP::Particle* atom,
        const std::map<int, std::string>& site_name_lookup = std::map<int, std::string>());

//! The fixed component's centre and minimal-variance axis.
/*!
    \param[in] atoms the fixed component's atoms
    \param[in] site_name_lookup names by site serial; see #probe_atom_name
    \param[in] ref_axis flip the axis to point along this one; empty for none
    \param[in] axis_element flip the axis to point toward the centroid of the
               atoms whose name starts with this; empty for none
    \return six values: centre, then unit axis
*/
IMPBFFEXPORT std::vector<double> probe_fixed_axis(
        const IMP::ParticlesTemp& atoms,
        const std::map<int, std::string>& site_name_lookup = std::map<int, std::string>(),
        const std::vector<double>& ref_axis = std::vector<double>(),
        const std::string& axis_element = "");

//! The maximal-variance axis of some atoms, flipped along \p ref_axis.
/*! \return three values, or none when there are fewer than two atoms */
IMPBFFEXPORT std::vector<double> probe_long_axis(
        const IMP::ParticlesTemp& atoms,
        const std::vector<double>& ref_axis = std::vector<double>());

//! Points as a density map in MRC format, standard-normalised.
/*!
    Every point becomes a unit-mass sphere of radius 1 A, sampled by
    `IMP::em::particles2density` at \p resolution on a \p voxel_size grid.
    \param[in] points flat, three per point
    \throw ValueException when there are no points
*/
IMPBFFEXPORT void write_probe_region_density(const std::vector<double>& points,
                                             const std::string& path, double resolution,
                                             double voxel_size);

//! What #analyze_probe_trajectories is told.
struct IMPBFFEXPORT ProbeTrajectoryDensityOptions {
  //! The run directory holding `<system>/rmfs/0.rmf3` (or `traj.rmf3`, `traj_init.rmf3`).
  std::string traj_root;
  //! Combine every `run*/` directory under #runs_root (default `<traj_root>/runs`).
  bool combine_runs;
  std::string runs_root;
  std::string output_dir;
  //! Mobile component names, one analysis each.
  std::vector<std::string> mobiles;
  //! Template CIFs, one per mobile in order; a mobile past the end uses
  //! `<template dir>/<mobile>.template.cif`.
  std::vector<std::string> mobile_template_cifs;
  //! The fixed component's name; inferred from the non-mobile children when empty.
  std::string fixed_name;
  //! The run subdirectory; `<fixed>_<mobile>_imp`, or auto-detected, when empty.
  std::string system_name;
  //! Orient the fixed axis toward the atoms whose name starts with this.
  std::string axis_element;
  double resolution, voxel_size, bin_width;
  //! 0 reads every frame.
  int max_frames;

  ProbeTrajectoryDensityOptions()
      : traj_root("traj_latest"), combine_runs(false), output_dir("analysis/latest"),
        resolution(4.0), voxel_size(1.0), bin_width(0.5), max_frames(0) {}
};

#if !defined(SWIG) && (!defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF))
//! Run the whole analysis, writing the files listed at the top of this header.
/*!
    Reads the RMF through `IMP::rmf`, so it exists where that module is linked:
    the IMP module build, or a standalone build configured with
    `IMPBFF_WITH_IMP_RMF`.

    \param[in] options what to analyse and where to write it
    \param[out] log the progress lines the command prints
    \throw ValueException when a template is missing or a region has no points
    \throw IOException when no run directory or RMF is found
*/
IMPBFFEXPORT void analyze_probe_trajectories(const ProbeTrajectoryDensityOptions& options,
                                             std::ostream& log);
#endif

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PROBETRAJECTORYDENSITY_H

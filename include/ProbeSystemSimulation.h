/**
 *  \file IMP/bff/ProbeSystemSimulation.h
 *  \brief Propagate an explicit dye on a force-field system: MD, rigid-body
 *         Monte Carlo, or both.
 *
 *  The runner behind `imp_bff simulate`. It reads a force-field system mmCIF
 *  (#IMP::bff::read_forcefield_cif), loads the component structures, builds
 *  the bonded, steric, native-contact (Go) and optional centre-of-mass pull
 *  restraints (`ProbePotentialRestraints.h`), places the guest around the host
 *  by score, decides which sites move, minimises, and then samples in one of
 *  three modes:
 *
 *    simple_md      Langevin MD over the movable sites
 *    hybrid_md_mc   alternating rigid-body Monte Carlo on the mobile
 *                   component (scored by the steric term alone) and MD
 *    multi_restart  independent re-placements, each a short hybrid run,
 *                   keeping each restart's final frame
 *
 *  Writes `<output_root>/<fixed>_<mobile>_imp/{initial.0.rmf3, rmfs/0.rmf3,
 *  stat.0.out, system.cif}` and a copy of the system under `systems/`.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_PROBESYSTEMSIMULATION_H
#define IMPBFF_PROBESYSTEMSIMULATION_H

#include <IMP/bff/bff_config.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

#ifndef SWIG
//! The settings of one `imp_bff simulate` run; the defaults are the program's.
struct IMPBFFEXPORT ProbeSystemSimulationOptions {
  //! Where `<fixed>_<mobile>_imp/` goes; empty is `./output/trajs`.
  std::string output_root;
  //! MD steps; negative keeps the system CIF's `n_steps`.
  int md_steps = 20000;
  //! Steps per written frame (and per MC/MD block); negative keeps the CIF's.
  int write_every = 500;
  //! `simple_md`, `hybrid_md_mc` or `multi_restart`.
  std::string sampling_mode = "hybrid_md_mc";
  int mc_steps = 8;
  //! kT of the rigid-body Metropolis test, in score units.
  double mc_temperature = 2.0;
  //! Standard deviation of a rigid translation is a third of this, A.
  double rb_max_translation = 0.35;
  //! Standard deviation of a rigid rotation angle is a third of this, degrees.
  double rb_max_rotation_deg = 4.0;
  //! Langevin friction, 1/ps; NaN keeps the system CIF's.
  double friction_ps = std::numeric_limits<double>::quiet_NaN();
  //! MD temperature, K; NaN keeps the system CIF's.
  double temperature_k = std::numeric_limits<double>::quiet_NaN();
  //! The group that moves; empty infers `<mobile component>_all`.
  std::string mobile_group;
  //! `static` (fixed component rigid) or `flex` (its `_flex` group released).
  std::string fixed_flex_mode = "static";
  //! Fallback when the CIF declares no MD-fixed group: freeze the mobile rings.
  bool freeze_mobile_rings = true;
  bool init_placement = true;
  double init_distance_a = 62.5;
  int init_trials = 200;
  //! Seeds the placement search; the program also seeds IMP's generator with it.
  int init_seed = 42;
  //! Harmonic pull of the guest centre of mass to the host's; 0 is off.
  double com_pull_k = 0.0;
  double go_mobile_k = 3.0;
  double go_fixed_k = 2.0;
  double go_cutoff = 6.0;
  //! Progress line every this many written frames.
  int log_every_frames = 10;
  //! Restarts of `multi_restart`.
  int n_restarts = 1;
};

//! The system files a run covers: \p system_cif alone, or every `*.cif` in
//! \p systems_dir, sorted.
/*! \throw ValueException when neither is given or the directory has none */
IMPBFFEXPORT std::vector<std::string> probe_system_paths(const std::string& system_cif,
                                                         const std::string& systems_dir);

//! Run one force-field system; progress goes to \p out.
/*! \throw ValueException when the system is inconsistent or its groups and
           components cannot be resolved */
IMPBFFEXPORT void run_probe_system_simulation(const std::string& system_cif,
                                              const ProbeSystemSimulationOptions& options,
                                              std::ostream& out = std::cout);
#endif

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PROBESYSTEMSIMULATION_H

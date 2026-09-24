/**
 *  \file IMP/bff/FRETNetworkSimulation.h
 *  \brief Photons of a FRET network measurement, simulated from its model.
 *
 *  What FRETNetworkModel scores, generated: a molecule's joint state
 *  `(hidden, donor, acceptor)` walks the measurement's joint generator
 *  (Gillespie); while it holds a state, each channel detects signal photons at
 *  `lambda_c(s) - beta_c` and background at `beta_c`; a signal photon's
 *  microtime bin is drawn from the state's emission table (the per-state
 *  distance distribution mixed in, as the likelihood mixes it), a background
 *  photon's from the channel's background density.
 *
 *  Each molecule is one segment of `duration`. A freely diffusing molecule is
 *  imitated by a Gaussian focus crossing: its signal (not the background) is
 *  thinned by `exp(-2 ((t - duration/2) / w)^2)` with `w = duration / 4`, so
 *  brightness changes within the segment for reasons no state explains --
 *  what the conditional arrival model is for. select_bursts() then cuts
 *  segments the way a burst search does, which puts the burst-selection
 *  bias into simulated data too.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FRET_NETWORK_SIMULATION_H
#define IMPBFF_FRET_NETWORK_SIMULATION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FRETNetwork.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! How simulate_fret_measurement() generates photons.
class IMPBFFEXPORT FRETSimulationOptions {
 public:
  //! Molecules, each one segment.
  int n_molecules = 100;
  //! Length of each molecule's segment (macrotime units, the rates' unit).
  double duration = 1.0;
  //! Thin the signal by a Gaussian focus crossing (see the file comment).
  bool focus = false;
  //! Start each molecule from the measurement's start distribution
  //! (FRETMeasurement::get_start), else from the joint stationary one.
  bool product_start = true;
  unsigned int seed = 1;

  IMP_SHOWABLE_INLINE(FRETSimulationOptions,
                      out << "FRETSimulationOptions(" << n_molecules << " x " << duration << ")");
};
IMP_VALUES(FRETSimulationOptions, FRETSimulationOptionsList);

//! Simulated photons of one measurement, one segment per molecule.
/*! The true joint state at each photon is returned by
    simulated_fret_states() for the same arguments (same seed). */
IMPBFFEXPORT FRETPhotonData simulate_fret_measurement(const FRETHiddenProcess& process,
                                                      const FRETMeasurement& measurement,
                                                      const FRETSimulationOptions& options);

//! The true joint state at each photon of simulate_fret_measurement().
IMPBFFEXPORT std::vector<int> simulated_fret_states(const FRETHiddenProcess& process,
                                                    const FRETMeasurement& measurement,
                                                    const FRETSimulationOptions& options);

//! Burst selection by interphoton time.
/*! A burst is a maximal run of photons inside one input segment whose
    consecutive gaps are all at most `max_gap`, kept when it has at least
    `min_photons`. Returns the same photons with the bursts as segments
    (photons outside every burst stay in the arrays, in no segment). */
IMPBFFEXPORT FRETPhotonData select_bursts(const FRETPhotonData& data, double max_gap,
                                          int min_photons);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_FRET_NETWORK_SIMULATION_H */

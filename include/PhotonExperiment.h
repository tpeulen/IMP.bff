/** \file IMP/bff/PhotonExperiment.h
 *  \brief Photon experiments simulated by TTTRLib, reduced in native memory.
 *
 *  Photons and curves belong to TTTRLib; this is the thin bridge a model
 *  search needs to hand it a scenario and read back what an instrument would
 *  have recorded. Without TTTRLib in the build every call refuses.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_PHOTONEXPERIMENT_H
#define IMPBFF_PHOTONEXPERIMENT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! One simulated photon stream, reduced to per-channel micro-time histograms.
class IMPBFFEXPORT PhotonExperimentResult {
 public:
  PhotonExperimentResult();
  int get_number_of_channels() const { return n_channels_; }
  int get_number_of_microtime_bins() const { return n_microtime_bins_; }
  int get_number_of_photons() const { return n_photons_; }
  //! Channel-major, `channels * microtime_bins`.
  const std::vector<double>& get_microtime_histogram() const { return histogram_; }
  //! One channel's micro-time histogram.
  std::vector<double> get_channel_histogram(int channel) const;
  //! Arrival times of one channel's photons, in the scenario's macro-time unit.
  std::vector<double> get_arrival_times(int channel) const;
  //! How long the experiment ran, in the scenario's macro-time unit.
  double get_duration() const { return duration_; }
  IMP_SHOWABLE_INLINE(PhotonExperimentResult,
                      out << "PhotonExperimentResult(" << n_photons_ << " photons)");

 private:
  friend class PhotonExperiment;
  int n_channels_;
  int n_microtime_bins_;
  int n_photons_;
  double duration_;
  std::vector<double> histogram_;
  std::vector<int> channel_;
  std::vector<int> microtime_;
  std::vector<double> time_;
};
IMP_VALUES(PhotonExperimentResult, PhotonExperimentResults);

//! TTTRLib's photon simulator, driven from a model search.
class IMPBFFEXPORT PhotonExperiment {
 public:
  //! True only when this build linked TTTRLib.
  static bool get_available();

  //! Run one scenario in TTTRLib's strict `SimEngine` JSON schema.
  /*! Diffusion, photophysics and state kinetics, FRET routing, polarised
      channels, PIE delays and ALEX are all the scenario's to declare, so bff
      carries no photon simulator of its own. */
  static PhotonExperimentResult simulate(const std::string& scenario_json);

  //! Record the counts an instrument collects for an expected micro-time curve.
  /*! An immobile emitter whose decay is `expected` is observed until
      Poisson(sum(expected)) photons have arrived, one micro-time channel per
      bin. The result is the histogram TTTRLib's encoder produced -- counting
      noise and all -- rather than a noise model applied to the curve. */
  static std::vector<double> record_pattern(const std::vector<double>& expected,
                                            unsigned int seed);
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PHOTONEXPERIMENT_H

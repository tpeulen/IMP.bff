/**
 * \file PhotonExperiment.cpp
 * \brief Photon experiments simulated by TTTRLib, reduced in native memory.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PhotonExperiment.h>
#include <IMP/bff/internal/json.h>

#if IMP_BFF_HAS_TTTRLIB
#include <tttrlib/SimEngine.h>
#endif

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>

IMPBFF_BEGIN_NAMESPACE

PhotonExperimentResult::PhotonExperimentResult()
    : n_channels_(0), n_microtime_bins_(0), n_photons_(0), duration_(0.0) {}

std::vector<double> PhotonExperimentResult::get_channel_histogram(int channel) const {
  if (channel < 0 || channel >= n_channels_) {
    IMP_THROW("no detection channel " << channel, IMP::ValueException);
  }
  const std::size_t begin = static_cast<std::size_t>(channel) * n_microtime_bins_;
  return std::vector<double>(histogram_.begin() + begin,
                             histogram_.begin() + begin + n_microtime_bins_);
}

std::vector<double> PhotonExperimentResult::get_arrival_times(int channel) const {
  std::vector<double> times;
  for (std::size_t i = 0; i < channel_.size(); ++i) {
    if (channel_[i] == channel) times.push_back(time_[i]);
  }
  return times;
}

bool PhotonExperiment::get_available() {
#if IMP_BFF_HAS_TTTRLIB
  return true;
#else
  return false;
#endif
}

PhotonExperimentResult PhotonExperiment::simulate(const std::string& scenario_json) {
#if !IMP_BFF_HAS_TTTRLIB
  (void)scenario_json;
  IMP_THROW("PhotonExperiment needs a bff build linked with TTTRLib",
            IMP::ValueException);
#else
  std::unique_ptr<tttrlib::SimEngine> engine(tttrlib::SimEngine::from_json(scenario_json));
  if (!engine) IMP_THROW("TTTRLib refused the photon scenario", IMP::ValueException);
  engine->run();
  const tttrlib::SimIntegrator& settings = engine->settings();
  if (settings.n_channels <= 0 || settings.n_microtime_channels <= 0) {
    IMP_THROW("the photon scenario declares no channels or no micro-time bins",
              IMP::ValueException);
  }
  PhotonExperimentResult result;
  result.n_channels_ = settings.n_channels;
  result.n_microtime_bins_ = settings.n_microtime_channels;
  result.histogram_.assign(
      static_cast<std::size_t>(result.n_channels_) * result.n_microtime_bins_, 0.0);
  const std::vector<int16_t>& channels = engine->channel();
  const std::vector<uint16_t>& microtimes = engine->micro_time();
  const std::vector<int8_t>& types = engine->event_type();
  const std::vector<uint32_t>& windows = engine->macro_window();
  const std::vector<double>& arrivals = engine->arrival_time();
  if (channels.size() != microtimes.size() || channels.size() != types.size() ||
      channels.size() != windows.size() || channels.size() != arrivals.size()) {
    IMP_THROW("TTTRLib returned inconsistent photon arrays", IMP::ValueException);
  }
  for (std::size_t i = 0; i < channels.size(); ++i) {
    if (types[i] != 0) continue;
    if (channels[i] < 0 || channels[i] >= result.n_channels_ ||
        microtimes[i] >= result.n_microtime_bins_) {
      IMP_THROW("a TTTRLib photon lies outside the declared acquisition",
                IMP::ValueException);
    }
    result.histogram_[static_cast<std::size_t>(channels[i]) * result.n_microtime_bins_ +
                      microtimes[i]] += 1.0;
    result.channel_.push_back(channels[i]);
    result.microtime_.push_back(microtimes[i]);
    result.time_.push_back(windows[i] * settings.dt + arrivals[i]);
    ++result.n_photons_;
  }
  result.duration_ = static_cast<double>(engine->current_window()) * settings.dt;
  return result;
#endif
}

std::vector<double> PhotonExperiment::record_pattern(const std::vector<double>& expected,
                                                     unsigned int seed) {
  std::vector<double> pattern(expected.size(), 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    pattern[i] = std::isfinite(expected[i]) ? std::max(0.0, expected[i]) : 0.0;
    total += pattern[i];
  }
  std::mt19937 rng(seed);
  std::poisson_distribution<long long> draw(total);
  const long long photons = total > 0.0 ? draw(rng) : 0;
  if (photons == 0) return std::vector<double>(expected.size(), 0.0);
  const int bins = static_cast<int>(expected.size());
  nlohmann::json scenario;
  // A thousand photons per window: short enough to stop within a window of
  // the budget, long enough that a large curve takes few steps.
  scenario["settings"] = {{"dt", 1e-3},
                          {"n_ph_max", photons},
                          {"max_windows", 0},
                          {"seed_diffusion", static_cast<unsigned int>(rng())},
                          {"seed_emission", static_cast<unsigned int>(rng())},
                          {"n_channels", 1},
                          {"n_microtime_channels", bins},
                          {"microtime_resolution", 1.0},
                          {"laser_period", static_cast<double>(bins)}};
  scenario["species"] = nlohmann::json::array(
      {{{"D", 0.0}, {"q", {1e6}}, {"decay", {{"pattern", pattern}, {"dt", 1.0}}}}});
  scenario["k_rad"] = {0.0};
  scenario["k_nrad"] = {0.0};
  scenario["background"] = {0.0};
  scenario["population"] = {0.0};
  scenario["emitters"] = nlohmann::json::array(
      {{{"x", 0.0}, {"y", 0.0}, {"z", 0.0}, {"species", 0}, {"mobile", false}}});
  scenario["excitation"] = {{"type", "uniform"}, {"value", 1.0}};
  const PhotonExperimentResult result = simulate(scenario.dump());
  // The engine stops at the end of the window that reached the budget; the
  // photons after the budget are not part of this measurement.
  std::vector<double> histogram(expected.size(), 0.0);
  const std::size_t keep =
      std::min(result.microtime_.size(), static_cast<std::size_t>(photons));
  for (std::size_t i = 0; i < keep; ++i) {
    histogram[static_cast<std::size_t>(result.microtime_[i])] += 1.0;
  }
  return histogram;
}

IMPBFF_END_NAMESPACE

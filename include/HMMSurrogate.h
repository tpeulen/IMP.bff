/**
 *  \file IMP/bff/HMMSurrogate.h
 *  \brief A neural network that estimates H2MM parameters from burst features.
 *
 * Photon-by-photon hidden Markov modelling (H2MM) recovers a model by
 * iterating Baum-Welch EM on a dataset. HmmSurrogate takes the other route:
 * a small dense network, trained once on bursts simulated from the HMM
 * generative model, maps a fixed-length summary of a dataset (the burst
 * features) to the model parameters `(prior, trans, obs)` in one forward pass
 * -- amortised, simulation-based inference.
 *
 * It predicts parameters; it does not seed or steer EM. tttrlib's EM starts
 * from `HMM::factory_model` with restarts and has no hook for a surrogate. A
 * caller who wants an EM polish passes the predicted model to
 * `tttrlib.HMM.optimize` or scores it with `tttrlib.HMM.evaluate` itself.
 *
 * The estimate is approximate, and a trained surrogate is specific to its
 * `(n_states, n_streams)` pair and to the burst-length and inter-photon-gap
 * regime it was trained on.
 *
 * Built only when IMP.bff links tttrlib (`IMP_BFF_HAS_TTTRLIB`): the photon
 * data and the model are tttrlib's `HMM` and `HmmModel`, and the training
 * simulation propagates tttrlib's transition matrices. The network is
 * IMP::bff::NeuralNet, trained by train_neural_net(). Moved here from tttrlib
 * (2026-09): learned models live in imp.bff even when their inputs are
 * photons, and tttrlib stays ML-free.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_HMMSURROGATE_H
#define IMPBFF_HMMSURROGATE_H

#include <IMP/bff/bff_config.h>

#if IMP_BFF_HAS_TTTRLIB

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/NeuralNetTraining.h>

#include <string>
#include <vector>

// Declared, not included: tttrlib's HMM.h pulls its own nlohmann/json
// forward header, which must not meet bff's vendored json in a translation
// unit that did not ask for it (the SWIG wrapper, for one). Only
// HMMSurrogate.cpp includes the real headers.
namespace tttrlib {
class HMM;
struct HmmModel;
}  // namespace tttrlib

IMPBFF_BEGIN_NAMESPACE

//! An HMM model as plain arrays: what HmmSurrogate predicts or decodes.
/*! `trans` is row-major `n_states x n_states`, `obs` row-major
    `n_states x n_streams`. In Python, HmmSurrogate.predict_model turns one
    into a `tttrlib.HmmModel`. */
class IMPBFFEXPORT HmmSurrogateEstimate {
 public:
  HmmSurrogateEstimate() {}
  HmmSurrogateEstimate(const std::vector<double>& prior, const std::vector<double>& trans,
                       const std::vector<double>& obs, long long n_photons = 0)
      : prior_(prior), trans_(trans), obs_(obs), n_photons_(n_photons) {}
  const std::vector<double>& get_prior() const { return prior_; }
  const std::vector<double>& get_trans() const { return trans_; }
  const std::vector<double>& get_obs() const { return obs_; }
  int get_n_states() const { return static_cast<int>(prior_.size()); }
  int get_n_streams() const {
    return prior_.empty() ? 0 : static_cast<int>(obs_.size() / prior_.size());
  }
  //! Photons in the dataset the estimate was made from; 0 for a decoded vector.
  long long get_n_photons() const { return n_photons_; }

  IMP_SHOWABLE_INLINE(HmmSurrogateEstimate,
                      out << "HmmSurrogateEstimate(n_states " << get_n_states()
                          << ", n_streams " << get_n_streams() << ")");

 private:
  std::vector<double> prior_, trans_, obs_;
  long long n_photons_ = 0;
};
IMP_VALUES(HmmSurrogateEstimate, HmmSurrogateEstimates);

//! Amortised neural estimator of H2MM parameters from burst features.
/*! A thin adapter: it owns the feature extractor and the output decoder;
    the network is the domain-agnostic IMP::bff::NeuralNet. Stored as a
    `bff.hmm_surrogate` JSON document that carries the `bff.neural_net`
    document of its network.

    The tttrlib-typed members (extract_features, predict, encode, decode
    taking an `HMM`/`HmmModel`) are C++ only: a `tttrlib.HMM` from tttrlib's
    own Python module cannot cross into IMP.bff's. Python uses the
    array entry points (`*_from_bursts`, `*_from_layout`, `*_arrays`) and the
    helpers `predict_model` / `predict_hmm`, which build the `tttrlib`
    objects on the Python side.
 */
class IMPBFFEXPORT HmmSurrogate {
 public:
  //! Feature-layout version; bumped when the features change so a stale
  //! model is refused rather than silently mis-fed.
  static const int FEATURES_VERSION = 1;
  //! Length of the feature vector: 1 mean + 10 histogram bins + 5 quantiles
  //! + 6 autocorrelation lags + 2 gap statistics.
  static const int N_FEATURES = 24;

  //! Wrap a `bff.neural_net` document as a surrogate.
  /*! \throws IMP::ValueException if the network's width does not match
      N_FEATURES inputs and n_targets(n_states, n_streams) outputs. */
  HmmSurrogate(const std::string& net_json, int n_states, int n_streams,
               int features_version = FEATURES_VERSION);

  // --- serialisation ------------------------------------------------------
  //! Parse a `bff.hmm_surrogate` JSON document.
  static HmmSurrogate from_json_string(const std::string& json);
  //! Read a `bff.hmm_surrogate` JSON file.
  static HmmSurrogate from_json_file(const std::string& path);
  //! Serialise to JSON; `indent < 0` is the compact form.
  std::string to_json_string(int indent = -1) const;
  //! Write the JSON document to `path`.
  void to_json_file(const std::string& path, int indent = 2) const;

  // --- features -----------------------------------------------------------
  //! Fixed-length, burst-order-invariant summary of a dataset.
  /*! The emission structure (windowed local-FRET histogram and quantiles),
      the kinetics (photon-lag autocorrelation of the per-photon FRET signal)
      and the inter-photon timing. Reproduces the NumPy reference bit for bit:
      the histogram is density-normalised over in-range samples, quantiles
      interpolate linearly on the sorted values, the gap spread is a
      population standard deviation. */
  static std::vector<double> extract_features(const tttrlib::HMM& data);
  //! extract_features() on the bursts `tttrlib::HMM::set_bursts` takes.
  static std::vector<double> extract_features_from_bursts(
      const std::vector<std::vector<long long> >& times,
      const std::vector<std::vector<int> >& streams, int n_streams);
  //! extract_features() on the CSR layout an HMM holds (`get_streams`,
  //! `get_offsets`, `get_gap_slot`, `get_unique_dt`).
  static std::vector<double> extract_features_from_layout(
      const std::vector<int>& streams, const std::vector<long long>& offsets,
      const std::vector<int>& gap_slot, const std::vector<long long>& unique_dt,
      int n_streams);

  // --- estimation ---------------------------------------------------------
  //! Estimate a model from `data` in one forward pass.
  /*! \throws IMP::ValueException if `data` has another stream count. */
  tttrlib::HmmModel predict(const tttrlib::HMM& data) const;
  //! predict() on the bursts `tttrlib::HMM::set_bursts` takes; the stream
  //! count is the surrogate's.
  HmmSurrogateEstimate predict_from_bursts(
      const std::vector<std::vector<long long> >& times,
      const std::vector<std::vector<int> >& streams) const;
  //! predict() on an HMM's CSR layout (see extract_features_from_layout).
  HmmSurrogateEstimate predict_from_layout(
      const std::vector<int>& streams, const std::vector<long long>& offsets,
      const std::vector<int>& gap_slot, const std::vector<long long>& unique_dt,
      int n_streams) const;
  //! Decode a feature vector's network output: predict() minus the features.
  HmmSurrogateEstimate predict_from_features(const std::vector<double>& features) const;

  // --- training -----------------------------------------------------------
  //! Simulate labelled datasets and fit the estimator.
  /*! Draws `n_samples` random models over a realistic FRET/kinetics range,
      simulates `n_bursts` bursts of `burst_len` photons (Poisson gaps of mean
      `mean_dt` ticks) from each, and regresses the encoded parameters on the
      features. `seed` seeds the simulation and, overriding `options.seed`,
      the network. */
  static HmmSurrogate train(int n_states, int n_streams, int n_samples = 2500,
                            int n_bursts = 150, int burst_len = 80, double mean_dt = 4.0,
                            const NeuralNetTrainOptions& options = NeuralNetTrainOptions(),
                            int seed = 0);

  //! The labelled training set train() would fit, without fitting.
  /*! Fills `X` (`n_samples x N_FEATURES`) and `Y` (`n_samples x
      n_targets`), both row-major. */
  static void generate_training_set(int n_states, int n_streams, int n_samples,
                                    int n_bursts, int burst_len, double mean_dt,
                                    int seed, std::vector<double>& X,
                                    std::vector<double>& Y);

  //! Length of the regression target: `n*p + n*(n-1) + n`.
  static int n_targets(int n_states, int n_streams);

  //! Flatten a model to the regression target (canonical state order:
  //! descending stream-0 emission; log10 off-diagonal transitions).
  static std::vector<double> encode(const tttrlib::HmmModel& model);
  //! encode() on a model given as arrays.
  static std::vector<double> encode_arrays(const std::vector<double>& prior,
                                           const std::vector<double>& trans,
                                           const std::vector<double>& obs);
  //! Rebuild a valid (row-stochastic, canonically ordered) model from a
  //! possibly noisy target vector.
  static tttrlib::HmmModel decode(const std::vector<double>& vec, int n_states,
                                  int n_streams);
  //! decode() returning arrays.
  static HmmSurrogateEstimate decode_arrays(const std::vector<double>& vec,
                                            int n_states, int n_streams);

  // --- introspection ------------------------------------------------------
  const NeuralNet& get_net() const { return net_; }
  //! The network's `bff.neural_net` document.
  const std::string& get_net_json() const { return net_json_; }
  int get_n_states() const { return n_states_; }
  int get_n_streams() const { return n_streams_; }
  int get_features_version() const { return features_version_; }
  //! Per-epoch training loss of the train() that made this surrogate;
  //! empty for one loaded from JSON.
  const std::vector<double>& get_loss_curve() const { return loss_curve_; }
  //! Per-epoch held-out loss, likewise; empty without early stopping.
  const std::vector<double>& get_validation_curve() const { return validation_curve_; }

  IMP_SHOWABLE_INLINE(HmmSurrogate,
                      out << "HmmSurrogate(n_states=" << n_states_
                          << ", n_streams=" << n_streams_
                          << ", features_version=" << features_version_ << ")");

 private:
  std::string net_json_;
  NeuralNet net_;
  int n_states_;
  int n_streams_;
  int features_version_;
  std::vector<double> loss_curve_;
  std::vector<double> validation_curve_;
};

IMPBFF_END_NAMESPACE

#endif  // IMP_BFF_HAS_TTTRLIB

#endif  // IMPBFF_HMMSURROGATE_H

/**
 * \file IMP/bff/Convolution.h
 * \brief A sampled curve convolved with a measured response, as a node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_CONVOLUTION_H
#define IMPBFF_CONVOLUTION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FitDataset.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/ResponseFunction.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Convolve whatever curve arrives with a measured response.
/*!
    The node knows nothing about the curve: an equation of time, a
    simulated trace, any sampled model. It reads the vector input port
    `curve`, shifts the bound response by the scalar input `timeshift`
    (channels; zero when the port is absent), normalises it to unit sum --
    the same preparation every response-convolving node here uses -- and
    writes the convolution, as long as the curve, to the output keyed by its
    own name. How the convolution is cut to that length is the mode:

    - `causal` (default): `out[i] = sum_{j<=i} curve[j] * r[i-j]` -- a
      response that starts at the curve's start, as a single excitation sees
      it; what the response carries past the last sample is dropped.
    - `periodic`: the same, plus what spilled past the end of every earlier
      excitation, repeated every #set_period samples (fractional periods are
      interpolated) -- a repetitively excited measurement. The curve's own
      periodicity is the curve's to carry.
    - `centered`: the response centred on each sample,
      `np.convolve(curve, r, "same")` -- a symmetric kernel smoothing a
      spectrum or an image line, which has no time direction.

    Kernels: tttrlib's `convolve_causal_ad`, `convolve_full_ad` and
    `fold_periodic_ad`.

    What happens to the convolved curve next is another node's business: an
    instrument that scales it, adds background and corrects pile-up, a
    misfit that compares it. Kept apart so the convolution is written once.

    Settings (#configure): `mode`, `period` (samples), `normalize_response`
    (default true). Datasets (#bind_dataset): `response`.
*/
class IMPBFFEXPORT Convolution : public GraphNode {
 public:
  explicit Convolution(const std::string& name = "convolution");

  //! The response the curve is convolved with, as measured.
  void set_response(const std::vector<double>& response);
  void set_response_array(double* in_response, int n_response);
  const std::vector<double>& get_response() const { return response_; }

  //! Whether the response is scaled to unit sum before convolving (default true).
  void set_normalize_response(bool v);
  bool get_normalize_response() const { return normalize_response_; }

  //! How the convolution is cut to the curve's length: causal, periodic, centered.
  /*! \throws std::domain_error for any other name. */
  void set_mode(const std::string& mode);
  const std::string& get_mode() const { return mode_; }

  //! The repetition period in samples, used by the periodic mode.
  /*! \throws std::domain_error unless positive and finite. */
  void set_period(double samples);
  double get_period() const { return period_; }

  //! Clean the response before use: zero it outside `[start, stop)` samples.
  /*! As TCSPCDecay::set_response_range; the two share one preparation. */
  void set_response_range(int start, int stop);

  //! The key of the optional scalar input carrying the response's background.
  static const char* response_background_port_key() { return "response_background"; }

  //! The key of the input port carrying the curve.
  static const char* curve_port_key() { return "curve"; }
  //! The key of the optional scalar input port carrying the timeshift.
  static const char* timeshift_port_key() { return "timeshift"; }

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;
  void bind_dataset(const std::string& role, const FitDataset& dataset) override;

 private:
  std::vector<double> response_;
  std::vector<double> shifted_;
  std::vector<double> prepared_;
  std::vector<double> out_;
  bool normalize_response_ = true;
  std::string mode_ = "causal";
  double period_ = 0.0;
  std::vector<double> full_;
  std::vector<double> cleaned_;
  internal::ResponsePreparation preparation_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_CONVOLUTION_H

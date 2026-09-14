/**
 * \file Convolution.cpp
 * \brief A sampled curve convolved with a measured response.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/Convolution.h>

#include <IMP/bff/internal/DecayConvolution.h>
#include <IMP/bff/internal/NodeConfig.h>
#include <IMP/bff/internal/ResponseFunction.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// No ports here: a node has to be owned by a shared_ptr before it can own
// ports, and a description creates the inputs and output it wires anyway.
Convolution::Convolution(const std::string& name) : GraphNode(name) {}

void Convolution::set_response(const std::vector<double>& response) {
  if (response.empty()) {
    throw std::domain_error("Convolution::set_response: the response is empty");
  }
  response_ = response;
  set_valid(false);
}

void Convolution::set_response_array(double* in_response, int n_response) {
  set_response(std::vector<double>(in_response, in_response + n_response));
}

void Convolution::set_normalize_response(bool v) {
  normalize_response_ = v;
  set_valid(false);
}

void Convolution::set_response_range(int start, int stop) {
  if (start < 0) {
    throw std::domain_error(
        "Convolution::set_response_range: the window starts before sample 0");
  }
  preparation_.active = true;
  preparation_.start = start;
  preparation_.stop = stop;
  set_valid(false);
}

void Convolution::set_response_from_port(bool v) {
  response_from_port_ = v;
  set_valid(false);
}

void Convolution::set_mode(const std::string& mode) {
  if (mode != "causal" && mode != "periodic" && mode != "centered") {
    throw std::domain_error("Convolution::set_mode: '" + mode +
                            "' is not a mode; the modes are causal, "
                            "periodic and centered");
  }
  mode_ = mode;
  set_valid(false);
}

void Convolution::set_period(double samples) {
  if (!(samples > 0.0) || !std::isfinite(samples)) {
    throw std::domain_error(
        "Convolution::set_period: a repetition period is a positive number "
        "of samples");
  }
  period_ = samples;
  set_valid(false);
}

void Convolution::evaluate() {
  if (response_from_port_) {
    const std::shared_ptr<GraphPort> port = get_input_port("response");
    if (!port) {
      throw std::domain_error("Convolution '" + get_name() +
                              "' reads its response from a port it does not have");
    }
    response_ = port->get_values_ref();
  }
  if (response_.empty()) {
    throw std::domain_error("Convolution '" + get_name() +
                            "' has no response to convolve with");
  }
  const std::shared_ptr<GraphPort> curve_port =
      get_input_port(curve_port_key());
  if (!curve_port) {
    throw std::domain_error("Convolution '" + get_name() +
                            "' has no input '" + curve_port_key() + "'");
  }
  const std::vector<double>& curve = curve_port->get_values_ref();
  if (curve.size() != response_.size()) {
    std::ostringstream m;
    m << "Convolution '" << get_name() << "' convolves a curve of "
      << curve.size() << " values with a response of " << response_.size();
    throw std::domain_error(m.str());
  }
  const std::shared_ptr<GraphPort> shift = get_input_port(timeshift_port_key());
  const double timeshift = shift ? shift->get_value() : 0.0;
  internal::ResponsePreparation preparation = preparation_;
  if (const std::shared_ptr<GraphPort> bg =
          get_input_port(response_background_port_key())) {
    preparation.active = true;
    preparation.background = bg->get_value();
  }
  if (normalize_response_) {
    internal::prepare_response(response_, preparation, timeshift, cleaned_,
                               shifted_, prepared_);
  } else {
    // The same cleaning and shift, without the unit sum: a kernel whose
    // area means something (a measured instrument broadening) keeps it.
    const double* source =
        internal::clean_response(response_, preparation, cleaned_);
    prepared_.resize(response_.size());
    if (timeshift != 0.0) {
      shift_lamp_ad<double>(prepared_.data(), source, -timeshift,
                            static_cast<int>(response_.size()), 0.0);
    } else {
      std::copy(source, source + response_.size(), prepared_.begin());
    }
  }
  const int n = static_cast<int>(curve.size());
  out_.resize(curve.size());
  if (mode_ == "causal") {
    convolve_causal_ad<double>(out_.data(), curve.data(), prepared_.data(), n);
  } else {
    full_.resize(static_cast<std::size_t>(2 * n - 1));
    convolve_full_ad<double>(full_.data(), curve.data(), n, prepared_.data(), n);
    if (mode_ == "periodic") {
      if (!(period_ > 0.0)) {
        throw std::domain_error("Convolution '" + get_name() +
                                "' is periodic but has no period");
      }
      fold_periodic_ad<double>(out_.data(), n, full_.data(), 2 * n - 1,
                               period_);
    } else {
      // numpy's "same": the middle n of the full convolution.
      const int offset = (n - 1) / 2;
      std::copy(full_.begin() + offset, full_.begin() + offset + n,
                out_.begin());
    }
  }

  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error("Convolution '" + get_name() +
                            "' writes to the output keyed by its own name, "
                            "which this node does not have");
  }
  // Fit transport: a NaN must reach the misfit rather than be floored.
  out->set_sanitize(false);
  out->set_value_vector(out_);
  set_valid(true);
}

std::string Convolution::get_node_type() const { return "Convolution"; }

void Convolution::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("response_range")) {
    const std::vector<int> range = config.get_ints("response_range");
    if (range.size() != 2) {
      throw std::domain_error(
          "node type 'Convolution': setting 'response_range' must be [start, stop]");
    }
    set_response_range(range[0], range[1]);
  }
  if (config.has("response_from_port")) {
    set_response_from_port(config.get_bool("response_from_port"));
  }
  if (config.has("mode")) set_mode(config.get_string("mode"));
  if (config.has("period")) set_period(config.get_double("period"));
  if (config.has("normalize_response")) {
    set_normalize_response(config.get_bool("normalize_response"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

void Convolution::bind_dataset(const std::string& role,
                               const FitDataset& dataset) {
  if (role == "response") {
    set_response(dataset.get_values());
    return;
  }
  throw std::domain_error("node type 'Convolution' has no role '" + role +
                          "'; it takes 'response'");
}

IMPBFF_END_NAMESPACE

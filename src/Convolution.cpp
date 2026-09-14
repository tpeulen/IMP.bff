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

void Convolution::evaluate() {
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
  if (normalize_response_) {
    internal::prepare_response(response_, timeshift, shifted_, prepared_);
  } else if (timeshift != 0.0) {
    prepared_.resize(response_.size());
    shift_lamp_ad<double>(prepared_.data(), response_.data(), -timeshift,
                          static_cast<int>(response_.size()), 0.0);
  } else {
    prepared_ = response_;
  }
  out_.resize(curve.size());
  convolve_causal_ad<double>(out_.data(), curve.data(), prepared_.data(),
                             static_cast<int>(curve.size()));

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

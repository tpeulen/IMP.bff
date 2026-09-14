/**
 * \file PhotophysicsTransferKineticsNode.cpp
 * \brief Transfer kinetics between two chromophores, as a graph node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PhotophysicsTransferKineticsNode.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/PhotophysicsCrosstalkMatrix.h>
#include <IMP/bff/PhotophysicsTransferKinetics.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <algorithm>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {
const char* kChromophores[2] = {"A", "B"};
}

PhotophysicsTransferKineticsNode::PhotophysicsTransferKineticsNode(const std::string& name)
    : GraphNode(name) {}

void PhotophysicsTransferKineticsNode::build_ports() {
  const auto scalar = [this](const std::string& key, double value) {
    if (!get_input_port(key)) add_input_port(key, std::make_shared<GraphPort>(value));
  };
  const auto vector = [this](const std::string& key) {
    if (!get_input_port(key)) {
      add_input_port(key, std::make_shared<GraphPort>(std::vector<double>{1.0, 4.0}));
    }
  };
  vector("spectrum_a");
  vector("spectrum_b");
  vector("rates");
  scalar("f_ab", 1.0);
  scalar("f_ba", 0.0);
  scalar("pure_a", 0.0);
  scalar("pure_b", 0.0);
  for (const std::string& pulse : pulses_) {
    for (const char* chromophore : kChromophores) {
      scalar("excitation_" + pulse + "_" + chromophore, 1.0);
    }
  }
  for (const char* chromophore : kChromophores) {
    for (const std::string& channel : channels_) {
      scalar(std::string("emission_") + chromophore + "_" + channel, 1.0);
    }
  }
  set_valid(false);
}

void PhotophysicsTransferKineticsNode::evaluate() {
  const std::string where = "PhotophysicsTransferKineticsNode '" + get_name() + "'";
  const auto value = [&](const std::string& key) {
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no port '" + key + "'");
    return port->get_value();
  };
  const auto values = [&](const std::string& key) {
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no port '" + key + "'");
    return port->get_values_ref();
  };
  std::vector<double> excitation_values, emission_values;
  for (const std::string& pulse : pulses_) {
    for (const char* chromophore : kChromophores) {
      excitation_values.push_back(value("excitation_" + pulse + "_" + chromophore));
    }
  }
  for (const char* chromophore : kChromophores) {
    for (const std::string& channel : channels_) {
      emission_values.push_back(value(std::string("emission_") + chromophore + "_" + channel));
    }
  }
  const std::vector<std::string> chromophores{"A", "B"};
  const PhotophysicsCrosstalkMatrix excitation(pulses_, chromophores, excitation_values);
  const PhotophysicsCrosstalkMatrix emission(chromophores, channels_, emission_values);
  const std::vector<double> spectrum = transfer_kinetics_spectrum(
      values("spectrum_a"), values("spectrum_b"), values("rates"), value("f_ab"),
      value("f_ba"),
      transfer_populations_from_pure_fractions(value("pure_a"), value("pure_b")), excitation,
      emission, pulse_, channel_, "A", "B", mode_, eps_);
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error(where + " writes to the output keyed by its own name, "
                                    "which this node does not have");
  }
  out->set_value_vector(spectrum);
  set_valid(true);
}

std::string PhotophysicsTransferKineticsNode::get_node_type() const {
  return "PhotophysicsTransferKineticsNode";
}

void PhotophysicsTransferKineticsNode::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("pulses")) pulses_ = config.get_strings("pulses");
  if (config.has("channels")) channels_ = config.get_strings("channels");
  pulse_ = config.has("pulse") ? config.get_string("pulse") : pulses_.front();
  channel_ = config.has("channel") ? config.get_string("channel") : channels_.front();
  if (std::find(pulses_.begin(), pulses_.end(), pulse_) == pulses_.end() ||
      std::find(channels_.begin(), channels_.end(), channel_) == channels_.end()) {
    throw std::domain_error("PhotophysicsTransferKineticsNode '" + get_name() +
                            "': pulse and channel must be among pulses and channels");
  }
  if (config.has("mode")) {
    const std::string mode = config.get_string("mode");
    if (mode == "exact") {
      mode_ = TRANSFER_EXACT;
    } else if (mode == "chisurf") {
      mode_ = TRANSFER_CHISURF_REGULARISED;
    } else {
      throw std::domain_error("PhotophysicsTransferKineticsNode: mode is 'exact' or 'chisurf'");
    }
  }
  if (config.has("eps")) eps_ = config.get_double("eps");
  build_ports();
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

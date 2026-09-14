/**
 * \file DiscreteDistances.cpp
 * \brief A distance distribution of a few discrete distances, as a node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/DiscreteDistances.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

DiscreteDistances::DiscreteDistances(const std::string& name) : GraphNode(name) {}

void DiscreteDistances::set_number_of_distances(int n) {
  if (n < 1) {
    throw std::domain_error("DiscreteDistances::set_number_of_distances: at least one");
  }
  // The ports are built here, as GaussianDistances builds its own; a port a
  // caller already added is kept.
  for (int i = 0; i < n; ++i) {
    for (const char* stem : {"distance", "amplitude"}) {
      const std::string key = stem + std::to_string(i);
      if (!get_input_port(key)) {
        add_input_port(key, std::make_shared<GraphPort>(
                                std::string(stem) == "distance" ? 50.0 : 1.0));
      }
    }
  }
  n_distances_ = n;
  set_valid(false);
}

void DiscreteDistances::evaluate() {
  const std::string where = "DiscreteDistances '" + get_name() + "'";
  if (n_distances_ < 1) {
    throw std::domain_error(where + " has no distances; set number_of_distances");
  }
  out_.assign(2 * static_cast<std::size_t>(n_distances_), 0.0);
  double total = 0.0;
  for (int i = 0; i < n_distances_; ++i) {
    const std::string suffix = std::to_string(i);
    const std::shared_ptr<GraphPort> distance = get_input_port("distance" + suffix);
    const std::shared_ptr<GraphPort> amplitude = get_input_port("amplitude" + suffix);
    if (!distance || !amplitude) {
      throw std::domain_error(where + " needs 'distance" + suffix +
                              "' and 'amplitude" + suffix + "'");
    }
    const std::size_t k = 2 * static_cast<std::size_t>(i);
    out_[k] = std::fabs(amplitude->get_value());
    out_[k + 1] = std::fabs(distance->get_value());
    total += out_[k];
  }
  if (!(total > 0.0)) throw std::domain_error(where + ": every amplitude is zero");
  for (std::size_t k = 0; k < out_.size(); k += 2) out_[k] /= total;
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error(where + " writes to the output keyed by its own "
                                    "name, which this node does not have");
  }
  out->set_value_vector(out_);
  set_valid(true);
}

std::string DiscreteDistances::get_node_type() const { return "DiscreteDistances"; }

void DiscreteDistances::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("number_of_distances")) {
    set_number_of_distances(config.get_int("number_of_distances"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

/**
 * \file AcceptorDensityDecay.cpp
 * \brief The donor decay quenched by an acceptor density, as a graph node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/AcceptorDensityDecay.h>

#include <IMP/bff/FRETAcceptorDensity.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

AcceptorDensityDecay::AcceptorDensityDecay(const std::string& name) : GraphNode(name) {}

void AcceptorDensityDecay::evaluate() {
  const std::string where = "AcceptorDensityDecay '" + get_name() + "'";
  if (n_points_ <= 0) throw std::domain_error(where + " has no sampling; set n_points and dt");
  const std::shared_ptr<GraphPort> spectrum = get_input_port("lifetime_spectrum");
  const std::shared_ptr<GraphPort> density = get_input_port("c_over_c0");
  const std::shared_ptr<GraphPort> tau0 = get_input_port("tau0");
  if (!spectrum || !density || !tau0) {
    throw std::domain_error(where + " needs lifetime_spectrum, c_over_c0 and tau0");
  }
  const std::vector<double> decay = acceptor_density_decay(
      spectrum->get_values_ref(), n_points_, dt_, tau0->get_value(),
      std::fabs(density->get_value()), dimension_, period_, folded_periods_);
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) throw std::domain_error(where + " writes to the output keyed by its own name");
  out->set_value_vector(decay);
  // What the density means, published where a port asks for it: the transfer
  // efficiency it implies, and the absolute density for the node's R0.
  if (const std::shared_ptr<GraphPort> e = get_output_port("efficiency")) {
    e->set_value(acceptor_transfer_efficiency(std::fabs(density->get_value()), dimension_));
  }
  if (const std::shared_ptr<GraphPort> rho = get_output_port("acceptor_density")) {
    const std::shared_ptr<GraphPort> r0 = get_input_port("forster_radius");
    if (!r0) throw std::domain_error(where + " publishes acceptor_density and needs forster_radius");
    rho->set_value(std::fabs(density->get_value()) *
                   acceptor_characteristic_density(r0->get_value(), dimension_));
  }
  set_valid(true);
}

std::string AcceptorDensityDecay::get_node_type() const { return "AcceptorDensityDecay"; }

void AcceptorDensityDecay::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("dimension")) dimension_ = config.get_int("dimension");
  if (config.has("n_points")) n_points_ = config.get_int("n_points");
  if (config.has("dt")) dt_ = config.get_double("dt");
  if (config.has("period")) period_ = config.get_double("period");
  if (config.has("folded_periods")) folded_periods_ = config.get_int("folded_periods");
  if (dimension_ < 1 || dimension_ > 3) {
    throw std::domain_error("AcceptorDensityDecay: dimension must be 1, 2 or 3");
  }
  set_valid(false);
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

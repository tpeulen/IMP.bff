/**
 * \file GeneralizedNormalCurve.cpp
 * \brief A sampled generalized-normal peak, as a node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/GeneralizedNormalCurve.h>

#include <IMP/bff/Distributions.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

GeneralizedNormalCurve::GeneralizedNormalCurve(const std::string& name)
    : GraphNode(name) {}

void GeneralizedNormalCurve::set_sampling(int n_points, double dt) {
  if (n_points <= 0 || !(dt > 0.0)) {
    throw std::domain_error(
        "GeneralizedNormalCurve::set_sampling: a positive number of points and "
        "a positive step");
  }
  n_points_ = n_points;
  dt_ = dt;
  set_valid(false);
}

void GeneralizedNormalCurve::set_area(double area) { area_ = area; set_valid(false); }
void GeneralizedNormalCurve::set_floor(double floor) { floor_ = floor; set_valid(false); }
void GeneralizedNormalCurve::set_normalize(bool v) { normalize_ = v; set_valid(false); }

double GeneralizedNormalCurve::scalar_input(const char* key, double fallback) const {
  const std::shared_ptr<GraphPort> port = get_input_port(key);
  return port ? port->get_value() : fallback;
}

void GeneralizedNormalCurve::evaluate() {
  if (n_points_ <= 0) {
    throw std::domain_error("GeneralizedNormalCurve '" + get_name() +
                            "' has no sampling; set n_points and dt");
  }
  std::vector<double> x(static_cast<std::size_t>(n_points_));
  for (int i = 0; i < n_points_; ++i) x[static_cast<std::size_t>(i)] = i * dt_;
  const double scale = std::max(std::fabs(scalar_input("scale", 1.0)),
                                std::numeric_limits<double>::epsilon());
  out_ = generalized_normal_density_impl(x, scalar_input("loc", 0.0), scale,
                                         scalar_input("shape", 0.0), normalize_);
  for (double& v : out_) {
    v *= area_;
    if (v < floor_) v = 0.0;
  }
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error("GeneralizedNormalCurve '" + get_name() +
                            "' writes to the output keyed by its own name, "
                            "which this node does not have");
  }
  out->set_value_vector(out_);
  set_valid(true);
}

std::string GeneralizedNormalCurve::get_node_type() const {
  return "GeneralizedNormalCurve";
}

void GeneralizedNormalCurve::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("n_points") || config.has("dt")) {
    set_sampling(config.has("n_points") ? config.get_int("n_points") : n_points_,
                 config.has("dt") ? config.get_double("dt") : dt_);
  }
  if (config.has("area")) set_area(config.get_double("area"));
  if (config.has("floor")) set_floor(config.get_double("floor"));
  if (config.has("normalize")) set_normalize(config.get_bool("normalize"));
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

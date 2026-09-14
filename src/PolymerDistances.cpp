/**\file PolymerDistances.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PolymerDistances.h>
#include <IMP/bff/internal/DistanceAxis.h>
#include <IMP/bff/internal/NodeConfig.h>
#include <IMP/bff/internal/DistanceKernels.h>
#include "internal/SpectrumNodeHelpers.h"
#include <IMP/bff/PolymerChain.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// ------------------------------------------------------- PolymerDistances

PolymerDistances::PolymerDistances(const std::string& name) : GraphNode(name) {}

void PolymerDistances::set_mode(const std::string& mode) {
  if (!mode_.empty()) {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "': the mode is set once");
  }
  parameter_ports_.clear();
  GraphPort* p = nullptr;
  if (mode == "worm_like_chain" || mode == "worm_like_chain_linker") {
    spectrum_node_detail::add_scalar_port(this, "chain_length", 100.0, &p);
    parameter_ports_.push_back(p);
    spectrum_node_detail::add_scalar_port(this, "persistence_length", 30.0, &p);
    parameter_ports_.push_back(p);
    // Both modes carry the linker width. Without the linker it is inert --
    // evaluate() never reads it -- which mirrors chisurf exactly: the
    // model's `w` is a fitting parameter whether or not the linker is on,
    // and with it off the numpy path fits an inert parameter too. A port
    // the optimiser can claim is what keeps a free-but-inert `w` from
    // refusing the whole graph.
    spectrum_node_detail::add_scalar_port(this, "sigma_linker", 6.0, &p);
    parameter_ports_.push_back(p);
  } else if (mode == "saw_nu") {
    spectrum_node_detail::add_scalar_port(this, "r_rms", 50.0, &p);
    parameter_ports_.push_back(p);
    spectrum_node_detail::add_scalar_port(this, "nu", 0.588, &p);
    parameter_ports_.push_back(p);
  } else if (mode == "ising_chain") {
    const char* keys[] = {"n_residues", "b_structured", "b_unstructured",
                          "coupling", "field"};
    const double defaults[] = {50.0, 4.0, 8.0, 1.5, 0.0};
    for (int i = 0; i < 5; ++i) {
      spectrum_node_detail::add_scalar_port(this, keys[i], defaults[i], &p);
      parameter_ports_.push_back(p);
    }
  } else {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "': unknown mode '" + mode + "'");
  }
  mode_ = mode;
  set_valid(false);
}

void PolymerDistances::set_axis(const std::vector<double>& axis) {
  if (axis.size() < 2) {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "': a distance axis needs at least two points");
  }
  axis_ = axis;
  set_valid(false);
}

void PolymerDistances::set_axis_array(double* in_axis, int n_axis) {
  set_axis(std::vector<double>(in_axis, in_axis + n_axis));
}

std::vector<double> PolymerDistances::weights_at(
    const std::vector<double>& parameters) const {
  if (mode_.empty()) {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "' has no mode; call set_mode() first");
  }
  if (axis_.empty()) {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "' has no distance axis");
  }
  // The kernels publish malloc'd views (their numpy contract); copy and
  // free. They are the same functions chisurf's rdf.py forwarders call, so
  // the graph path and the Python path evaluate one implementation.
  double* view = nullptr;
  int n_view = 0;
  if (mode_ == "worm_like_chain" || mode_ == "worm_like_chain_linker") {
    const double chain_length = parameters[0];
    const double persistence_length = parameters[1];
    if (!(chain_length > 0.0)) {
      throw std::domain_error("PolymerDistances '" + get_name() +
                              "': chain_length is not positive");
    }
    // The kernel takes the dimensionless kappa; the ports carry what the
    // model fits. Same derivation as chisurf's model property.
    const double kappa = persistence_length / chain_length;
    if (mode_ == "worm_like_chain") {
      // `distance = false`, always: chisurf's forwarder has never applied
      // the r^2 factor its signature advertises -- the flag was accepted
      // and dropped on the floor, every fit in the stack was made against
      // that behaviour, and preserving the answer is the port's contract
      // (rdf.py says the same over the same call; owner decision pending
      // in PRD-105 on the flag itself).
      worm_like_chain(axis_, kappa, chain_length, true, false, &view, &n_view);
    } else {
      worm_like_chain_linker(axis_, kappa, chain_length, parameters[2], true,
                             &view, &n_view);
    }
  } else if (mode_ == "saw_nu") {
    saw_nu(axis_, parameters[0], parameters[1], 1.1615, &view, &n_view);
  } else {  // ising_chain
    const int n_residues = static_cast<int>(std::lround(parameters[0]));
    ising_chain(axis_, n_residues, parameters[1], parameters[2], parameters[3],
                parameters[4], n_k_, &view, &n_view);
  }
  if (view == nullptr || n_view != static_cast<int>(axis_.size())) {
    std::free(view);
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "': the kernel returned no distribution");
  }
  std::vector<double> weights(view, view + n_view);
  std::free(view);
  if (normalize_weights_) {
    double total = 0.0;
    for (double v : weights) total += v;
    if (total > 0.0) {
      for (double& v : weights) v /= total;
    }
  }
  return weights;
}

std::vector<double> PolymerDistances::current_parameters() const {
  std::vector<double> values;
  for (GraphPort* port : parameter_ports_) values.push_back(port->get_value());
  return values;
}

void PolymerDistances::evaluate() {
  const std::vector<double> weights = weights_at(current_parameters());
  spectrum_.resize(2 * axis_.size());
  for (std::size_t j = 0; j < axis_.size(); ++j) {
    spectrum_[2 * j] = weights[j];
    spectrum_[2 * j + 1] = axis_[j];
  }
  spectrum_node_detail::publish(this, spectrum_);
  set_valid(true);
}

std::vector<std::string> PolymerDistances::get_parameter_names() const {
  if (mode_ == "worm_like_chain" || mode_ == "worm_like_chain_linker") {
    return {"chain_length", "persistence_length", "sigma_linker"};
  }
  if (mode_ == "saw_nu") return {"r_rms", "nu"};
  if (mode_ == "ising_chain") {
    return {"n_residues", "b_structured", "b_unstructured", "coupling", "field"};
  }
  return {};
}

std::vector<double> PolymerDistances::get_weights_jacobian() const {
  if (mode_.empty() || axis_.empty()) {
    throw std::domain_error("PolymerDistances '" + get_name() +
                            "' needs a mode and a distance axis");
  }
  using internal::Dual;
  const std::vector<double> at = current_parameters();
  const std::size_t n = axis_.size();
  const std::size_t n_params = at.size();
  std::vector<Dual> weights;
  // The same kernels evaluate() uses, on dual numbers: the derivative of
  // exactly that arithmetic, branches and normalisations included.
  if (mode_ == "worm_like_chain" || mode_ == "worm_like_chain_linker") {
    const Dual chain_length = Dual::variable(at[0], 0);
    const Dual persistence_length = Dual::variable(at[1], 1);
    if (!(at[0] > 0.0)) {
      throw std::domain_error("PolymerDistances '" + get_name() +
                              "': chain_length is not positive");
    }
    const Dual kappa = persistence_length / chain_length;
    weights = mode_ == "worm_like_chain"
                  ? internal::worm_like_chain_t(axis_, kappa, chain_length, true, false)
                  : internal::worm_like_chain_linker_t(axis_, kappa, chain_length,
                                                       Dual::variable(at[2], 2), true);
  } else if (mode_ == "saw_nu") {
    weights = internal::saw_nu_t(axis_, Dual::variable(at[0], 0), Dual::variable(at[1], 1),
                                 1.1615);
  } else {
    // The residue count is rounded to an integer: the weights are piecewise
    // constant in it, and its column is 0.
    weights = internal::ising_chain_t(axis_, static_cast<int>(std::lround(at[0])),
                                      Dual::variable(at[1], 1), Dual::variable(at[2], 2),
                                      Dual::variable(at[3], 3), Dual::variable(at[4], 4), n_k_);
  }
  if (normalize_weights_) internal::normalize_sum_t(weights);
  std::vector<double> jacobian(n * n_params, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t c = 0; c < n_params; ++c) jacobian[j * n_params + c] = weights[j].d[c];
  }
  return jacobian;
}

std::string PolymerDistances::get_node_type() const { return "PolymerDistances"; }

void PolymerDistances::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  const std::vector<double> axis =
      internal::configured_distance_axis(config, "PolymerDistances '" + get_name() + "'");
  if (!axis.empty()) set_axis(axis);
  if (config.has("mode")) set_mode(config.get_string("mode"));
  if (config.has("n_k")) set_n_k(config.get_int("n_k"));
  if (config.has("normalize_weights")) {
    set_normalize_weights(config.get_bool("normalize_weights"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

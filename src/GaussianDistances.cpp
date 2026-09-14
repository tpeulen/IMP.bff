/**\file GaussianDistances.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/GaussianDistances.h>
#include <IMP/bff/internal/DistanceAxis.h>
#include <IMP/bff/internal/DistanceKernels.h>
#include <IMP/bff/internal/NodeConfig.h>
#include "internal/SpectrumNodeHelpers.h"
#include <IMP/bff/Distributions.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// ------------------------------------------------------- GaussianDistances

GaussianDistances::GaussianDistances(const std::string& name) : GraphNode(name) {}

void GaussianDistances::set_number_of_components(int n) {
  if (n <= 0) {
    throw std::domain_error(
        "GaussianDistances::set_number_of_components: a distribution has at "
        "least one component");
  }
  component_ports_.clear();
  component_ports_.reserve(static_cast<std::size_t>(4 * n));
  static const char* kNames[4] = {"mean", "sigma", "shape", "amplitude"};
  static const double kDefaults[4] = {50.0, 6.0, 0.0, 1.0};
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 4; ++k) {
      std::ostringstream key;
      key << kNames[k] << i;
      GraphPort* port = nullptr;
      if (get_input_port(key.str())) {
        port = get_input_port(key.str()).get();
      } else {
        spectrum_node_detail::add_scalar_port(this, key.str(), kDefaults[k], &port);
      }
      component_ports_.push_back(port);
    }
  }
  n_components_ = n;
  set_valid(false);
}

void GaussianDistances::set_axis(const std::vector<double>& axis) {
  if (axis.size() < 2) {
    throw std::domain_error(
        "GaussianDistances::set_axis: a distance axis needs at least two "
        "points");
  }
  axis_ = axis;
  density_.assign(axis_.size(), 0.0);
  set_valid(false);
}

void GaussianDistances::set_axis_array(double* in_axis, int n_axis) {
  set_axis(std::vector<double>(in_axis, in_axis + n_axis));
}

void GaussianDistances::evaluate() {
  if (axis_.empty()) {
    throw std::domain_error("GaussianDistances '" + get_name() +
                            "' has no distance axis");
  }
  const std::size_t n = axis_.size();
  std::vector<double> means, sigmas, shapes, amplitudes;
  means.reserve(n_components_);
  for (int c = 0; c < n_components_; ++c) {
    const std::size_t base = static_cast<std::size_t>(4 * c);
    // The mean and the amplitude are taken absolute, as ChiSurf's getters do:
    // a distance is positive, and a fit that walks one through zero would
    // otherwise put the distribution on the wrong side of the axis.
    means.push_back(std::fabs(component_ports_[base]->get_value()));
    sigmas.push_back(component_ports_[base + 1]->get_value());
    shapes.push_back(component_ports_[base + 2]->get_value());
    amplitudes.push_back(std::fabs(component_ports_[base + 3]->get_value()));
  }

  // One kernel, shared with the free function ChiSurf's own
  // `Gaussians.distribution` calls -- so the graph path and the Python path
  // cannot drift apart. `normalize_components` follows the branch, because the
  // reference is asymmetric between its two; see `Distributions.h`.
  //
  // ChiSurf normalises the *weights* in `Gaussians.amplitude` and the combined
  // density again in `combine_distributions`; the first is a division by a
  // constant that the second undoes, so normalising once here is the same
  // number.
  density_ = gaussian_distance_mixture_impl(
      axis_, means, sigmas, shapes, amplitudes,
      distance_between_gaussians_ ? GAUSSIAN_MIXTURE_DISTANCE_BETWEEN_GAUSSIANS
                                  : GAUSSIAN_MIXTURE_GENERALIZED_NORMAL,
      !distance_between_gaussians_, true);

  spectrum_.resize(2 * n);
  for (std::size_t j = 0; j < n; ++j) {
    spectrum_[2 * j] = density_[j];
    spectrum_[2 * j + 1] = axis_[j];
  }
  spectrum_node_detail::publish(this, spectrum_);
  set_valid(true);
}

std::string GaussianDistances::get_node_type() const { return "GaussianDistances"; }

std::vector<std::string> GaussianDistances::get_parameter_names() const {
  static const char* kNames[4] = {"mean", "sigma", "shape", "amplitude"};
  std::vector<std::string> names;
  for (int c = 0; c < n_components_; ++c) {
    for (int k = 0; k < 4; ++k) names.push_back(kNames[k] + std::to_string(c));
  }
  return names;
}

std::vector<double> GaussianDistances::get_weights_jacobian() const {
  if (axis_.empty()) {
    throw std::domain_error("GaussianDistances '" + get_name() +
                            "' has no distance axis");
  }
  using internal::Dual;
  const std::size_t n = axis_.size();
  const std::size_t n_params = static_cast<std::size_t>(4 * n_components_);
  std::vector<double> jacobian(n * n_params, 0.0);

  // Each component through the same kernel evaluate() uses, on dual numbers
  // in its own mean, sigma and shape: g[c] and its derivatives.
  std::vector<std::vector<Dual> > g(static_cast<std::size_t>(n_components_));
  std::vector<double> raw_mean(n_components_), raw_amplitude(n_components_);
  for (int c = 0; c < n_components_; ++c) {
    const std::size_t base = static_cast<std::size_t>(4 * c);
    raw_mean[c] = component_ports_[base]->get_value();
    raw_amplitude[c] = component_ports_[base + 3]->get_value();
    const Dual mean = Dual::variable(std::fabs(raw_mean[c]), 0);
    const Dual sigma = Dual::variable(component_ports_[base + 1]->get_value(), 1);
    const Dual shape = Dual::variable(component_ports_[base + 2]->get_value(), 2);
    g[c] = distance_between_gaussians_
               ? internal::distance_between_gaussian_t(axis_, mean, sigma)
               : internal::generalized_normal_density_t(axis_, mean, sigma, shape, true);
  }

  // The mixture D = sum_c |a_c| g_c and its normalisation p = D / S, as
  // gaussian_distance_mixture_impl combines them.
  std::vector<double> D(n, 0.0);
  for (int c = 0; c < n_components_; ++c) {
    const double w = std::fabs(raw_amplitude[c]);
    for (std::size_t j = 0; j < n; ++j) D[j] += w * g[c][j].v;
  }
  double S = 0.0;
  for (double v : D) S += v;
  if (!(S > 0.0)) return jacobian;
  for (int c = 0; c < n_components_; ++c) {
    const double w = std::fabs(raw_amplitude[c]);
    // |mean| and |amplitude|: sign(x), undefined at 0.
    const double s_mean = raw_mean[c] > 0.0 ? 1.0 : (raw_mean[c] < 0.0 ? -1.0 : 0.0);
    const double s_amplitude =
        raw_amplitude[c] > 0.0 ? 1.0 : (raw_amplitude[c] < 0.0 ? -1.0 : 0.0);
    const double signs[3] = {s_mean, 1.0, 1.0};
    double dS[3] = {0.0, 0.0, 0.0};
    double G = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      for (int k = 0; k < 3; ++k) dS[k] += w * g[c][j].d[k];
      G += g[c][j].v;
    }
    const std::size_t col = static_cast<std::size_t>(4 * c);
    for (std::size_t j = 0; j < n; ++j) {
      const double p = D[j] / S;
      const std::size_t row = j * n_params;
      for (int k = 0; k < 3; ++k) {
        jacobian[row + col + k] = signs[k] * (w * g[c][j].d[k] - p * dS[k]) / S;
      }
      jacobian[row + col + 3] = s_amplitude * (g[c][j].v - p * G) / S;
    }
  }
  return jacobian;
}

void GaussianDistances::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  const std::vector<double> axis =
      internal::configured_distance_axis(config, "GaussianDistances '" + get_name() + "'");
  if (!axis.empty()) set_axis(axis);
  if (config.has("number_of_components")) {
    set_number_of_components(config.get_int("number_of_components"));
  }
  if (config.has("distance_between_gaussians")) {
    set_distance_between_gaussians(config.get_bool("distance_between_gaussians"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

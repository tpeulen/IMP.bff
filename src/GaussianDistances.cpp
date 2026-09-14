/**\file GaussianDistances.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/GaussianDistances.h>
#include <IMP/bff/internal/DistanceAxis.h>
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

namespace {
inline double sign_of(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }
}  // namespace

std::vector<double> GaussianDistances::get_weights_jacobian() const {
  if (axis_.empty()) {
    throw std::domain_error("GaussianDistances '" + get_name() +
                            "' has no distance axis");
  }
  const std::size_t n = axis_.size();
  const std::size_t n_params = static_cast<std::size_t>(4 * n_components_);
  std::vector<double> jacobian(n * n_params, 0.0);
  const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);

  // Per component: the component as the mixture adds it (g), its derivative
  // by mean, sigma and shape (dg), and the raw amplitude.
  std::vector<std::vector<double> > g(n_components_), dg_mean(n_components_),
      dg_sigma(n_components_), dg_shape(n_components_);
  std::vector<double> raw_mean(n_components_), raw_amplitude(n_components_);
  for (int c = 0; c < n_components_; ++c) {
    const std::size_t base = static_cast<std::size_t>(4 * c);
    raw_mean[c] = component_ports_[base]->get_value();
    const double sigma = component_ports_[base + 1]->get_value();
    const double shape = component_ports_[base + 2]->get_value();
    raw_amplitude[c] = component_ports_[base + 3]->get_value();
    const double mean = std::fabs(raw_mean[c]);
    std::vector<double>& u = g[c];
    std::vector<double>& du_m = dg_mean[c];
    std::vector<double>& du_s = dg_sigma[c];
    std::vector<double>& du_k = dg_shape[c];
    u.assign(n, 0.0); du_m.assign(n, 0.0); du_s.assign(n, 0.0); du_k.assign(n, 0.0);
    if (sigma == 0.0) continue;
    if (distance_between_gaussians_) {
      // Unnormalised per component; the mixture is normalised once below.
      const double s2 = sigma * sigma;
      const double a = inv_sqrt_2pi / sigma;
      for (std::size_t j = 0; j < n; ++j) {
        const double r = axis_[j];
        if (mean > 0.0) {
          const double na = a * std::exp(-(r - mean) * (r - mean) / (2.0 * s2));
          const double nb = a * std::exp(-(r + mean) * (r + mean) / (2.0 * s2));
          u[j] = r / mean * (na - nb);
          du_m[j] = -u[j] / mean +
                    r / mean * (na * (r - mean) / s2 + nb * (r + mean) / s2);
          du_s[j] = r / mean *
                    (na * ((r - mean) * (r - mean) / (s2 * sigma) - 1.0 / sigma) -
                     nb * ((r + mean) * (r + mean) / (s2 * sigma) - 1.0 / sigma));
        } else {
          const double n0 = a * std::exp(-r * r / (2.0 * s2));
          u[j] = 2.0 * r * r / s2 * n0;
          du_s[j] = u[j] * (-2.0 / sigma + r * r / (s2 * sigma) - 1.0 / sigma);
        }
      }
      continue;
    }
    // The generalized normal: phi(z) with z a transform of t = (x - mean) / sigma,
    // normalised to unit sum on the axis.
    const double tiny = std::nextafter(1.0, 2.0) - 1.0;
    for (std::size_t j = 0; j < n; ++j) {
      const double t = (axis_[j] - mean) / sigma;
      double z, dz_dt, dz_dk;
      if (shape == 0.0) {
        z = t;
        dz_dt = 1.0;
        dz_dk = 0.5 * t * t;  // the limit of the transform's derivative at 0
      } else {
        double T = 1.0 - shape * t;
        if (T < 0.0) {
          T = tiny;
          z = -std::log(T) / shape;
          dz_dt = 0.0;
          dz_dk = 0.0;
        } else if (std::fabs(shape * t) < 1e-3) {
          // log(T)/k^2 + t/(kT) cancels for a small shape; its series does not.
          z = -std::log(T) / shape;
          dz_dt = 1.0 / T;
          const double kt = shape * t;
          dz_dk = t * t * (0.5 + kt * (2.0 / 3.0 + kt * (0.75 + 0.8 * kt)));
        } else {
          z = -std::log(T) / shape;
          dz_dt = 1.0 / T;
          dz_dk = std::log(T) / (shape * shape) + t / (shape * T);
        }
      }
      const double phi = inv_sqrt_2pi * std::exp(-0.5 * z * z);
      u[j] = phi;
      const double dphi_dz = -z * phi;
      du_m[j] = dphi_dz * dz_dt * (-1.0 / sigma);
      du_s[j] = dphi_dz * dz_dt * (-t / sigma);
      du_k[j] = dphi_dz * dz_dk;
    }
    // Unit sum per component: q = u / U, dq = (du - q dU) / U.
    double U = 0.0, dU_m = 0.0, dU_s = 0.0, dU_k = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      U += u[j]; dU_m += du_m[j]; dU_s += du_s[j]; dU_k += du_k[j];
    }
    if (!(U > 0.0)) {
      std::fill(u.begin(), u.end(), 0.0);
      std::fill(du_m.begin(), du_m.end(), 0.0);
      std::fill(du_s.begin(), du_s.end(), 0.0);
      std::fill(du_k.begin(), du_k.end(), 0.0);
      continue;
    }
    for (std::size_t j = 0; j < n; ++j) {
      const double q = u[j] / U;
      du_m[j] = (du_m[j] - q * dU_m) / U;
      du_s[j] = (du_s[j] - q * dU_s) / U;
      du_k[j] = (du_k[j] - q * dU_k) / U;
      u[j] = q;
    }
  }

  // The mixture D = sum_c |a_c| g_c, and p = D / S with S = sum_j D_j.
  std::vector<double> D(n, 0.0);
  for (int c = 0; c < n_components_; ++c) {
    const double w = std::fabs(raw_amplitude[c]);
    for (std::size_t j = 0; j < n; ++j) D[j] += w * g[c][j];
  }
  double S = 0.0;
  for (double v : D) S += v;
  if (!(S > 0.0)) return jacobian;
  for (int c = 0; c < n_components_; ++c) {
    const double w = std::fabs(raw_amplitude[c]);
    double dS_m = 0.0, dS_s = 0.0, dS_k = 0.0, G = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      dS_m += w * dg_mean[c][j];
      dS_s += w * dg_sigma[c][j];
      dS_k += w * dg_shape[c][j];
      G += g[c][j];
    }
    const double s_mean = sign_of(raw_mean[c]);
    const double s_amplitude = sign_of(raw_amplitude[c]);
    const std::size_t col = static_cast<std::size_t>(4 * c);
    for (std::size_t j = 0; j < n; ++j) {
      const double p = D[j] / S;
      const std::size_t row = j * n_params;
      jacobian[row + col] = s_mean * (w * dg_mean[c][j] - p * dS_m) / S;
      jacobian[row + col + 1] = (w * dg_sigma[c][j] - p * dS_s) / S;
      jacobian[row + col + 2] = (w * dg_shape[c][j] - p * dS_k) / S;
      jacobian[row + col + 3] = s_amplitude * (g[c][j] - p * G) / S;
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

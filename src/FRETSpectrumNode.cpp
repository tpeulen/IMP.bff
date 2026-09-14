/**\file FRETSpectrumNode.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/FRETSpectrumNode.h>
#include <limits>
#include <IMP/bff/internal/NodeConfig.h>
#include <IMP/bff/FRETOrientationFactor.h>
#include "internal/SpectrumNodeHelpers.h"
#include <IMP/bff/PhotophysicsLifetimeSpectrum.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// ------------------------------------------------------------ FRETSpectrumNode

FRETSpectrumNode::FRETSpectrumNode(const std::string& name) : GraphNode(name) {}

void FRETSpectrumNode::build_ports() {
  if (donor_port_ != nullptr) return;
  std::shared_ptr<GraphPort> donor(new GraphPort(std::vector<double>{1.0, 4.0}));
  donor->set_sanitize(false);
  add_input_port(donor_port_key(), donor);
  donor_port_ = donor.get();

  std::shared_ptr<GraphPort> distances(new GraphPort(std::vector<double>{1.0, 52.0}));
  distances->set_sanitize(false);
  add_input_port(distance_port_key(), distances);
  distance_port_ = distances.get();

  spectrum_node_detail::add_scalar_port(this, "x_donly", 0.0, &x_donly_port_);
  spectrum_node_detail::add_scalar_port(this, "forster_radius", 52.0, &forster_radius_port_);
  spectrum_node_detail::add_scalar_port(this, "tau0", 4.0, &tau0_port_);
  spectrum_node_detail::add_scalar_port(this, "kappa2", 2.0 / 3.0, &kappa2_port_);
  // Read only under `orientation: "static"`.
  std::shared_ptr<GraphPort> kappa2s(new GraphPort(std::vector<double>{1.0, 2.0 / 3.0}));
  kappa2s->set_sanitize(false);
  add_input_port("kappa2_distribution", kappa2s);
  set_valid(false);
}

void FRETSpectrumNode::evaluate() {
  if (donor_port_ == nullptr) {
    throw std::domain_error("FRETSpectrumNode '" + get_name() +
                            "' has no ports; call build_ports() first");
  }
  // By key, not through the cached pointer, for the reason
  // `PhotophysicsAnisotropySpectrumNode` gives: these two are the ports a caller replaces.
  const std::shared_ptr<GraphPort> donor_in = get_input_port(donor_port_key());
  const std::shared_ptr<GraphPort> distance_in = get_input_port(distance_port_key());
  if (!donor_in || !distance_in) {
    throw std::domain_error("FRETSpectrumNode '" + get_name() +
                            "' is missing one of its two spectrum ports");
  }
  const std::vector<double>& donor = donor_in->get_values_ref();
  const std::vector<double>& distances = distance_in->get_values_ref();
  spectrum_node_detail::check_interleaved("FRETSpectrumNode '" + get_name() + "' (donor)", donor);
  spectrum_node_detail::check_interleaved("FRETSpectrumNode '" + get_name() + "' (distances)",
                    distances);

  const double x_donly = std::fabs(x_donly_port_->get_value());
  const double forster_radius = forster_radius_port_->get_value();
  const double tau0 = tau0_port_->get_value();
  const double kappa2 = kappa2_port_->get_value();
  if (!(tau0 > 0.0)) {
    throw std::domain_error("FRETSpectrumNode '" + get_name() +
                            "': tau0 is not positive");
  }

  // The donor as *rates*, which is the space the two spectra combine in.
  const std::size_t n_donor = donor.size() / 2;
  std::vector<double> donor_rates(2 * n_donor);
  for (std::size_t i = 0; i < n_donor; ++i) {
    donor_rates[2 * i] = donor[2 * i];
    donor_rates[2 * i + 1] = 1.0 / donor[2 * i + 1];
  }

  // Each distance (and, for static dipoles, each orientation factor) as a
  // transfer rate: k = 3/2 kappa^2 / tau0 * (R0/r)^6.
  const std::vector<double> fret_rates =
      transfer_rates(distances, forster_radius, tau0, kappa2);
  const std::size_t n_distances = fret_rates.size() / 2;

  // Rates add: a quenched species decays at its own rate plus the transfer
  // rate, so the quenched spectrum is the Cartesian product with the
  // amplitudes multiplied. FRET-major, which is the order ChiSurf's
  // `ere2(fret, donor)` produces.
  std::vector<double> combined;
  combined.reserve(2 * (n_distances * n_donor + n_donor));
  for (std::size_t i = 0; i < n_distances; ++i) {
    for (std::size_t j = 0; j < n_donor; ++j) {
      combined.push_back(fret_rates[2 * i] * donor_rates[2 * j] *
                         (1.0 - x_donly));
      combined.push_back(fret_rates[2 * i + 1] + donor_rates[2 * j + 1]);
    }
  }
  // The donor-only fraction, appended unconditionally -- including at
  // `x_donly == 0`, where it carries zero amplitude. A node that sometimes
  // returns a shorter spectrum is a second code path for no gain, and the
  // length is observable.
  for (std::size_t j = 0; j < n_donor; ++j) {
    combined.push_back(donor_rates[2 * j] * x_donly);
    combined.push_back(donor_rates[2 * j + 1]);
  }

  // Back to lifetimes, which is what the instrument node reconvolves.
  spectrum_.resize(combined.size());
  for (std::size_t i = 0; i < combined.size() / 2; ++i) {
    spectrum_[2 * i] = combined[2 * i];
    spectrum_[2 * i + 1] = 1.0 / combined[2 * i + 1];
  }
  spectrum_node_detail::publish(this, spectrum_);
  // The transfer rates themselves, `(p, k_FRET)` per distance, for a caller
  // that composes species in rate space -- rates of a mixture add, lifetimes
  // do not. Published only where a port asks for it.
  if (const std::shared_ptr<GraphPort> rates = get_output_port("fret_rates")) {
    rates->set_value_vector(fret_rates);
  }
  set_valid(true);
}

std::string FRETSpectrumNode::describe() const {
  std::ostringstream out;
  out << "species out    : " << spectrum_.size() / 2 << "\n"
      << "R0 / tau0      : " << forster_radius_port_->get_value() << " / "
      << tau0_port_->get_value() << "\n";
  return out.str();
}

std::string FRETSpectrumNode::get_node_type() const { return "FRETSpectrumNode"; }

void FRETSpectrumNode::set_orientation(const std::string& orientation) {
  if (orientation != "dynamic" && orientation != "static" &&
      orientation != "static_isotropic") {
    throw std::domain_error("FRETSpectrumNode '" + get_name() +
                            "': orientation is 'dynamic', 'static' or "
                            "'static_isotropic', not '" + orientation + "'");
  }
  orientation_ = orientation;
  set_valid(false);
}

void FRETSpectrumNode::set_kappa2_bins(int n) {
  if (n < 0) {
    throw std::domain_error("FRETSpectrumNode::set_kappa2_bins: 0 or more");
  }
  kappa2_bins_ = n;
  set_valid(false);
}

void FRETSpectrumNode::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("orientation")) set_orientation(config.get_string("orientation"));
  if (config.has("kappa2_bins")) set_kappa2_bins(config.get_int("kappa2_bins"));
  if (config.has("kappa2_points")) {
    const int n = config.get_int("kappa2_points");
    if (n < 2) {
      throw std::domain_error("FRETSpectrumNode: kappa2_points is at least 2");
    }
    kappa2_points_ = n;
    set_valid(false);
  }
  config.apply_common(*this);
  config.require_all_used();
}

std::vector<double> FRETSpectrumNode::transfer_rates(
    const std::vector<double>& distances, double forster_radius, double tau0,
    double kappa2) const {
  const std::size_t n_distances = distances.size() / 2;
  const auto rate = [&](double r, double k2) {
    const double ratio = forster_radius / r;
    const double ratio3 = ratio * ratio * ratio;
    return 1.5 * k2 / tau0 * ratio3 * ratio3;
  };
  if (orientation_ == "dynamic") {
    std::vector<double> rates(2 * n_distances);
    for (std::size_t i = 0; i < n_distances; ++i) {
      rates[2 * i] = distances[2 * i];
      rates[2 * i + 1] = rate(distances[2 * i + 1], kappa2);
    }
    return rates;
  }

  // The orientation factors, as normalised masses.
  std::vector<double> k2, mass;
  if (orientation_ == "static_isotropic") {
    const std::vector<double> grid = [&] {
      std::vector<double> g(static_cast<std::size_t>(kappa2_points_));
      for (int i = 0; i < kappa2_points_; ++i) {
        g[static_cast<std::size_t>(i)] = 0.01 + (4.0 - 0.01) * i / (kappa2_points_ - 1);
      }
      return g;
    }();
    double* view = nullptr;
    int n_view = 0;
    isotropic_kappa2_density(grid, &view, &n_view);
    if (view == nullptr || n_view != static_cast<int>(grid.size())) {
      std::free(view);
      throw std::domain_error("FRETSpectrumNode '" + get_name() +
                              "': no isotropic kappa^2 density");
    }
    // A uniform grid: the density is proportional to each point's mass.
    k2 = grid;
    mass.assign(view, view + n_view);
    std::free(view);
  } else {
    const std::shared_ptr<GraphPort> port = get_input_port("kappa2_distribution");
    const std::vector<double>& values = port->get_values_ref();
    spectrum_node_detail::check_interleaved(
        "FRETSpectrumNode '" + get_name() + "' (kappa2)", values);
    for (std::size_t i = 0; i < values.size(); i += 2) {
      mass.push_back(std::fabs(values[i]));
      k2.push_back(values[i + 1]);
    }
  }
  double total = 0.0;
  for (double m : mass) total += m;
  if (!(total > 0.0)) {
    throw std::domain_error("FRETSpectrumNode '" + get_name() +
                            "': the kappa^2 distribution carries no weight");
  }
  double k2_mean = 0.0;
  for (std::size_t k = 0; k < mass.size(); ++k) {
    mass[k] /= total;
    k2_mean += mass[k] * k2[k];
  }

  if (kappa2_bins_ == 0) {
    // Every (distance, orientation factor) pair is a species of its own.
    std::vector<double> rates;
    rates.reserve(2 * n_distances * k2.size());
    for (std::size_t i = 0; i < n_distances; ++i) {
      for (std::size_t k = 0; k < k2.size(); ++k) {
        rates.push_back(distances[2 * i] * mass[k]);
        rates.push_back(rate(distances[2 * i + 1], k2[k]));
      }
    }
    return rates;
  }

  // By apparent distance: a pair transfers at the rate of distance
  // r (<k2>/k2)^(1/6) under <k2>, so histogramming apparent distances keeps
  // each pair's rate to within its bin. That histogram is every distance
  // times every ratio, which is outer_product_histogram's. Linear bins over
  // the products' range; every bin is kept, so the length of the result does
  // not depend on the parameters.
  if (!(k2_mean > 0.0)) {
    throw std::domain_error("FRETSpectrumNode '" + get_name() + "': <kappa^2> is not positive");
  }
  std::vector<double> r(n_distances), p(n_distances), ratio, ratio_mass;
  for (std::size_t i = 0; i < n_distances; ++i) {
    p[i] = distances[2 * i];
    r[i] = distances[2 * i + 1];
  }
  for (std::size_t k = 0; k < k2.size(); ++k) {
    if (!(k2[k] > 0.0) || mass[k] == 0.0) continue;
    ratio.push_back(std::pow(k2_mean / k2[k], 1.0 / 6.0));
    ratio_mass.push_back(mass[k]);
  }
  const std::size_t n_bins = static_cast<std::size_t>(kappa2_bins_);
  std::vector<double> rates(2 * n_bins, 0.0);
  if (ratio.empty() || r.empty()) return rates;
  double lo = *std::min_element(r.begin(), r.end()) *
              *std::min_element(ratio.begin(), ratio.end());
  double hi = *std::max_element(r.begin(), r.end()) *
              *std::max_element(ratio.begin(), ratio.end());
  if (!(hi > lo)) {
    lo -= 0.5;
    hi += 0.5;
  }
  const std::vector<double> hist =
      outer_product_histogram(r, p, ratio, ratio_mass, kappa2_bins_, lo, hi);
  const double width = (hi - lo) / static_cast<double>(n_bins);
  for (std::size_t bin = 0; bin < n_bins; ++bin) {
    rates[2 * bin] = hist[bin];
    rates[2 * bin + 1] = rate(lo + (bin + 0.5) * width, k2_mean);
  }
  return rates;
}

IMPBFF_END_NAMESPACE

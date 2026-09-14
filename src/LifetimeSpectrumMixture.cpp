/**
 * \file LifetimeSpectrumMixture.cpp
 * \brief Lifetime spectra of several species, weighted by fractions and joined.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/LifetimeSpectrumMixture.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

LifetimeSpectrumMixture::LifetimeSpectrumMixture(const std::string& name) : GraphNode(name) {}

void LifetimeSpectrumMixture::set_number_of_species(int n) {
  if (n < 1) {
    throw std::domain_error(
        "LifetimeSpectrumMixture::set_number_of_species: at least one lifetime spectrum");
  }
  n_species_ = n;
  set_valid(false);
}

void LifetimeSpectrumMixture::set_normalize_amplitudes(bool v) {
  normalize_amplitudes_ = v;
  set_valid(false);
}

void LifetimeSpectrumMixture::evaluate() {
  const std::string where = "LifetimeSpectrumMixture '" + get_name() + "'";
  if (n_species_ < 1) {
    throw std::domain_error(where + " mixes nothing; set number_of_species");
  }
  std::vector<double> fractions(static_cast<std::size_t>(n_species_));
  double total = 0.0;
  for (int k = 0; k < n_species_; ++k) {
    const std::string key = "x" + std::to_string(k);
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no fraction '" + key + "'");
    fractions[static_cast<std::size_t>(k)] = std::fabs(port->get_value());
    total += fractions[static_cast<std::size_t>(k)];
  }
  if (!(total > 0.0)) {
    throw std::domain_error(where + ": every fraction is zero");
  }
  out_.clear();
  for (int k = 0; k < n_species_; ++k) {
    const std::string key = "s" + std::to_string(k);
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no lifetime spectrum '" + key + "'");
    const std::vector<double>& spectrum = port->get_values_ref();
    if (spectrum.size() % 2 != 0) {
      throw std::domain_error(where + ": lifetime spectrum '" + key +
                              "' is not (amplitude, lifetime) pairs");
    }
    double scale = fractions[static_cast<std::size_t>(k)] / total;
    if (normalize_amplitudes_) {
      double amplitude = 0.0;
      for (std::size_t i = 0; i < spectrum.size(); i += 2) {
        amplitude += std::fabs(spectrum[i]);
      }
      // An empty species contributes nothing rather than dividing by zero.
      scale = amplitude > 0.0 ? scale / amplitude : 0.0;
    }
    for (std::size_t i = 0; i < spectrum.size(); i += 2) {
      out_.push_back(spectrum[i] * scale);
      out_.push_back(spectrum[i + 1]);
    }
  }
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error(where + " writes to the output keyed by its own "
                                    "name, which this node does not have");
  }
  out->set_value_vector(out_);
  set_valid(true);
}

std::string LifetimeSpectrumMixture::get_node_type() const { return "LifetimeSpectrumMixture"; }

void LifetimeSpectrumMixture::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("number_of_species")) {
    set_number_of_species(config.get_int("number_of_species"));
  }
  if (config.has("normalize_amplitudes")) {
    set_normalize_amplitudes(config.get_bool("normalize_amplitudes"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

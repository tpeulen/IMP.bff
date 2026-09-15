/**
 * \file SpectrumGrid.cpp
 * \brief A fixed grid as a spectrum of unit amplitudes.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SpectrumGrid.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

SpectrumGrid::SpectrumGrid(const std::string& name) : GraphNode(name) {}

void SpectrumGrid::set_logarithmic(bool v) {
  logarithmic_ = v;
  set_valid(false);
}

std::vector<double> SpectrumGrid::points(double from, double to, int bins, bool logarithmic) {
  if (!(to > from) || bins < 2 || (logarithmic && !(from > 0.0))) {
    throw std::domain_error(
        "SpectrumGrid: a grid needs from < to, at least two bins, and from > 0 when logarithmic");
  }
  std::vector<double> x(static_cast<std::size_t>(bins));
  for (int k = 0; k < bins; ++k) {
    const double f = static_cast<double>(k) / static_cast<double>(bins - 1);
    x[static_cast<std::size_t>(k)] = logarithmic ? from * std::pow(to / from, f) : from + (to - from) * f;
  }
  x.back() = to;
  return x;
}

void SpectrumGrid::evaluate() {
  const std::string where = "SpectrumGrid '" + get_name() + "'";
  const auto scalar = [&](const char* key) {
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no input '" + key + "'");
    return port->get_value();
  };
  const std::vector<double> x =
      points(scalar("from"), scalar("to"), static_cast<int>(std::lround(scalar("bins"))), logarithmic_);
  std::vector<double> pairs(2 * x.size());
  for (std::size_t k = 0; k < x.size(); ++k) {
    pairs[2 * k] = 1.0;
    pairs[2 * k + 1] = x[k];
  }
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) throw std::domain_error(where + " writes to the output keyed by its own name, which it does not have");
  out->set_value_vector(pairs);
  set_valid(true);
}

std::string SpectrumGrid::get_node_type() const { return "SpectrumGrid"; }

void SpectrumGrid::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("spacing")) {
    const std::string spacing = config.get_string("spacing");
    if (spacing != "linear" && spacing != "log") {
      throw std::domain_error("node type 'SpectrumGrid': spacing is 'linear' or 'log', not '" + spacing + "'");
    }
    set_logarithmic(spacing == "log");
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

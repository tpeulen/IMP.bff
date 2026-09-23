/**
 * \file FitObjective.cpp
 * \brief A graph node whose evaluation is a weighted residual vector.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/FitObjective.h>

#include <cmath>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

const char* const FitObjective::residuals_port_name = "residuals";

FitObjective::FitObjective(const std::string& name) : GraphNode(name) {}

FitObjective::~FitObjective() {}

std::shared_ptr<GraphPort> FitObjective::get_residuals_port() {
  std::shared_ptr<GraphPort> port = get_output_port(residuals_port_name);
  if (!port) {
    port = std::make_shared<GraphPort>(std::vector<double>(), false, true,
                                       false, false, 0.0, 0.0,
                                       GRAPH_PORT_FLOAT_VECTOR);
    add_output_port(residuals_port_name, port);
  }
  // Residuals are fit transport: a NaN has to reach the minimiser, which
  // is where it is diagnosed, rather than be cleaned into a number.
  port->set_sanitize(false);
  return port;
}

const std::vector<double>& FitObjective::get_residuals() const {
  static const std::vector<double> none;
  const std::shared_ptr<GraphPort> port = get_output_port(residuals_port_name);
  return port ? port->get_values_ref() : none;
}

void FitObjective::set_residuals(const std::vector<double>& residuals) {
  get_residuals_port()->set_value_vector(residuals);
}

double FitObjective::get_chi2() const {
  double chi2 = 0.0;
  for (double r : get_residuals()) chi2 += r * r;
  return std::isnan(chi2) ? std::numeric_limits<double>::infinity() : chi2;
}

double FitObjective::get_chi2r(int n_free) const {
  const double dof = static_cast<double>(get_number_of_residuals()) -
                     static_cast<double>(n_free) - 1.0;
  return get_chi2() / dof;
}

IMPBFF_END_NAMESPACE

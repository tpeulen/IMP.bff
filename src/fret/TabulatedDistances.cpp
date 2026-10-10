/**
 * \file TabulatedDistances.cpp
 * \brief Given distance distributions on a common axis, mixed by fractions.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/TabulatedDistances.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

TabulatedDistances::TabulatedDistances(const std::string& name) : GraphNode(name) {}

void TabulatedDistances::set_number_of_distributions(int n) {
  if (n < 1) {
    throw std::domain_error("TabulatedDistances::set_number_of_distributions: at least one");
  }
  n_distributions_ = n;
  set_valid(false);
}

void TabulatedDistances::set_threshold(double relative) {
  if (!(relative >= 0.0 && relative < 1.0)) {
    throw std::domain_error("TabulatedDistances::set_threshold: in [0, 1)");
  }
  threshold_ = relative;
  set_valid(false);
}

void TabulatedDistances::read_inputs(std::vector<double>* axis,
                                     std::vector<std::vector<double> >* q,
                                     std::vector<double>* fractions) const {
  const std::string where = "TabulatedDistances '" + get_name() + "'";
  if (n_distributions_ < 1) {
    throw std::domain_error(where + " has no distributions; set number_of_distributions");
  }
  const std::shared_ptr<GraphPort> axis_port = get_input_port("axis");
  if (!axis_port) throw std::domain_error(where + " has no axis");
  *axis = axis_port->get_values_ref();
  q->assign(static_cast<std::size_t>(n_distributions_), std::vector<double>());
  fractions->assign(static_cast<std::size_t>(n_distributions_), 0.0);
  for (int k = 0; k < n_distributions_; ++k) {
    const std::string suffix = std::to_string(k);
    const std::shared_ptr<GraphPort> d = get_input_port("distribution" + suffix);
    const std::shared_ptr<GraphPort> x = get_input_port("fraction" + suffix);
    if (!d || !x) {
      throw std::domain_error(where + " needs distribution" + suffix + " and fraction" + suffix);
    }
    std::vector<double>& qk = (*q)[static_cast<std::size_t>(k)];
    qk = d->get_values_ref();
    if (qk.size() != axis->size()) {
      throw std::domain_error(where + ": distribution" + suffix + " has " +
                              std::to_string(qk.size()) + " values against an axis of " +
                              std::to_string(axis->size()));
    }
    double total = 0.0;
    for (double v : qk) total += v;
    if (total > 0.0) {
      for (double& v : qk) v /= total;
    }
    (*fractions)[static_cast<std::size_t>(k)] = x->get_value();
  }
}

void TabulatedDistances::evaluate() {
  std::vector<double> axis, fractions;
  std::vector<std::vector<double> > q;
  read_inputs(&axis, &q, &fractions);
  const std::size_t n = axis.size();
  std::vector<double> p(n, 0.0);
  for (std::size_t k = 0; k < q.size(); ++k) {
    const double w = std::fabs(fractions[k]);
    for (std::size_t j = 0; j < n; ++j) p[j] += w * q[k][j];
  }
  double total = 0.0;
  for (double v : p) total += v;
  if (total > 0.0) {
    for (double& v : p) v /= total;
  }
  if (threshold_ > 0.0 && n > 0) {
    const double cut = threshold_ * *std::max_element(p.begin(), p.end());
    for (double& v : p) {
      if (v < cut) v = 0.0;
    }
  }
  out_.resize(2 * n);
  for (std::size_t j = 0; j < n; ++j) {
    out_[2 * j] = p[j];
    out_[2 * j + 1] = axis[j];
  }
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error("TabulatedDistances '" + get_name() +
                            "' writes to the output keyed by its own name");
  }
  out->set_value_vector(out_);
  set_valid(true);
}

std::vector<std::string> TabulatedDistances::get_parameter_names() const {
  std::vector<std::string> names;
  for (int k = 0; k < n_distributions_; ++k) names.push_back("fraction" + std::to_string(k));
  return names;
}

std::vector<double> TabulatedDistances::get_weights_jacobian() const {
  std::vector<double> axis, fractions;
  std::vector<std::vector<double> > q;
  read_inputs(&axis, &q, &fractions);
  const std::size_t n = axis.size();
  const std::size_t n_k = q.size();
  std::vector<double> D(n, 0.0), G(n_k, 0.0);
  for (std::size_t k = 0; k < n_k; ++k) {
    const double w = std::fabs(fractions[k]);
    for (std::size_t j = 0; j < n; ++j) {
      D[j] += w * q[k][j];
      G[k] += q[k][j];
    }
  }
  double S = 0.0;
  for (double v : D) S += v;
  std::vector<double> jacobian(n * n_k, 0.0);
  if (!(S > 0.0)) return jacobian;
  double cut = 0.0;
  if (threshold_ > 0.0 && n > 0) cut = threshold_ * *std::max_element(D.begin(), D.end()) / S;
  for (std::size_t j = 0; j < n; ++j) {
    const double p = D[j] / S;
    if (threshold_ > 0.0 && p < cut) continue;
    for (std::size_t k = 0; k < n_k; ++k) {
      const double sign = fractions[k] > 0.0 ? 1.0 : (fractions[k] < 0.0 ? -1.0 : 0.0);
      jacobian[j * n_k + k] = sign * (q[k][j] - p * G[k]) / S;
    }
  }
  return jacobian;
}

std::string TabulatedDistances::get_node_type() const { return "TabulatedDistances"; }

void TabulatedDistances::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("number_of_distributions")) {
    set_number_of_distributions(config.get_int("number_of_distributions"));
  }
  if (config.has("threshold")) set_threshold(config.get_double("threshold"));
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

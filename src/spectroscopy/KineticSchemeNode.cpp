/**
 * \file KineticSchemeNode.cpp
 * \brief A chain of interconverting states, seen by TCSPC and by FCS at once.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/KineticSchemeNode.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

std::string indexed(const char* stem, int i) {
  std::ostringstream key;
  key << stem << i;
  return key.str();
}

void ensure_scalar(GraphNode* node, const std::string& key, double value) {
  if (!node->get_input_port(key)) {
    node->add_input_port(key, std::make_shared<GraphPort>(value));
  }
}

double read(const GraphNode& node, const std::string& key) {
  return node.get_input_port(key)->get_value();
}

}  // namespace

KineticSchemeNode::KineticSchemeNode(const std::string& name) : GraphNode(name) {}

void KineticSchemeNode::set_number_of_states(int n) {
  if (n <= 0) {
    throw std::domain_error("KineticSchemeNode: a scheme has at least one state");
  }
  for (int i = 0; i < n; ++i) ensure_scalar(this, indexed("t", i), 1.0 + i);
  for (int i = 1; i < n; ++i) {
    ensure_scalar(this, indexed("kf", i), 1.0);
    ensure_scalar(this, indexed("kb", i), 1.0);
  }
  if (!get_input_port("x")) {
    add_input_port("x", std::make_shared<GraphPort>(std::vector<double>(1, 0.0)));
  }
  n_states_ = n;
  set_valid(false);
}

std::vector<double> KineticSchemeNode::get_relaxation_rates() const {
  std::vector<double> rates;
  for (double lambda : eigenvalues_) {
    if (lambda < -1e-12) rates.push_back(-lambda);
  }
  std::sort(rates.begin(), rates.end());
  return rates;
}

void KineticSchemeNode::evaluate() {
  if (n_states_ <= 0) {
    throw std::domain_error("KineticSchemeNode '" + get_name() +
                            "' has no states; call set_number_of_states() first");
  }
  const int n = n_states_;
  std::vector<double> tau(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) tau[static_cast<std::size_t>(i)] = std::fabs(read(*this, indexed("t", i)));

  // Stationary populations from detailed balance along the chain.
  std::vector<double> kf(static_cast<std::size_t>(std::max(0, n - 1)));
  std::vector<double> kb(kf.size());
  std::vector<double> p(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i + 1 < n; ++i) {
    kf[static_cast<std::size_t>(i)] = std::max(std::fabs(read(*this, indexed("kf", i + 1))), 1e-300);
    kb[static_cast<std::size_t>(i)] = std::max(std::fabs(read(*this, indexed("kb", i + 1))), 1e-300);
    p[static_cast<std::size_t>(i) + 1] =
        p[static_cast<std::size_t>(i)] * kf[static_cast<std::size_t>(i)] / kb[static_cast<std::size_t>(i)];
  }
  double total = 0.0;
  for (double v : p) total += v;
  for (double& v : p) v /= total;
  populations_ = p;

  // S = D^1/2 Q D^-1/2 is symmetric for a chain; its eigenvectors give P(tau).
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i + 1 < n; ++i) {
    const double off = std::sqrt(kf[static_cast<std::size_t>(i)] * kb[static_cast<std::size_t>(i)]);
    s(i, i + 1) = off;
    s(i + 1, i) = off;
    s(i, i) -= kf[static_cast<std::size_t>(i)];
    s(i + 1, i + 1) -= kb[static_cast<std::size_t>(i)];
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(s);
  const Eigen::VectorXd lambda = solver.eigenvalues();
  const Eigen::MatrixXd& v = solver.eigenvectors();
  eigenvalues_.assign(lambda.data(), lambda.data() + n);
  double mean_brightness = 0.0;
  for (int i = 0; i < n; ++i) mean_brightness += tau[static_cast<std::size_t>(i)] * p[static_cast<std::size_t>(i)];
  std::vector<double> weight(static_cast<std::size_t>(n), 0.0);
  for (int k = 0; k < n; ++k) {
    double w = 0.0;
    for (int i = 0; i < n; ++i) {
      w += tau[static_cast<std::size_t>(i)] * std::sqrt(p[static_cast<std::size_t>(i)]) * v(i, k);
    }
    weight[static_cast<std::size_t>(k)] = w * w / (mean_brightness * mean_brightness);
  }

  const std::shared_ptr<GraphPort> spectrum_port = get_output_port("lifetime_spectrum");
  if (spectrum_port) {
    std::vector<double> spectrum;
    for (int i = 0; i < n; ++i) {
      spectrum.push_back(p[static_cast<std::size_t>(i)]);
      spectrum.push_back(tau[static_cast<std::size_t>(i)]);
    }
    spectrum_port->set_sanitize(false);
    spectrum_port->set_value_vector(spectrum);
  }
  const std::shared_ptr<GraphPort> correlation_port = get_output_port("correlation");
  if (correlation_port) {
    const std::vector<double>& lag = get_input_port("x")->get_values_ref();
    std::vector<double> correlation(lag.size(), 0.0);
    for (std::size_t j = 0; j < lag.size(); ++j) {
      double value = 0.0;
      for (int k = 0; k < n; ++k) {
        // The stationary mode carries exactly 1; the rest decay.
        value += weight[static_cast<std::size_t>(k)] *
                 std::exp(std::min(0.0, lambda(k)) * lag[j]);
      }
      correlation[j] = value;
    }
    correlation_port->set_sanitize(false);
    correlation_port->set_value_vector(correlation);
  }
  set_valid(true);
}

std::string KineticSchemeNode::get_node_type() const { return "KineticSchemeNode"; }

void KineticSchemeNode::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("number_of_states")) set_number_of_states(config.get_int("number_of_states"));
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE

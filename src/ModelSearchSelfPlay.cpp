/**
 * \file ModelSearchSelfPlay.cpp
 * \brief A model family teaching itself where its parameters start.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ModelSearchSelfPlay.h>
#include <IMP/bff/internal/AdamUpdate.h>

#include <IMP/bff/NeuralNet.h>
// The vendored kernels are placed in this module's namespace, as
// NeuralNet.cpp places them -- one copy, one home, and the unity build
// cannot end up with two opinions about where mlpcore lives.
#define TTTRLIB_MLPCORE_NAMESPACE IMP::bff::internal
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! How many points of each simulated curve the features keep.
const int kProfile = 16;
//! Plus its total and its peak position, which carry the scale and the shape.
const int kPerDataset = kProfile + 2;

//! A positive quantity is placed on a log scale; anything else linearly.
bool is_log_scaled(double lower, double upper, double reference) {
  return lower > 0.0 && reference > 0.0 && upper > lower;
}

//! Where a value sits relative to the family's own declared start.
/*!
    Not relative to the bounds. A bound is deliberately generous -- an FCS
    baseline may sit anywhere in +/-1e6 -- so a network scored against it
    learns almost nothing and proposes almost anything: trained that way, the
    proposer offered a baseline of 94501 for a curve whose baseline is 1.01.

    The declared start is computed from the measurement, so it is already in
    the right region, and what is left to learn is the *correction* to it.
    That is a small, well-conditioned quantity, and it keeps the proposal
    anchored to the data at inference time rather than to a constant.
*/
double to_unit(double value, double reference, double spread, double lower,
               double upper) {
  double offset;
  if (is_log_scaled(lower, upper, reference) && value > 0.0) {
    offset = std::log10(value / reference) / spread;
  } else {
    const double scale = std::max(std::fabs(reference), 1e-12) * spread;
    offset = (value - reference) / scale;
  }
  return 0.5 + 0.5 * std::min(1.0, std::max(-1.0, offset));
}

double from_unit(double unit, double reference, double spread, double lower,
                 double upper) {
  const double offset = 2.0 * std::min(1.0, std::max(0.0, unit)) - 1.0;
  double value;
  if (is_log_scaled(lower, upper, reference)) {
    value = reference * std::pow(10.0, spread * offset);
  } else {
    value = reference + offset * std::max(std::fabs(reference), 1e-12) * spread;
  }
  return std::min(upper, std::max(lower, value));
}

//! A curve reduced to what a fitter could see before it knows the answer.
/*!
    On a log scale, and *not* normalised. An earlier version divided by the
    peak to give the network a shape rather than a brightness -- which threw
    away the brightness, and then asked it to predict the amplitude and the
    baseline from what was left. It learned nothing, at exactly the mean
    squared error of predicting the mean. Scale is signal here.
*/
void describe_curve(const std::vector<double>& curve, std::vector<double>& out) {
  const std::size_t n = curve.size();
  double total = 0.0;
  double peak = 0.0;
  std::size_t peak_at = 0;
  for (std::size_t i = 0; i < n; ++i) {
    total += curve[i];
    if (curve[i] > peak) { peak = curve[i]; peak_at = i; }
  }
  for (int k = 0; k < kProfile; ++k) {
    const double f = static_cast<double>(k) / static_cast<double>(kProfile - 1);
    const std::size_t i = n == 0 ? 0 : static_cast<std::size_t>(
        std::min<double>(n - 1, std::pow(static_cast<double>(n - 1) + 1.0, f) - 1.0));
    // log1p keeps a zero bin finite and a counting curve comparable with a
    // correlation curve, whose values sit near one rather than near 10^4.
    out.push_back(std::log10(1.0 + std::max(0.0, curve[i])));
  }
  out.push_back(total > 0.0 ? std::log10(total) : 0.0);
  out.push_back(n == 0 ? 0.0 : static_cast<double>(peak_at) / static_cast<double>(n));
}

}  // namespace

struct ModelSearchSelfPlay::Impl {
  ModelSearchSpec spec;
  std::string structure;
  double spread = 1.0;
  bool poisson = true;
  int episodes = 0;
  int n_features = 0;
  std::vector<double> features;
  std::vector<double> targets;
  std::vector<std::string> target_ids;
  std::vector<double> lower;
  std::vector<double> upper;
  std::vector<double> error;

  explicit Impl(const ModelSearchSpec& s) : spec(s) {}
};

ModelSearchSelfPlay::ModelSearchSelfPlay(const ModelSearchSpec& spec)
    : impl_(new Impl(spec)) {}
ModelSearchSelfPlay::~ModelSearchSelfPlay() {}

void ModelSearchSelfPlay::set_structure(const std::string& key) {
  impl_->structure = key;
}
const std::string& ModelSearchSelfPlay::get_structure() const {
  return impl_->structure;
}
void ModelSearchSelfPlay::set_spread(double decades) {
  if (!(decades > 0.0)) {
    throw std::domain_error("self play: the spread must be positive");
  }
  impl_->spread = decades;
}
double ModelSearchSelfPlay::get_spread() const { return impl_->spread; }
void ModelSearchSelfPlay::set_poisson(bool value) { impl_->poisson = value; }
bool ModelSearchSelfPlay::get_poisson() const { return impl_->poisson; }

int ModelSearchSelfPlay::get_number_of_episodes() const {
  return impl_->episodes;
}
int ModelSearchSelfPlay::get_number_of_features() const {
  return impl_->n_features;
}
int ModelSearchSelfPlay::get_number_of_targets() const {
  return static_cast<int>(impl_->target_ids.size());
}
const std::vector<double>& ModelSearchSelfPlay::get_features() const {
  return impl_->features;
}
const std::vector<double>& ModelSearchSelfPlay::get_targets() const {
  return impl_->targets;
}
std::vector<std::string> ModelSearchSelfPlay::get_target_ids() const {
  return impl_->target_ids;
}
std::vector<double> ModelSearchSelfPlay::get_training_error() const {
  return impl_->error;
}

void ModelSearchSelfPlay::generate(int episodes, unsigned int seed) {
  if (episodes <= 0) throw std::domain_error("self play: no episodes to play");
  if (impl_->structure.empty()) {
    throw std::domain_error("self play: no structure to play");
  }
  std::shared_ptr<MultiStructureModelSearchProblem> problem = impl_->spec.build();
  const std::vector<std::string> datasets =
      problem->get_structure_curve_datasets(impl_->structure);
  if (datasets.empty()) {
    throw std::domain_error("self play: structure '" + impl_->structure +
                            "' produces no curve to learn from");
  }
  // Activating the structure is what decides which parameters it frees, and
  // only those are worth proposing: a fixed one is already where it belongs.
  problem->activate_structure(impl_->structure);
  const std::vector<std::string> ids = problem->get_parameter_ids();
  std::vector<std::shared_ptr<GraphPort> > ports;
  std::vector<double> reference;
  impl_->target_ids.clear();
  impl_->lower.clear();
  impl_->upper.clear();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::shared_ptr<GraphPort> port = problem->get_parameter(ids[i]);
    if (port->get_fixed()) continue;
    impl_->target_ids.push_back(ids[i]);
    ports.push_back(port);
    reference.push_back(port->get_value());
    impl_->lower.push_back(port->get_lower_bound());
    impl_->upper.push_back(port->get_upper_bound());
  }
  if (ports.empty()) {
    throw std::domain_error("self play: structure '" + impl_->structure +
                            "' frees nothing to propose");
  }

  impl_->n_features = static_cast<int>(datasets.size()) * kPerDataset;
  impl_->episodes = episodes;
  impl_->features.clear();
  impl_->targets.clear();
  impl_->features.reserve(static_cast<std::size_t>(episodes) * impl_->n_features);
  impl_->targets.reserve(static_cast<std::size_t>(episodes) * ports.size());

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (int episode = 0; episode < episodes; ++episode) {
    // Sample around the family's own declared start rather than across the
    // whole bound: an episode has to remain a measurement someone could have
    // taken, and the bounds are deliberately generous.
    for (std::size_t p = 0; p < ports.size(); ++p) {
      const double lo = impl_->lower[p];
      const double hi = impl_->upper[p];
      double value;
      if (is_log_scaled(lo, hi, reference[p])) {
        value = reference[p] *
                std::pow(10.0, impl_->spread * uniform(rng));
      } else {
        // Relative to the declared start, never to the bound. An FCS
        // baseline is bounded at +/-1e6 and sits near 1: sampling across the
        // bound produced a flat curve of 10^6 every episode, identical
        // features, and a network that could only predict the mean.
        const double span = std::max(std::fabs(reference[p]), 1e-12) *
                            impl_->spread;
        value = reference[p] + span * uniform(rng);
      }
      ports[p]->set_value(std::min(hi, std::max(lo, value)));
    }
    for (std::size_t d = 0; d < datasets.size(); ++d) {
      std::vector<double> curve = problem->get_structure_output(
          impl_->structure,
          problem->get_structure_curve_node(impl_->structure, datasets[d]));
      if (impl_->poisson) {
        for (std::size_t i = 0; i < curve.size(); ++i) {
          const double mean = std::max(0.0, curve[i]);
          std::poisson_distribution<int> counts(mean);
          curve[i] = static_cast<double>(counts(rng));
        }
      }
      describe_curve(curve, impl_->features);
    }
    for (std::size_t p = 0; p < ports.size(); ++p) {
      impl_->targets.push_back(to_unit(ports[p]->get_value(), reference[p],
                                       impl_->spread, impl_->lower[p],
                                       impl_->upper[p]));
    }
  }
}

std::string ModelSearchSelfPlay::train(const std::vector<int>& hidden,
                                       int epochs, double learning_rate) {
  if (impl_->episodes <= 0) {
    throw std::domain_error("self play: nothing generated to train on");
  }
  const int n_in = impl_->n_features;
  const int n_out = get_number_of_targets();

  internal::MlpModel model;
  std::mt19937 rng(12345);
  int previous = n_in;
  std::vector<int> widths(hidden);
  widths.push_back(n_out);
  for (std::size_t l = 0; l < widths.size(); ++l) {
    internal::DenseLayer layer;
    layer.n_in = previous;
    layer.n_out = widths[l];
    layer.activation = (l + 1 == widths.size()) ? internal::Activation::Sigmoid
                                                : internal::Activation::Tanh;
    // Targets are unit-scaled, so a logistic head cannot propose a start
    // outside the parameter's own bounds -- the proposal is admissible by
    // construction rather than by a clamp afterwards.
    const double limit = std::sqrt(6.0 / (layer.n_in + layer.n_out));
    std::uniform_real_distribution<double> init(-limit, limit);
    layer.weight.resize(static_cast<std::size_t>(layer.n_in) * layer.n_out);
    for (std::size_t k = 0; k < layer.weight.size(); ++k) layer.weight[k] = init(rng);
    layer.bias.assign(layer.n_out, 0.0);
    model.layers.push_back(layer);
    previous = layer.n_out;
  }
  // Standardise the inputs from the episodes themselves; the outputs are
  // already in [0, 1] and the head keeps them there.
  model.x_scaler.mean.assign(n_in, 0.0);
  model.x_scaler.scale.assign(n_in, 1.0);
  for (int j = 0; j < n_in; ++j) {
    double sum = 0.0;
    for (int r = 0; r < impl_->episodes; ++r) sum += impl_->features[r * n_in + j];
    const double mean = sum / impl_->episodes;
    double var = 0.0;
    for (int r = 0; r < impl_->episodes; ++r) {
      const double d = impl_->features[r * n_in + j] - mean;
      var += d * d;
    }
    model.x_scaler.mean[j] = mean;
    model.x_scaler.scale[j] = std::sqrt(var / impl_->episodes) + 1e-12;
  }
  model.validate();

  std::vector<double> flat;
  internal::mlpcore::flatten(model.layers, flat);
  // Adam, the one implementation (tttrlib's AdamUpdate.h, vendored): state
  // reset once, one step per epoch -- its step count is the epoch.
  tttrlib::AdamState adam;
  adam.reset(flat.size());
  std::vector<double> prediction, dparams, dX, dV, residual;
  const int rows = impl_->episodes;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    std::vector<double> dy, d2y;
    internal::mlpcore::model_predict(model, impl_->features.data(), rows, 0, nullptr,
                            prediction, dy, d2y);
    residual.assign(static_cast<std::size_t>(rows) * n_out, 0.0);
    for (std::size_t k = 0; k < residual.size(); ++k) {
      residual[k] = 2.0 * (prediction[k] - impl_->targets[k]) / rows;
    }
    internal::mlpcore::model_backward(model, impl_->features.data(), rows, nullptr,
                             residual.data(), nullptr, nullptr, dparams, dX, dV);
    tttrlib::adam_update(flat.data(), dparams.data(), flat.size(), adam, learning_rate);
    internal::mlpcore::unflatten(model.layers, flat.data(), flat.size());
  }

  std::vector<double> dy, d2y;
  internal::mlpcore::model_predict(model, impl_->features.data(), rows, 0, nullptr,
                          prediction, dy, d2y);
  impl_->error.assign(n_out, 0.0);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < n_out; ++j) {
      const double d = prediction[r * n_out + j] - impl_->targets[r * n_out + j];
      impl_->error[j] += d * d / rows;
    }
  }

  return internal::mlpcore::model_to_json<nlohmann::json>(model).dump();
}

std::vector<double> ModelSearchSelfPlay::propose(
    const std::string& network) const {
  if (impl_->target_ids.empty()) {
    throw std::domain_error("self play: generate episodes before proposing");
  }
  std::shared_ptr<MultiStructureModelSearchProblem> problem =
      impl_->spec.build();
  problem->activate_structure(impl_->structure);
  const std::vector<std::string> datasets =
      problem->get_structure_curve_datasets(impl_->structure);
  // The measurement as it is, not as an episode imagined it.
  std::vector<double> features;
  for (std::size_t d = 0; d < datasets.size(); ++d) {
    describe_curve(impl_->spec.get_dataset_values(datasets[d]), features);
  }
  NeuralNet net(network);
  double* out = nullptr;
  int n_out = 0;
  net.predict(features, 1, &out, &n_out);
  const std::vector<std::string> ids = problem->get_parameter_ids();
  std::vector<double> proposal;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    proposal.push_back(problem->get_parameter(ids[i])->get_value());
  }
  for (std::size_t t = 0; t < impl_->target_ids.size() &&
                          static_cast<int>(t) < n_out; ++t) {
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] != impl_->target_ids[t]) continue;
      proposal[i] = from_unit(out[t], proposal[i], impl_->spread,
                              impl_->lower[t], impl_->upper[t]);
    }
  }
  std::free(out);
  return proposal;
}

IMPBFF_END_NAMESPACE

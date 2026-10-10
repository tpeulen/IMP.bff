/**
 * \file ModelSearchSelfPlay.cpp
 * \brief A model family teaching itself where its parameters start.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ModelSearchSelfPlay.h>
#include <IMP/bff/internal/AdamUpdate.h>

#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/PhotonExperiment.h>
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/NetworkDocument.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <map>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! How many points of each simulated curve the features keep.
const int kProfile = 16;
//! Plus its total and its peak position, which carry the scale and the shape.
const int kPerDataset = kProfile + 2;

bool is_log_scaled(double lower, double upper, double reference);

void sample_ports(const std::shared_ptr<FittingModelSearchProblem>& problem,
                  double spread, std::mt19937& rng) {
  const std::vector<std::string> ids = problem->get_parameter_ids();
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::shared_ptr<GraphPort> port = problem->get_parameter(ids[i]);
    if (port->get_fixed()) continue;
    const double reference = port->get_value();
    const double lo = port->get_lower_bound();
    const double hi = port->get_upper_bound();
    const double value = is_log_scaled(lo, hi, reference)
        ? reference * std::pow(10.0, spread * uniform(rng))
        : reference + std::max(std::fabs(reference), 1e-12) * spread * uniform(rng);
    port->set_value(std::min(hi, std::max(lo, value)));
  }
}

void add_matching_noise(FitDataset& dataset, std::vector<double>& values,
                        bool enabled, std::mt19937& rng) {
  if (!enabled) return;
  if (dataset.get_noise_family() == FIT_NOISE_FAMILY_POISSON) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::poisson_distribution<int> draw(std::max(0.0, values[i]));
      values[i] = static_cast<double>(draw(rng));
    }
    return;
  }
  double* view = nullptr;
  int n = 0;
  dataset.variance(values, &view, &n);
  for (int i = 0; i < n; ++i) {
    std::normal_distribution<double> draw(0.0, std::sqrt(std::max(0.0, view[i])));
    values[static_cast<std::size_t>(i)] += draw(rng);
  }
  std::free(view);
}

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
  bool photons = false;

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
  std::shared_ptr<FittingModelSearchProblem> problem = impl_->spec.build();
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

MsgpackBytes ModelSearchSelfPlay::train(const std::vector<int>& hidden,
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
  // Adam, the one implementation (internal/AdamUpdate.h): state
  // reset once, one step per epoch -- its step count is the epoch.
  internal::AdamState adam;
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
    internal::adam_update(flat.data(), dparams.data(), flat.size(), adam, learning_rate);
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

  return internal::model_to_msgpack(model);
}

std::vector<double> ModelSearchSelfPlay::propose(
    const MsgpackBytes& network) const {
  if (impl_->target_ids.empty()) {
    throw std::domain_error("self play: generate episodes before proposing");
  }
  std::shared_ptr<FittingModelSearchProblem> problem =
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

void ModelSearchSelfPlay::set_photon_simulation(bool value) {
  if (value && !PhotonExperiment::get_available()) {
    throw std::domain_error(
        "self play: photon simulation needs a bff build linked with TTTRLib");
  }
  impl_->photons = value;
}
bool ModelSearchSelfPlay::get_photon_simulation() const { return impl_->photons; }

ModelSearchSpec ModelSearchSelfPlay::simulate(const std::string& structure_key,
                                              unsigned int seed) const {
  std::mt19937 rng(seed);
  const std::shared_ptr<FittingModelSearchProblem> generator = impl_->spec.build();
  generator->activate_structure(structure_key);
  sample_ports(generator, impl_->spread, rng);
  ModelSearchSpec simulated(impl_->spec);
  for (const std::string& name : generator->get_structure_curve_datasets(structure_key)) {
    FitDataset measurement = impl_->spec.get_dataset(name);
    std::vector<double> values = generator->get_structure_output(
        structure_key, generator->get_structure_curve_node(structure_key, name));
    if (impl_->photons && measurement.get_noise_family() == FIT_NOISE_FAMILY_POISSON) {
      values = PhotonExperiment::record_pattern(values, rng());
    } else {
      add_matching_noise(measurement, values, impl_->poisson, rng);
    }
    measurement.replace_values(values);
    simulated.set_dataset(name, measurement);
  }
  return simulated;
}

ModelSearchPolicyData ModelSearchSelfPlay::generate_policy(int episodes,
                                                           unsigned int seed) {
  if (episodes <= 0) throw std::domain_error("self play: no policy episodes to play");
  const std::shared_ptr<FittingModelSearchProblem> layout = impl_->spec.build();
  const std::vector<std::string> keys = layout->get_structure_keys();

  // The declared moves out of every structure, and where each leads first.
  std::map<std::string, ModelSearchActions> moves;
  std::map<std::string, int> terminal;
  for (const std::string& key : keys) {
    moves[key] = layout->get_actions(ModelSearchState(key, key, 0.0));
    for (std::size_t i = 0; i < moves[key].size(); ++i) {
      if (moves[key][i].get_terminal() && !terminal.count(key)) {
        terminal[key] = static_cast<int>(i);
      }
    }
  }
  //! first[source][target]: the move that starts a shortest path there.
  std::map<std::string, std::map<std::string, int> > first;
  for (const std::string& source : keys) {
    std::map<std::string, int>& reach = first[source];
    std::deque<std::pair<std::string, int> > frontier;
    std::set<std::string> seen;
    seen.insert(source);
    const ModelSearchActions& out = moves[source];
    for (std::size_t i = 0; i < out.size(); ++i) {
      const std::string& next = out[i].get_predicted_state_key();
      if (out[i].get_terminal() || seen.count(next)) continue;
      seen.insert(next);
      reach[next] = static_cast<int>(i);
      frontier.push_back(std::make_pair(next, static_cast<int>(i)));
    }
    while (!frontier.empty()) {
      const std::pair<std::string, int> at = frontier.front();
      frontier.pop_front();
      const ModelSearchActions& onward = moves[at.first];
      for (std::size_t i = 0; i < onward.size(); ++i) {
        const std::string& next = onward[i].get_predicted_state_key();
        if (onward[i].get_terminal() || seen.count(next)) continue;
        seen.insert(next);
        reach[next] = at.second;
        frontier.push_back(std::make_pair(next, at.second));
      }
    }
  }

  ModelSearchPolicyData data;
  const std::string family = impl_->spec.get_family();
  std::mt19937 rng(seed);
  for (int measurement = 0; measurement < episodes; ++measurement) {
    const std::string& generating =
        keys[std::uniform_int_distribution<std::size_t>(0, keys.size() - 1)(rng)];
    const unsigned int simulation_seed = rng();
    try {
      const std::shared_ptr<FittingModelSearchProblem> fitted =
          simulate(generating, simulation_seed).build();
      // Every structure reachable from the root, each fitted once from its
      // declared starts -- what an exhaustive walk sees, so the label is
      // the structure selection picks on this data, not the one that made it.
      std::map<std::string, ModelSearchState> states;
      const ModelSearchState root = fitted->get_initial_state();
      states[root.get_structure_key()] = root;
      std::deque<std::string> frontier(1, root.get_structure_key());
      while (!frontier.empty()) {
        const std::string at = frontier.front();
        frontier.pop_front();
        const ModelSearchActions& out = moves[at];
        for (std::size_t i = 0; i < out.size(); ++i) {
          const std::string& next = out[i].get_predicted_state_key();
          if (out[i].get_terminal() || states.count(next)) continue;
          const ModelSearchState reached = fitted->evaluate(states[at], out[i]);
          if (reached.get_structure_key() != next) continue;  // did not converge
          states[next] = reached;
          frontier.push_back(next);
        }
      }
      std::string best;
      double best_reward = -std::numeric_limits<double>::infinity();
      for (const auto& entry : states) {
        if (entry.second.get_reward() > best_reward) {
          best_reward = entry.second.get_reward();
          best = entry.first;
        }
      }
      for (const auto& entry : states) {
        const std::string& source = entry.first;
        const ModelSearchActions& offered = moves[source];
        // One move is no decision; there is nothing in it to learn.
        if (offered.size() < 2) continue;
        int label = -1;
        if (source == best) {
          if (terminal.count(source)) label = terminal[source];
        } else if (first[source].count(best)) {
          label = first[source][best];
        }
        if (label < 0) continue;
        const std::vector<double> residual = fitted->get_cached_residual(entry.second.get_key());
        if (residual.empty()) continue;
        std::vector<double> priors;
        for (std::size_t i = 0; i < offered.size(); ++i) priors.push_back(offered[i].get_prior());
        data.add_episode(
            fitted->get_policy_rows(source, residual, offered,
                                    fitted->get_cached_residual_blocks(entry.second.get_key())),
            priors, label, family, source == best);
      }
    } catch (const std::exception&) {
      // A parameter draw the model cannot evaluate is not a measurement
      // anyone could have taken; skip it.
      continue;
    }
  }
  return data;
}

IMPBFF_END_NAMESPACE

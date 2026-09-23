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
// The vendored kernels are placed in this module's namespace, as
// NeuralNet.cpp places them -- one copy, one home, and the unity build
// cannot end up with two opinions about where mlpcore lives.
#define TTTRLIB_MLPCORE_NAMESPACE IMP::bff::internal
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
const int kPolicyProfile = 32;

bool is_log_scaled(double lower, double upper, double reference);

void sample_ports(const std::shared_ptr<MultiStructureModelSearchProblem>& problem,
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
  int policy_episodes = 0;
  int policy_features = 0;
  std::vector<double> policy_x;
  std::vector<double> policy_y;
  std::vector<std::string> policy_actions;
  double policy_loss = 0.0;
  int policy_validation_rows = 0;
  double policy_validation_loss = std::numeric_limits<double>::quiet_NaN();
  double policy_validation_accuracy = std::numeric_limits<double>::quiet_NaN();

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

void ModelSearchSelfPlay::generate_policy(int episodes, unsigned int seed) {
  if (episodes <= 0) throw std::domain_error("self play: no policy episodes to play");
  std::shared_ptr<MultiStructureModelSearchProblem> template_problem =
      impl_->spec.build();
  struct Transition {
    std::string source, action, target;
    std::vector<std::string> datasets;
  };
  std::vector<Transition> transitions;
  for (const std::string& source : template_problem->get_structure_keys()) {
    const std::vector<std::string> source_data =
        template_problem->get_structure_curve_datasets(source);
    ModelSearchState state(source, source, 0.0);
    const ModelSearchActions actions = template_problem->get_actions(state);
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (source_data.empty()) continue;
      // Stopping is a decision too. Its episode fits the right topology to
      // its own data, so the policy sees what an adequate residual looks like.
      if (actions[i].get_terminal()) {
        transitions.push_back({source, actions[i].get_key(), source, source_data});
        continue;
      }
      if (actions[i].get_predicted_state_key() == source) continue;
      const std::vector<std::string> target_data =
          template_problem->get_structure_curve_datasets(
              actions[i].get_predicted_state_key());
      if (source_data != target_data) continue;
      transitions.push_back(
          {source, actions[i].get_key(), actions[i].get_predicted_state_key(), source_data});
    }
  }
  if (transitions.empty()) {
    throw std::domain_error("self play: no compatible structural transitions to learn");
  }
  impl_->policy_episodes = 0;
  impl_->policy_features = kPolicyProfile;
  impl_->policy_x.clear();
  impl_->policy_y.clear();
  impl_->policy_actions.clear();
  std::set<std::string> seen_actions;
  for (std::size_t i = 0; i < transitions.size(); ++i) {
    if (seen_actions.insert(transitions[i].action).second) {
      impl_->policy_actions.push_back(transitions[i].action);
    }
  }
  std::map<std::string, int> action_index;
  // An action reachable from many parents must not dominate the labels, so
  // an episode draws its action uniformly and only then one of its sources.
  std::vector<std::vector<std::size_t> > by_action(impl_->policy_actions.size());
  for (std::size_t i = 0; i < impl_->policy_actions.size(); ++i) {
    action_index[impl_->policy_actions[i]] = static_cast<int>(i);
  }
  for (std::size_t i = 0; i < transitions.size(); ++i) {
    by_action[static_cast<std::size_t>(action_index[transitions[i].action])].push_back(i);
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pick_action(
      0, static_cast<int>(by_action.size()) - 1);
  for (int episode = 0; episode < episodes; ++episode) {
    const std::vector<std::size_t>& sources =
        by_action[static_cast<std::size_t>(pick_action(rng))];
    std::uniform_int_distribution<int> pick_source(
        0, static_cast<int>(sources.size()) - 1);
    const Transition& move =
        transitions[sources[static_cast<std::size_t>(pick_source(rng))]];
    std::shared_ptr<MultiStructureModelSearchProblem> generator = impl_->spec.build();
    generator->activate_structure(move.target);
    sample_ports(generator, impl_->spread, rng);
    ModelSearchSpec simulated(impl_->spec);
    for (std::size_t d = 0; d < move.datasets.size(); ++d) {
      FitDataset data = impl_->spec.get_dataset(move.datasets[d]);
      std::vector<double> values = generator->get_structure_output(
          move.target, generator->get_structure_curve_node(move.target, move.datasets[d]));
      add_matching_noise(data, values, impl_->poisson, rng);
      data.replace_values(values);
      simulated.set_dataset(move.datasets[d], data);
    }
    std::shared_ptr<MultiStructureModelSearchProblem> fitted = simulated.build();
    fitted->activate_structure(move.source);
    const int status = fitted->fit_active_structure();
    if (status < 1 || status > 4) continue;
    const std::vector<double> profile =
        get_residual_profile(fitted->get_active_residual(), kPolicyProfile);
    impl_->policy_x.insert(impl_->policy_x.end(), profile.begin(), profile.end());
    for (std::size_t a = 0; a < impl_->policy_actions.size(); ++a) {
      impl_->policy_y.push_back(
          static_cast<int>(a) == action_index[move.action] ? 1.0 : 0.0);
    }
    ++impl_->policy_episodes;
  }
  if (impl_->policy_episodes == 0) {
    throw std::domain_error("self play: no simulated parent fit converged");
  }
}

int ModelSearchSelfPlay::get_number_of_policy_episodes() const {
  return impl_->policy_episodes;
}
int ModelSearchSelfPlay::get_number_of_policy_features() const {
  return impl_->policy_features;
}
std::vector<std::string> ModelSearchSelfPlay::get_policy_action_keys() const {
  return impl_->policy_actions;
}
const std::vector<double>& ModelSearchSelfPlay::get_policy_features() const {
  return impl_->policy_x;
}
const std::vector<double>& ModelSearchSelfPlay::get_policy_targets() const {
  return impl_->policy_y;
}

namespace {

//! Mean cross-entropy and top-1 accuracy of softmax logits against one-hot rows.
std::pair<double, double> softmax_scores(const std::vector<double>& logits,
                                         const std::vector<double>& targets,
                                         int rows, int n_out) {
  double loss = 0.0;
  int correct = 0;
  for (int row = 0; row < rows; ++row) {
    const double* z = &logits[static_cast<std::size_t>(row) * n_out];
    const double* y = &targets[static_cast<std::size_t>(row) * n_out];
    const int best = static_cast<int>(std::max_element(z, z + n_out) - z);
    const double maximum = z[best];
    double total = 0.0;
    for (int col = 0; col < n_out; ++col) total += std::exp(z[col] - maximum);
    for (int col = 0; col < n_out; ++col) {
      if (y[col] > 0.0) {
        loss -= y[col] * (z[col] - maximum - std::log(total));
        if (col == best) ++correct;
      }
    }
  }
  return std::make_pair(loss / rows, static_cast<double>(correct) / rows);
}

}  // namespace

std::string ModelSearchSelfPlay::train_policy(
    const std::vector<int>& hidden, int epochs, double learning_rate,
    unsigned int seed, double validation_fraction) {
  if (impl_->policy_episodes <= 0 || epochs <= 0 || !(learning_rate > 0.0)) {
    throw std::domain_error("self play: policy needs episodes, epochs and learning rate");
  }
  if (!(validation_fraction >= 0.0) || !(validation_fraction < 1.0)) {
    throw std::domain_error("self play: validation fraction must lie in [0, 1)");
  }
  const int n_in = impl_->policy_features;
  const int n_out = static_cast<int>(impl_->policy_actions.size());
  std::mt19937 rng(seed);

  std::vector<int> order(static_cast<std::size_t>(impl_->policy_episodes));
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
  std::shuffle(order.begin(), order.end(), rng);
  const int n_valid = static_cast<int>(validation_fraction * impl_->policy_episodes);
  const int n_train = impl_->policy_episodes - n_valid;
  if (n_train <= 0) throw std::domain_error("self play: no policy episodes left to train on");
  std::vector<double> train_x, train_y, valid_x, valid_y;
  for (int k = 0; k < impl_->policy_episodes; ++k) {
    const std::size_t row = static_cast<std::size_t>(order[static_cast<std::size_t>(k)]);
    std::vector<double>& x = k < n_train ? train_x : valid_x;
    std::vector<double>& y = k < n_train ? train_y : valid_y;
    x.insert(x.end(), impl_->policy_x.begin() + row * n_in,
             impl_->policy_x.begin() + (row + 1) * n_in);
    y.insert(y.end(), impl_->policy_y.begin() + row * n_out,
             impl_->policy_y.begin() + (row + 1) * n_out);
  }

  internal::MlpModel model;
  int previous = n_in;
  std::vector<int> widths(hidden);
  widths.push_back(n_out);
  for (std::size_t l = 0; l < widths.size(); ++l) {
    internal::DenseLayer layer;
    layer.n_in = previous;
    layer.n_out = widths[l];
    layer.activation = l + 1 == widths.size() ? internal::Activation::Identity
                                                : internal::Activation::Tanh;
    const double limit = std::sqrt(6.0 / (layer.n_in + layer.n_out));
    std::uniform_real_distribution<double> init(-limit, limit);
    layer.weight.resize(static_cast<std::size_t>(layer.n_in) * layer.n_out);
    for (double& value : layer.weight) value = init(rng);
    layer.bias.assign(layer.n_out, 0.0);
    model.layers.push_back(layer);
    previous = layer.n_out;
  }
  model.x_scaler.mean.assign(n_in, 0.0);
  model.x_scaler.scale.assign(n_in, 1.0);
  for (int j = 0; j < n_in; ++j) {
    double sum = 0.0, variance = 0.0;
    for (int row = 0; row < n_train; ++row) sum += train_x[row * n_in + j];
    model.x_scaler.mean[j] = sum / n_train;
    for (int row = 0; row < n_train; ++row) {
      const double d = train_x[row * n_in + j] - model.x_scaler.mean[j];
      variance += d * d;
    }
    model.x_scaler.scale[j] = std::sqrt(variance / n_train) + 1e-12;
  }
  model.validate();
  std::vector<double> flat, prediction, dparams, dX, dV, gradient;
  internal::mlpcore::flatten(model.layers, flat);
  tttrlib::AdamState adam;
  adam.reset(flat.size());
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    std::vector<double> dy, d2y;
    internal::mlpcore::model_predict(model, train_x.data(), n_train, 0, nullptr,
                                     prediction, dy, d2y);
    gradient.assign(prediction.size(), 0.0);
    for (int row = 0; row < n_train; ++row) {
      double maximum = -std::numeric_limits<double>::infinity();
      for (int col = 0; col < n_out; ++col) maximum = std::max(
          maximum, prediction[row * n_out + col]);
      double total = 0.0;
      for (int col = 0; col < n_out; ++col) {
        gradient[row * n_out + col] = std::exp(prediction[row * n_out + col] - maximum);
        total += gradient[row * n_out + col];
      }
      for (int col = 0; col < n_out; ++col) {
        gradient[row * n_out + col] =
            (gradient[row * n_out + col] / total - train_y[row * n_out + col]) / n_train;
      }
    }
    internal::mlpcore::model_backward(model, train_x.data(), n_train, nullptr,
                                      gradient.data(), nullptr, nullptr, dparams, dX, dV);
    tttrlib::adam_update(flat.data(), dparams.data(), flat.size(), adam, learning_rate);
    internal::mlpcore::unflatten(model.layers, flat.data(), flat.size());
  }
  std::vector<double> dy, d2y;
  internal::mlpcore::model_predict(model, train_x.data(), n_train, 0, nullptr,
                                   prediction, dy, d2y);
  impl_->policy_loss = softmax_scores(prediction, train_y, n_train, n_out).first;
  impl_->policy_validation_rows = n_valid;
  impl_->policy_validation_loss = std::numeric_limits<double>::quiet_NaN();
  impl_->policy_validation_accuracy = std::numeric_limits<double>::quiet_NaN();
  if (n_valid > 0) {
    internal::mlpcore::model_predict(model, valid_x.data(), n_valid, 0, nullptr,
                                     prediction, dy, d2y);
    const std::pair<double, double> scores =
        softmax_scores(prediction, valid_y, n_valid, n_out);
    impl_->policy_validation_loss = scores.first;
    impl_->policy_validation_accuracy = scores.second;
  }
  return internal::mlpcore::model_to_json<nlohmann::json>(model).dump();
}

double ModelSearchSelfPlay::get_policy_training_loss() const {
  return impl_->policy_loss;
}

int ModelSearchSelfPlay::get_number_of_policy_validation_episodes() const {
  return impl_->policy_validation_rows;
}
double ModelSearchSelfPlay::get_policy_validation_loss() const {
  return impl_->policy_validation_loss;
}
double ModelSearchSelfPlay::get_policy_validation_accuracy() const {
  return impl_->policy_validation_accuracy;
}

IMPBFF_END_NAMESPACE

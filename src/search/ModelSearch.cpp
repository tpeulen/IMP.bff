/** \file IMP/bff/ModelSearch.cpp */
#include <IMP/bff/ModelSearch.h>

#include <IMP/bff/SpecialFunctions.h>
#include <IMP/bff/FitMinimizer.h>
#include <IMP/bff/FitJointChiSquared.h>
#include <IMP/bff/FitObjective.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/internal/NetworkDocument.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <utility>

IMPBFF_BEGIN_NAMESPACE

namespace {

void require_key(const std::string& key, const char* what) {
  if (key.empty()) {
    throw ModelSearchConfigurationError(std::string(what) +
                                           " key must not be empty");
  }
}

void require_state(const ModelSearchState& state, const char* what) {
  require_key(state.get_key(), what);
  require_key(state.get_structure_key(), "state structure");
  if (!std::isfinite(state.get_reward())) {
    throw ModelSearchConfigurationError(std::string(what) +
                                           " reward must be finite");
  }
}

struct TreeNode {
  explicit TreeNode(double p = 1.0, TreeNode* up = nullptr)
      : prior(p), parent(up) {}

  ModelSearchState state;
  ModelSearchAction parent_action;
  double prior = 1.0;
  std::vector<std::unique_ptr<TreeNode> > children;
  int visits = 0;
  double value_sum = 0.0;
  bool expanded = false;
  bool evaluated = false;
  bool terminal = false;
  TreeNode* parent = nullptr;
  std::string structure_key;
};

double reward_value(const ModelSearchState& state, double root_reward,
                    double scale) {
  return std::tanh((state.get_reward() - root_reward) / scale);
}

double puct(const TreeNode& child, int parent_visits, double c_puct) {
  const double q = child.value_sum / (child.visits + 1.0);
  const double u = c_puct * child.prior *
                   std::sqrt(static_cast<double>(parent_visits)) /
                   (1.0 + child.visits);
  return q + u;
}

//! A network scoring (state, action) feature rows; one logit per row.
class ActionPolicy {
 public:
  void configure(const MsgpackBytes& network, double temperature) {
    if (network.empty()) {
      net_.reset();
      return;
    }
    if (!std::isfinite(temperature) || temperature < 0.0) {
      throw ModelSearchConfigurationError("an action policy's temperature must be positive");
    }
    // Decoded here for the temperature (NeuralNet below reads the layers); a
    // malformed document is an IMP::ValueException naming what is wrong.
    const nlohmann::json doc =
        internal::document_from_msgpack(network, "bff.neural_net", "set_action_policy");
    if (temperature == 0.0) {
      // The document's own: a policy is gated at the temperature it ships
      // with, and a different one is a different, ungated policy.
      temperature = 1.0;
      if (doc.contains("temperature")) {
        if (!doc.at("temperature").is_number()) {
          throw ModelSearchConfigurationError(
              "an action policy document's temperature must be a number");
        }
        temperature = doc.at("temperature").get<double>();
      }
      if (!(temperature > 0.0) || !std::isfinite(temperature)) {
        throw ModelSearchConfigurationError(
            "an action policy document's temperature must be positive");
      }
    }
    temperature_ = temperature;
    std::shared_ptr<NeuralNet> net(new NeuralNet(network));
    const int width = get_policy_state_width() + get_policy_action_width();
    if (net->get_n_inputs() != width || net->get_n_outputs() != 1) {
      std::ostringstream message;
      message << "an action policy scores one row of " << width
              << " features with one output; this network takes "
              << net->get_n_inputs() << " and gives " << net->get_n_outputs();
      throw ModelSearchConfigurationError(message.str());
    }
    net_ = net;
  }

  void clear() { net_.reset(); }
  bool active() const { return static_cast<bool>(net_); }

  //! Declared prior times exp(logit), normalised over the available actions.
  /*! A product rather than a replacement: a move a family declares at zero
      stays at zero, and the network only reweights what the family offers. */
  ModelSearchActions apply(const ModelSearchActions& actions,
                           const std::vector<double>& rows) const {
    if (!net_ || actions.empty()) return actions;
    double* view = nullptr;
    int n_out = 0;
    net_->predict(rows, static_cast<int>(actions.size()), &view, &n_out);
    std::vector<double> logits(view, view + n_out);
    std::free(view);
    if (logits.size() != actions.size()) {
      throw ModelSearchConfigurationError("action policy returned the wrong number of scores");
    }
    double declared_total = 0.0;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      declared_total += std::max(0.0, actions[i].get_prior());
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
      if (!std::isfinite(logits[i])) {
        throw ModelSearchConfigurationError("action policy output is not finite");
      }
      maximum = std::max(maximum, logits[i]);
    }
    std::vector<double> weight(actions.size());
    double total = 0.0;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      const double declared = declared_total > 0.0
          ? std::max(0.0, actions[i].get_prior()) / declared_total
          : 1.0 / static_cast<double>(actions.size());
      weight[i] = declared * std::exp((logits[i] - maximum) / temperature_);
      total += weight[i];
    }
    ModelSearchActions adjusted;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      const double prior = total > 0.0 ? weight[i] / total : 0.0;
      adjusted.push_back(ModelSearchAction(actions[i].get_key(),
                                           actions[i].get_predicted_state_key(),
                                           prior, actions[i].get_terminal()));
    }
    return adjusted;
  }

 private:
  std::shared_ptr<NeuralNet> net_;
  double temperature_ = 1.0;
};

const int kPolicyResidualBins = 32;
const int kPolicyStateStatistics = 7;
const int kPolicyActionWidth = 7;

}  // namespace

std::vector<double> get_residual_profile(const std::vector<double>& residual,
                                         int width) {
  if (width <= 0) {
    throw ModelSearchConfigurationError("residual profile width must be positive");
  }
  std::vector<double> profile(static_cast<std::size_t>(width), 0.0);
  if (residual.empty()) return profile;
  for (int bucket = 0; bucket < width; ++bucket) {
    const std::size_t begin = static_cast<std::size_t>(bucket) * residual.size() /
                              static_cast<std::size_t>(width);
    const std::size_t end = static_cast<std::size_t>(bucket + 1) * residual.size() /
                            static_cast<std::size_t>(width);
    double sum = 0.0;
    int count = 0;
    for (std::size_t i = begin; i < end; ++i) {
      if (std::isfinite(residual[i])) {
        sum += residual[i];
        ++count;
      }
    }
    // A bucket's z-score, compressed: independent of how many points a curve
    // has, so one network reads a 64-point FCS curve and a 4096-channel decay.
    if (count) profile[static_cast<std::size_t>(bucket)] = std::asinh(sum / std::sqrt(count));
  }
  return profile;
}

int get_policy_state_width() { return 2 * kPolicyResidualBins + kPolicyStateStatistics; }

int get_policy_action_width() { return kPolicyActionWidth; }

std::vector<double> get_policy_state_features(const std::vector<double>& residual,
                                              int n_free,
                                              const std::vector<int>& block_sizes) {
  std::vector<double> features = get_residual_profile(residual, kPolicyResidualBins);
  // The worst-fitting block of a joint residual, on its own: laid end to end
  // the members' scales and lengths blur which measurement the model misses.
  std::size_t worst_begin = 0, worst_end = residual.size();
  int n_blocks = 1;
  std::size_t total = 0;
  for (int size : block_sizes) total += static_cast<std::size_t>(std::max(0, size));
  if (block_sizes.size() > 1 && total == residual.size()) {
    n_blocks = static_cast<int>(block_sizes.size());
    double worst = -1.0;
    std::size_t at = 0;
    for (int size : block_sizes) {
      const std::size_t end = at + static_cast<std::size_t>(std::max(0, size));
      double sum = 0.0;
      int count = 0;
      for (std::size_t i = at; i < end; ++i) {
        if (std::isfinite(residual[i])) {
          sum += residual[i] * residual[i];
          ++count;
        }
      }
      const double mean = count ? sum / count : 0.0;
      if (mean > worst) {
        worst = mean;
        worst_begin = at;
        worst_end = end;
      }
      at = end;
    }
  }
  const std::vector<double> block(residual.begin() + worst_begin, residual.begin() + worst_end);
  const std::vector<double> block_profile = get_residual_profile(block, kPolicyResidualBins);
  features.insert(features.end(), block_profile.begin(), block_profile.end());
  double block_chi2 = 0.0;
  for (double r : block) {
    if (std::isfinite(r)) block_chi2 += r * r;
  }
  double chi2 = 0.0, lag = 0.0;
  int n = 0, positive = 0, runs = 0, n_pos = 0, n_neg = 0;
  double previous = 0.0;
  bool has_previous = false;
  for (std::size_t i = 0; i < residual.size(); ++i) {
    const double r = residual[i];
    if (!std::isfinite(r)) continue;
    chi2 += r * r;
    if (has_previous) lag += r * previous;
    if (r > 0.0) ++positive;
    const int sign = r > 0.0 ? 1 : -1;
    if (sign > 0) ++n_pos; else ++n_neg;
    if (!has_previous || (previous > 0.0 ? 1 : -1) != sign) ++runs;
    previous = r;
    has_previous = true;
    ++n;
  }
  const double dof = std::max(1.0, static_cast<double>(n - n_free));
  features.push_back(std::log10(std::max(chi2 / dof, 1e-12)));
  features.push_back(chi2 > 0.0 ? lag / chi2 : 0.0);
  // Wald-Wolfowitz: too few sign runs is structure the model has not caught.
  double runs_z = 0.0;
  if (n_pos > 0 && n_neg > 0) {
    const double np = n_pos, nn = n_neg, total = np + nn;
    const double expected = 2.0 * np * nn / total + 1.0;
    const double variance =
        2.0 * np * nn * (2.0 * np * nn - total) / (total * total * (total - 1.0));
    if (variance > 0.0) runs_z = (runs - expected) / std::sqrt(variance);
  }
  features.push_back(std::asinh(runs_z));
  features.push_back(n ? static_cast<double>(positive) / n : 0.5);
  features.push_back(std::log1p(static_cast<double>(std::max(0, n_free))));
  features.push_back(std::log10(std::max(block_chi2 / std::max<double>(1.0, block.size()), 1e-12)));
  features.push_back(std::log(static_cast<double>(n_blocks)));
  return features;
}

std::vector<double> get_policy_action_features(int source_free, int target_free,
                                               bool terminal, bool self_loop,
                                               double prior_share) {
  const int delta = target_free - source_free;
  std::vector<double> features;
  features.push_back(terminal ? 1.0 : 0.0);
  features.push_back(self_loop ? 1.0 : 0.0);
  features.push_back(std::max(-5, std::min(5, delta)) / 5.0);
  features.push_back(delta > 0 ? 1.0 : 0.0);
  features.push_back(delta < 0 ? 1.0 : 0.0);
  features.push_back(std::log1p(static_cast<double>(std::max(0, target_free))));
  features.push_back(std::log(std::max(prior_share, 1e-6)));
  return features;
}

ModelSearchState::ModelSearchState()
    : reward_(0.0), acceptable_(false) {}

ModelSearchState::ModelSearchState(const std::string& key, double reward,
                                         bool acceptable)
    : key_(key), structure_key_(key), reward_(reward),
      acceptable_(acceptable) {}

ModelSearchState::ModelSearchState(
    const std::string& key, const std::string& structure_key, double reward,
    bool acceptable)
    : key_(key), structure_key_(structure_key), reward_(reward),
      acceptable_(acceptable) {}

const std::string& ModelSearchState::get_key() const { return key_; }
const std::string& ModelSearchState::get_structure_key() const {
  return structure_key_;
}
double ModelSearchState::get_reward() const { return reward_; }
bool ModelSearchState::get_acceptable() const { return acceptable_; }

ModelSearchAction::ModelSearchAction()
    : prior_(1.0), terminal_(false) {}

ModelSearchAction::ModelSearchAction(
    const std::string& key, const std::string& predicted_state_key,
    double prior, bool terminal)
    : key_(key), predicted_state_key_(predicted_state_key), prior_(prior),
      terminal_(terminal) {}

const std::string& ModelSearchAction::get_key() const { return key_; }
const std::string& ModelSearchAction::get_predicted_state_key() const {
  return predicted_state_key_;
}
double ModelSearchAction::get_prior() const { return prior_; }
bool ModelSearchAction::get_terminal() const { return terminal_; }

ModelSearchProblem::~ModelSearchProblem() {}
void ModelSearchProblem::request_cancel() {}
void ModelSearchProblem::clear_cancel() {}
void ModelSearchProblem::activate_state(const ModelSearchState&) {}

struct TabularModelSearchProblem::Impl {
  std::map<std::string, ModelSearchState> states;
  std::map<std::string, ModelSearchActions> actions;
  std::string initial_key;
  int evaluations = 0;
};

TabularModelSearchProblem::TabularModelSearchProblem()
    : impl_(new Impl) {}

void TabularModelSearchProblem::add_state(const std::string& key,
                                           double reward,
                                           bool acceptable) {
  require_key(key, "state");
  if (!std::isfinite(reward)) {
    throw ModelSearchConfigurationError("state reward must be finite");
  }
  impl_->states[key] = ModelSearchState(key, reward, acceptable);
}

void TabularModelSearchProblem::set_initial_state(const std::string& key) {
  require_key(key, "initial state");
  impl_->initial_key = key;
}

void TabularModelSearchProblem::add_action(
    const std::string& parent_key, const std::string& action_key,
    const std::string& result_key, double prior, bool terminal) {
  require_key(parent_key, "parent state");
  require_key(action_key, "action");
  require_key(result_key, "result state");
  if (!std::isfinite(prior) || prior < 0.0) {
    throw ModelSearchConfigurationError(
        "action prior must be finite and non-negative");
  }
  ModelSearchActions& actions = impl_->actions[parent_key];
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (actions[i].get_key() == action_key) {
      throw ModelSearchConfigurationError(
          "action keys must be unique within a state");
    }
  }
  actions.push_back(
      ModelSearchAction(action_key, result_key, prior, terminal));
}

ModelSearchState TabularModelSearchProblem::get_initial_state() {
  const std::map<std::string, ModelSearchState>::const_iterator found =
      impl_->states.find(impl_->initial_key);
  if (found == impl_->states.end()) {
    throw ModelSearchConfigurationError(
        "initial state has not been added to the graph problem");
  }
  return found->second;
}

ModelSearchActions TabularModelSearchProblem::get_actions(
    const ModelSearchState& state) {
  const std::map<std::string, ModelSearchActions>::const_iterator
      found = impl_->actions.find(state.get_key());
  if (found == impl_->actions.end()) return ModelSearchActions();
  return found->second;
}

ModelSearchState TabularModelSearchProblem::evaluate(
    const ModelSearchState&, const ModelSearchAction& action) {
  const std::map<std::string, ModelSearchState>::const_iterator found =
      impl_->states.find(action.get_predicted_state_key());
  if (found == impl_->states.end()) {
    throw ModelSearchConfigurationError(
        "action result state has not been added to the graph problem");
  }
  ++impl_->evaluations;
  return found->second;
}

int TabularModelSearchProblem::get_number_of_evaluations() const {
  return impl_->evaluations;
}

void TabularModelSearchProblem::reset_number_of_evaluations() {
  impl_->evaluations = 0;
}

namespace {

class FitSearchCancelObserver : public FitMinimizerObserver {
 public:
  explicit FitSearchCancelObserver(std::atomic<bool>* cancelled)
      : FitMinimizerObserver("FitSearchCancelObserver%1%"),
        cancelled_(cancelled) {}
  bool report(int, int, double) override { return !cancelled_->load(); }
  IMP_OBJECT_METHODS(FitSearchCancelObserver);

 private:
  std::atomic<bool>* cancelled_;
};

struct FitSearchSnapshot {
  std::vector<double> values;
  std::vector<int> fixed;
  //! The weighted residual of the fitted state, read once when it was scored.
  std::vector<double> residual;
  //! How long each member's block of that residual is; empty for one member.
  std::vector<int> blocks;
};

}  // namespace

namespace {

//! Copies one port of whichever topology is current to a port that stays put.
class OutputRelay : public GraphNode {
 public:
  explicit OutputRelay(const std::string& name) : GraphNode(name) {}
  void evaluate() override {
    const std::shared_ptr<GraphPort> in = get_input_port("in");
    const std::shared_ptr<GraphPort> out = get_output_port("out");
    if (in && out && in->is_linked()) {
      out->set_value_vector(in->get_link_ref()->get_values_ref());
    }
    set_valid(true);
  }
  std::string get_node_type() const override { return "OutputRelay"; }
};

struct PublishedOutput {
  std::string node;
  std::string port;
  std::shared_ptr<GraphNode> relay;
  //! The port the relay currently follows; compared, never dereferenced.
  const GraphPort* source = nullptr;
};

struct StructureRecord {
  std::shared_ptr<FitObjective> objective;
  std::vector<std::shared_ptr<GraphNode> > graph_nodes;
  std::string score_output;
  std::string acceptable_output;
  std::vector<double> initial_values;
  //! Further declared starting points, tried alongside initial_values.
  /*! A single seed cannot be trusted to reach the best fit a topology
      admits -- measured on a two-species FCS curve, where the generating
      model reaches -915 from its declared seed and -12.6 from a good one.
      Every start is declared, so the best of them is still a property of
      the model and the data rather than of the route taken. */
  std::vector<std::vector<double> > extra_starts;
  std::vector<int> fixed;
  double effective_sample_size = 0.0;
  double complexity = 0.0;
  bool use_bic = false;
  ModelSelectionCriterion criterion = MODEL_SELECTION_BIC;
  //! Least goodness-of-fit probability this structure may have and still be
  //! reported as describing the data.
  double acceptance = 0.0;
  bool has_acceptance = false;
  //! Per registry index: 1 when this topology's graph reads the parameter.
  /*! Empty means undeclared, and then only what the topology frees counts
      as used -- a release cannot free a port no node reads. */
  std::vector<int> uses;
  //! measurement name -> the node whose curve is compared against it.
  std::map<std::string, std::string> curves;
  std::vector<std::string> curve_order;
};

}  // namespace

struct FittingModelSearchProblem::Impl {
  std::map<std::string, std::shared_ptr<GraphPort> > parameters;
  std::vector<std::string> parameter_order;
  std::map<std::string, StructureRecord> structures;
  std::vector<std::string> structure_order;
  std::map<std::string, ModelSearchActions> actions;
  ActionPolicy action_policy;
  std::string initial_structure;
  std::string active_structure;
  double last_reduced_chi2 = 0.0;
  double last_chi2_p_value = 0.0;
  std::map<std::string, FitSearchSnapshot> snapshots;
  std::map<std::string, std::string> snapshot_structures;
  unsigned long long next_snapshot = 1;
  //! See FittingModelSearchProblem::set_warm_start; off by default so
  //! a candidate's score belongs to the candidate.
  bool warm_start = false;
  int maxfev = 0;
  //! Canonical ids a user holds; see set_parameter_locked.
  std::set<std::string> locked;
  //! Canonical ids a user frees where a topology reads them; see
  //! set_parameter_released.
  std::set<std::string> released;
  std::atomic<bool> cancelled;
  //! See FittingModelSearchProblem::publish_output.
  std::map<std::string, PublishedOutput> outputs;
  int last_status = 0;
  std::string last_failure;

  Impl() : cancelled(false) {}

  void validate_registry() const {
    std::set<GraphPort*> owners;
    if (parameter_order.size() != parameters.size()) {
      throw ModelSearchConfigurationError(
          "canonical parameter registry order is inconsistent");
    }
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const std::map<std::string, std::shared_ptr<GraphPort> >::const_iterator
          found = parameters.find(parameter_order[i]);
      if (found == parameters.end() || !found->second) {
        throw ModelSearchConfigurationError(
            "canonical parameter registry contains a missing owner");
      }
      // A follower reads through its link, so the scalar is the one it follows.
      GraphPort* source = found->second.get();
      while (source->get_link()) source = source->get_link().get();
      if (source->get_is_vector()) {
        throw ModelSearchConfigurationError(
            "canonical parameters must remain scalar ports");
      }
      if (!owners.insert(found->second.get()).second) {
        throw ModelSearchConfigurationError(
            "canonical parameter registry contains a duplicate owner");
      }
    }
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      GraphPort* followed = parameters.find(parameter_order[i])->second->get_link().get();
      while (followed) {
        if (owners.count(followed)) {
          throw ModelSearchConfigurationError(
              "canonical parameter '" + parameter_order[i] +
              "' follows another parameter of the same model");
        }
        followed = followed->get_link().get();
      }
    }
  }

  const StructureRecord& structure(const std::string& key) const {
    const std::map<std::string, StructureRecord>::const_iterator found =
        structures.find(key);
    if (found == structures.end()) {
      throw ModelSearchConfigurationError("unknown model structure '" + key +
                                           "'");
    }
    return found->second;
  }

  StructureRecord& structure(const std::string& key) {
    const std::map<std::string, StructureRecord>::iterator found =
        structures.find(key);
    if (found == structures.end()) {
      throw ModelSearchConfigurationError("unknown model structure '" + key +
                                           "'");
    }
    return found->second;
  }

  //! The structure's weighted residual as its objective last evaluated it.
  std::vector<double> residual_of(const StructureRecord& record) const {
    return record.objective->get_residuals();
  }

  std::vector<int> blocks_of(const StructureRecord& record) const {
    const std::shared_ptr<FitJointChiSquared> joint =
        std::dynamic_pointer_cast<FitJointChiSquared>(record.objective);
    return joint ? joint->get_block_sizes() : std::vector<int>();
  }

  FitSearchSnapshot capture() const {
    FitSearchSnapshot snapshot;
    snapshot.values.reserve(parameter_order.size());
    snapshot.fixed.reserve(parameter_order.size());
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const std::shared_ptr<GraphPort>& port =
          parameters.find(parameter_order[i])->second;
      snapshot.values.push_back(port->get_value());
      snapshot.fixed.push_back(port->get_fixed() ? 1 : 0);
    }
    return snapshot;
  }

  void restore_values(const FitSearchSnapshot& snapshot) {
    if (snapshot.values.size() != parameter_order.size() ||
        snapshot.fixed.size() != parameter_order.size()) {
      throw ModelSearchConfigurationError("invalid cached model snapshot");
    }
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const std::shared_ptr<GraphPort>& port =
          parameters.find(parameter_order[i])->second;
      port->set_value(snapshot.values[i]);
      port->set_fixed(snapshot.fixed[i] != 0);
    }
  }

  void select_and_update(const std::string& structure_key) {
    StructureRecord& selected = structure(structure_key);
    active_structure = structure_key;
    relink_outputs();
    selected.objective->update();
  }

  //! Point every published output at the current topology's port.
  void relink_outputs() {
    if (outputs.empty() || active_structure.empty()) return;
    const StructureRecord& selected = structure(active_structure);
    for (std::map<std::string, PublishedOutput>::iterator it = outputs.begin();
         it != outputs.end(); ++it) {
      PublishedOutput& published = it->second;
      const std::string wanted = active_structure + "." + published.node;
      std::shared_ptr<GraphPort> source;
      std::vector<std::shared_ptr<GraphNode> > candidates = selected.graph_nodes;
      candidates.push_back(selected.objective);
      for (std::size_t i = 0; i < candidates.size() && !source; ++i) {
        if (candidates[i] && (candidates[i]->get_name() == wanted ||
                              candidates[i]->get_name() == published.node)) {
          // `@name`: the output a node writes under its own name.
          source = candidates[i]->get_output_port(
              published.port == "@name" ? candidates[i]->get_name() : published.port);
        }
      }
      if (source.get() == published.source) continue;
      const std::shared_ptr<GraphPort> in = published.relay->get_input_port("in");
      if (source) {
        in->set_link(source);
      } else {
        in->unlink();
      }
      published.relay->set_valid(false);
      published.source = source.get();
    }
  }

  void restore(const FitSearchSnapshot& snapshot,
               const std::string& structure_key) {
    restore_values(snapshot);
    select_and_update(structure_key);
  }

  //! Move a free port strictly inside its bounds before it is fitted.
  /*!
      The minimiser works in chisurf's transformed coordinate, and a start
      exactly on a bound is that transform's degenerate point: for a two-sided
      bound `lower + (upper - lower)/2 * (sin(xi) + 1)` it is `xi = -pi/2`,
      where the derivative is zero, and for a narrow box the finite-difference
      step lands in that flat spot and Levenberg-Marquardt stops without
      moving anything. Measured on a ChiSurf lifetime fit: a free scatter
      fraction starting at 0 in [0, 1] froze every parameter of the fit.

      Ordinary fitting starts wherever the user put a parameter and keeps
      leastsqbound's behaviour. Here the engine chooses the start, so an edge
      start is the engine's defect. The nudge is a millionth of the range and
      is applied only to ports about to be fitted, immediately before the
      fit -- a structure activated at declared values is left exactly there,
      and the optimiser remains free to return to the bound.
  */
  static void keep_inside(const std::shared_ptr<GraphPort>& port) {
    if (!port->get_is_bounded()) return;
    const double lower = port->get_lower_bound();
    const double upper = port->get_upper_bound();
    if (!(std::isfinite(lower) && std::isfinite(upper) && upper > lower)) return;
    const double margin = 1.0e-6 * (upper - lower);
    const double value = port->get_value();
    if (value <= lower) {
      port->set_value(lower + margin);
    } else if (value >= upper) {
      port->set_value(upper - margin);
    }
  }

  void apply_initial(const StructureRecord& target,
                     const std::vector<double>& seeds) {
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const std::shared_ptr<GraphPort>& port =
          parameters.find(parameter_order[i])->second;
      if (held(i)) {
        // Held by the user: neither freed nor re-seeded.
        port->set_fixed(true);
        continue;
      }
      if (!released.count(parameter_order[i])) port->set_value(seeds[i]);
      port->set_fixed(!is_free(target, i));
    }
  }

  bool uses(const StructureRecord& record, std::size_t i) const {
    return record.uses.empty() ? record.fixed[i] == 0 : record.uses[i] != 0;
  }

  //! Whether a topology fits one parameter, after the user has had a say.
  //! Held by the user: locked, or following a port of another model.
  bool held(std::size_t i) const {
    const std::string& id = parameter_order[i];
    return locked.count(id) != 0 || parameters.find(id)->second->get_link();
  }

  bool is_free(const StructureRecord& record, std::size_t i) const {
    const std::string& id = parameter_order[i];
    if (held(i)) return false;
    if (released.count(id) && uses(record, i)) return true;
    return record.fixed[i] == 0;
  }

  //! How many parameters the user's locks and releases add to a topology.
  double user_complexity_change(const StructureRecord& record) const {
    double change = 0.0;
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const bool declared = record.fixed[i] == 0;
      const bool actual = is_free(record, i);
      if (declared && !actual) change -= 1.0;
      if (!declared && actual) change += 1.0;
    }
    return change;
  }

  std::vector<std::shared_ptr<GraphPort> > apply_transition(
      const StructureRecord& parent,
      const StructureRecord& target,
      const std::vector<double>& seeds) {
    std::vector<std::shared_ptr<GraphPort> > free_ports;
    for (std::size_t i = 0; i < parameter_order.size(); ++i) {
      const std::shared_ptr<GraphPort>& port =
          parameters.find(parameter_order[i])->second;
      if (held(i)) {
        port->set_fixed(true);
        continue;
      }
      const bool was_free = this->is_free(parent, i);
      const bool is_free = this->is_free(target, i);
      // Without warm starting every structure begins from its declared
      // seeds, so its score does not depend on the route that reached it.
      if ((!warm_start || !is_free || !was_free) &&
          !released.count(parameter_order[i])) {
        port->set_value(seeds[i]);
      }
      port->set_fixed(!is_free);
      if (is_free) free_ports.push_back(port);
    }
    return free_ports;
  }

  //! The declared starting points for a structure, primary first.
  std::vector<std::vector<double> > starts_for(
      const StructureRecord& record) const {
    std::vector<std::vector<double> > starts;
    starts.push_back(record.initial_values);
    for (std::size_t i = 0; i < record.extra_starts.size(); ++i) {
      starts.push_back(record.extra_starts[i]);
    }
    return starts;
  }

  std::pair<double, bool> score_current(
      const StructureRecord& selected) {
    selected.objective->update();
    double reward = 0.0;
    bool acceptable_from_fit = false;
    bool has_acceptable_from_fit = false;
    if (!selected.score_output.empty()) {
      const std::shared_ptr<GraphPort> score =
          selected.objective->get_output_port(selected.score_output);
      if (!score) {
        throw ModelSearchConfigurationError(
            "objective has no score output '" + selected.score_output + "'");
      }
      reward = score->get_value();
    } else {
      const std::vector<double>& values = selected.objective->get_residuals();
      double chi2 = 0.0;
      for (std::size_t i = 0; i < values.size(); ++i) {
        chi2 += values[i] * values[i];
      }
      // A comparison between models, not a preference for the larger one.
      if (!selected.use_bic) {
        throw ModelSearchConfigurationError(
            "structure '" + active_structure +
            "' says nothing about how it should be compared with the others; "
            "declare a selection criterion or a score output, because "
            "ranking by misfit alone always prefers the richer model");
      }
      // A locked parameter is not estimated, so it is not paid for.
      const double complexity = std::max(
          0.0, selected.complexity + user_complexity_change(selected));
      const double penalty =
          selected.criterion == MODEL_SELECTION_AIC
              ? complexity
              : 0.5 * complexity * std::log(selected.effective_sample_size);
      reward = -0.5 * chi2 - penalty;
      // Goodness of fit is a different question from model choice, and a
      // model can win its family while describing the data badly.
      const double dof =
          selected.effective_sample_size - complexity - 1.0;
      last_reduced_chi2 = dof > 0.0 ? chi2 / dof
                                    : std::numeric_limits<double>::infinity();
      last_chi2_p_value = chi2_p_value(chi2, dof);
      if (selected.has_acceptance) {
        // Accepting on the test rather than on the ranking: a topology can
        // be the best of its family and still be refused here.
        acceptable_from_fit = last_chi2_p_value >= selected.acceptance;
        has_acceptable_from_fit = true;
      }
    }
    if (!std::isfinite(reward)) {
      throw ModelSearchConfigurationError("model-search reward is not finite");
    }
    bool acceptable = has_acceptable_from_fit ? acceptable_from_fit : false;
    if (!selected.acceptable_output.empty()) {
      const std::shared_ptr<GraphPort> port =
          selected.objective->get_output_port(selected.acceptable_output);
      if (!port) {
        throw ModelSearchConfigurationError(
            "objective has no acceptable output '" +
            selected.acceptable_output + "'");
      }
      acceptable = port->get_value_bool();
    }
    return std::make_pair(reward, acceptable);
  }

  void rollback(const FitSearchSnapshot& snapshot,
                const std::string& structure_key) {
    restore_values(snapshot);
    active_structure = structure_key;
    try {
      structure(structure_key).objective->update();
    } catch (...) {
      // Preserve the original transition failure. Values and masks are the
      // transaction boundary; a subsequent explicit restore will re-evaluate.
    }
  }
};

FittingModelSearchProblem::FittingModelSearchProblem()
    : impl_(new Impl) {}

FittingModelSearchProblem::~FittingModelSearchProblem() {}

void FittingModelSearchProblem::add_parameter(
    const std::string& canonical_id, std::shared_ptr<GraphPort> owner) {
  require_key(canonical_id, "canonical parameter");
  if (!impl_->structures.empty()) {
    throw ModelSearchConfigurationError(
        "canonical parameters must be registered before structures");
  }
  if (impl_->parameters.count(canonical_id)) {
    throw ModelSearchConfigurationError("duplicate canonical parameter id '" +
                                         canonical_id + "'");
  }
  if (!owner) {
    throw ModelSearchConfigurationError("canonical parameter is null");
  }
  {
    // A follower reads through its link, so the scalar is the one it follows.
    const GraphPort* source = owner.get();
    while (source->get_link()) source = source->get_link().get();
    if (source->get_is_vector()) {
      throw ModelSearchConfigurationError(
          "model search supports scalar canonical parameters only");
    }
  }
  // A canonical parameter may follow a port *outside* this registry -- a user
  // linking a lifetime across two fits. It is then held like a locked one;
  // following another canonical parameter of the same model is refused when
  // the registry is validated, since the two ids would be one parameter.
  for (std::map<std::string, std::shared_ptr<GraphPort> >::const_iterator it =
           impl_->parameters.begin();
       it != impl_->parameters.end(); ++it) {
    if (it->second.get() == owner.get()) {
      throw ModelSearchConfigurationError(
          "a canonical parameter port cannot have more than one id");
    }
  }
  impl_->parameters[canonical_id] = owner;
  impl_->parameter_order.push_back(canonical_id);
}

std::vector<std::string>
FittingModelSearchProblem::get_parameter_ids() const {
  return impl_->parameter_order;
}

std::shared_ptr<GraphPort> FittingModelSearchProblem::get_parameter(
    const std::string& canonical_id) const {
  const std::map<std::string, std::shared_ptr<GraphPort> >::const_iterator found =
      impl_->parameters.find(canonical_id);
  if (found == impl_->parameters.end()) return std::shared_ptr<GraphPort>();
  return found->second;
}

void FittingModelSearchProblem::add_structure(
    const std::string& key, std::shared_ptr<FitObjective> objective,
    const std::vector<std::string>& parameter_ids,
    const std::vector<std::shared_ptr<GraphPort> >& parameter_ports,
    const std::vector<double>& initial_values,
    const std::vector<int>& fixed_mask) {
  require_key(key, "model structure");
  if (impl_->structures.count(key)) {
    throw ModelSearchConfigurationError("duplicate model structure '" + key +
                                         "'");
  }
  if (!objective) {
    throw ModelSearchConfigurationError("model structure objective is null");
  }
  impl_->validate_registry();
  const std::size_t count = impl_->parameter_order.size();
  if (parameter_ids.size() != count || parameter_ports.size() != count ||
      initial_values.size() != count || fixed_mask.size() != count) {
    throw ModelSearchConfigurationError(
        "structure parameter state must cover the complete canonical registry");
  }
  std::set<std::string> seen_ids;
  std::set<GraphPort*> seen_ports;
  StructureRecord record;
  record.objective = std::move(objective);
  record.objective->get_residuals_port();
  record.initial_values.resize(count);
  record.fixed.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    require_key(parameter_ids[i], "structure parameter");
    if (!seen_ids.insert(parameter_ids[i]).second) {
      throw ModelSearchConfigurationError(
          "duplicate canonical parameter id in structure '" + key + "'");
    }
    if (!parameter_ports[i] ||
        !seen_ports.insert(parameter_ports[i].get()).second) {
      throw ModelSearchConfigurationError(
          "duplicate or null canonical parameter port in structure '" + key +
          "'");
    }
    const std::map<std::string, std::shared_ptr<GraphPort> >::const_iterator
        registered = impl_->parameters.find(parameter_ids[i]);
    if (registered == impl_->parameters.end()) {
      throw ModelSearchConfigurationError("unknown canonical parameter id '" +
                                           parameter_ids[i] + "'");
    }
    if (registered->second.get() != parameter_ports[i].get()) {
      throw ModelSearchConfigurationError(
          "structure parameter is not the registered canonical owner for '" +
          parameter_ids[i] + "'");
    }
    const std::vector<std::string>::const_iterator ordered = std::find(
        impl_->parameter_order.cbegin(), impl_->parameter_order.cend(),
        parameter_ids[i]);
    const std::size_t index = static_cast<std::size_t>(
        std::distance(impl_->parameter_order.cbegin(), ordered));
    if (!std::isfinite(initial_values[i])) {
      throw ModelSearchConfigurationError(
          "structure initial parameter values must be finite");
    }
    if (fixed_mask[i] != 0 && fixed_mask[i] != 1) {
      throw ModelSearchConfigurationError(
          "structure fixed mask values must be zero or one");
    }
    record.initial_values[index] = initial_values[i];
    record.fixed[index] = fixed_mask[i];
  }
  if (seen_ids.size() != impl_->parameters.size()) {
    throw ModelSearchConfigurationError(
        "structure is missing canonical parameter ids");
  }
  impl_->structures[key] = record;
  impl_->structure_order.push_back(key);
}

std::vector<std::string>
FittingModelSearchProblem::get_structure_keys() const {
  return impl_->structure_order;
}

std::shared_ptr<FitObjective>
FittingModelSearchProblem::get_structure_objective(
    const std::string& key) const {
  return impl_->structure(key).objective;
}

void FittingModelSearchProblem::add_structure_node(
    const std::string& structure_key, std::shared_ptr<GraphNode> node) {
  if (!node) {
    throw ModelSearchConfigurationError("structure graph node is null");
  }
  StructureRecord& selected = impl_->structure(structure_key);
  if (selected.objective.get() == node.get()) {
    throw ModelSearchConfigurationError(
        "the structure objective is retained automatically");
  }
  for (std::size_t i = 0; i < selected.graph_nodes.size(); ++i) {
    if (selected.graph_nodes[i].get() == node.get()) {
      throw ModelSearchConfigurationError(
          "duplicate node in structure graph");
    }
  }
  selected.graph_nodes.push_back(std::move(node));
}

void FittingModelSearchProblem::set_minimizer_maxfev(int value) {
  impl_->maxfev = value;
}

int FittingModelSearchProblem::get_minimizer_maxfev() const {
  return impl_->maxfev;
}

void FittingModelSearchProblem::set_warm_start(bool value) {
  impl_->warm_start = value;
}

bool FittingModelSearchProblem::get_warm_start() const {
  return impl_->warm_start;
}

void FittingModelSearchProblem::add_structure_start(
    const std::string& structure_key,
    const std::vector<double>& initial_values) {
  StructureRecord& selected = impl_->structure(structure_key);
  if (initial_values.size() != impl_->parameter_order.size()) {
    throw ModelSearchConfigurationError(
        "a declared start must cover every canonical parameter");
  }
  for (std::size_t i = 0; i < initial_values.size(); ++i) {
    if (!std::isfinite(initial_values[i])) {
      throw ModelSearchConfigurationError(
          "a declared start must be finite");
    }
  }
  selected.extra_starts.push_back(initial_values);
}

void FittingModelSearchProblem::set_initial_structure(
    const std::string& key) {
  require_key(key, "initial structure");
  if (!impl_->structures.count(key)) {
    throw ModelSearchConfigurationError("unknown initial structure '" + key +
                                         "'");
  }
  impl_->initial_structure = key;
}

void FittingModelSearchProblem::add_action(
    const std::string& parent_structure, const std::string& action_key,
    const std::string& result_structure, double prior, bool terminal) {
  require_key(action_key, "action");
  impl_->structure(parent_structure);
  impl_->structure(result_structure);
  if (!std::isfinite(prior) || prior < 0.0) {
    throw ModelSearchConfigurationError(
        "action prior must be finite and non-negative");
  }
  ModelSearchActions& actions = impl_->actions[parent_structure];
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (actions[i].get_key() == action_key) {
      throw ModelSearchConfigurationError(
          "action keys must be unique within a structure");
    }
  }
  actions.push_back(ModelSearchAction(action_key, result_structure, prior,
                                      terminal));
}

void FittingModelSearchProblem::set_action_policy(const MsgpackBytes& network,
                                                  double temperature) {
  impl_->action_policy.configure(network, temperature);
}

void FittingModelSearchProblem::clear_action_policy() {
  impl_->action_policy.clear();
}

bool FittingModelSearchProblem::get_has_action_policy() const {
  return impl_->action_policy.active();
}

int FittingModelSearchProblem::get_number_of_free_parameters(
    const std::string& structure_key) const {
  const StructureRecord& record = impl_->structure(structure_key);
  int count = 0;
  for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
    if (impl_->is_free(record, i)) ++count;
  }
  return count;
}

std::vector<double> FittingModelSearchProblem::get_policy_rows(
    const std::string& structure_key, const std::vector<double>& residual,
    const ModelSearchActions& actions, const std::vector<int>& block_sizes) const {
  const int source_free = get_number_of_free_parameters(structure_key);
  const std::vector<double> state =
      get_policy_state_features(residual, source_free, block_sizes);
  double declared = 0.0;
  for (std::size_t i = 0; i < actions.size(); ++i) declared += std::max(0.0, actions[i].get_prior());
  std::vector<double> rows;
  rows.reserve(actions.size() * (state.size() + kPolicyActionWidth));
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const std::string& target = actions[i].get_predicted_state_key();
    const std::vector<double> action = get_policy_action_features(
        source_free, get_number_of_free_parameters(target), actions[i].get_terminal(),
        target == structure_key,
        declared > 0.0 ? std::max(0.0, actions[i].get_prior()) / declared
                       : 1.0 / static_cast<double>(actions.size()));
    rows.insert(rows.end(), state.begin(), state.end());
    rows.insert(rows.end(), action.begin(), action.end());
  }
  return rows;
}

void FittingModelSearchProblem::set_structure_score_output(
    const std::string& structure_key, const std::string& output_key) {
  require_key(output_key, "score output");
  StructureRecord& selected = impl_->structure(structure_key);
  if (!selected.objective->get_output_port(output_key)) {
    throw ModelSearchConfigurationError("objective has no score output '" +
                                         output_key + "'");
  }
  selected.score_output = output_key;
}

void FittingModelSearchProblem::clear_structure_score_output(
    const std::string& structure_key) {
  impl_->structure(structure_key).score_output.clear();
}

void FittingModelSearchProblem::set_structure_acceptable_output(
    const std::string& structure_key, const std::string& output_key) {
  require_key(output_key, "acceptable output");
  StructureRecord& selected = impl_->structure(structure_key);
  if (!selected.objective->get_output_port(output_key)) {
    throw ModelSearchConfigurationError(
        "objective has no acceptable output '" + output_key + "'");
  }
  selected.acceptable_output = output_key;
}

void FittingModelSearchProblem::clear_structure_acceptable_output(
    const std::string& structure_key) {
  impl_->structure(structure_key).acceptable_output.clear();
}

void FittingModelSearchProblem::set_structure_selection(
    const std::string& structure_key, ModelSelectionCriterion criterion,
    double effective_sample_size, double complexity) {
  StructureRecord& selected = impl_->structure(structure_key);
  if (!(effective_sample_size > 0.0)) {
    throw ModelSearchConfigurationError(
        "a selection criterion needs a positive number of observations");
  }
  if (complexity < 0.0) {
    throw ModelSearchConfigurationError(
        "a selection criterion needs a non-negative parameter count");
  }
  selected.criterion = criterion;
  selected.effective_sample_size = effective_sample_size;
  selected.complexity = complexity;
  selected.use_bic = true;
}

void FittingModelSearchProblem::set_structure_acceptance(
    const std::string& structure_key, double least_probability) {
  if (least_probability < 0.0 || least_probability > 1.0) {
    throw ModelSearchConfigurationError(
        "an acceptance level is a probability, between zero and one");
  }
  StructureRecord& selected = impl_->structure(structure_key);
  selected.acceptance = least_probability;
  selected.has_acceptance = true;
}

void FittingModelSearchProblem::clear_structure_selection(
    const std::string& structure_key) {
  StructureRecord& selected = impl_->structure(structure_key);
  selected.effective_sample_size = 0.0;
  selected.complexity = 0.0;
  selected.use_bic = false;
}

double FittingModelSearchProblem::get_last_chi2_p_value() const {
  return impl_->last_chi2_p_value;
}

double FittingModelSearchProblem::get_last_reduced_chi2() const {
  return impl_->last_reduced_chi2;
}

ModelSearchState FittingModelSearchProblem::get_initial_state() {
  impl_->validate_registry();
  const StructureRecord& selected =
      impl_->structure(impl_->initial_structure);
  const FitSearchSnapshot previous = impl_->capture();
  const std::string previous_structure = impl_->active_structure;
  try {
    // The root is scored like any other candidate, so it is fitted from
    // every declared start too; otherwise the one topology nothing has to
    // move to would be the one judged on a single seed.
    const std::vector<std::vector<double> > starts =
        impl_->starts_for(selected);
    bool have_best = false;
    double best_reward = 0.0;
    bool best_acceptable = false;
    FitSearchSnapshot best_snapshot;
    double best_reduced_chi2 = 0.0;
    double best_p_value = 0.0;
    for (std::size_t attempt = 0; attempt < starts.size(); ++attempt) {
      impl_->apply_initial(selected, starts[attempt]);
      impl_->active_structure = impl_->initial_structure;
      std::vector<std::shared_ptr<GraphPort> > free_ports;
      for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
        const std::shared_ptr<GraphPort>& port =
            impl_->parameters.find(impl_->parameter_order[i])->second;
        if (!port->get_fixed()) free_ports.push_back(port);
      }
      if (!free_ports.empty() && !impl_->cancelled.load()) {
        for (std::size_t p = 0; p < free_ports.size(); ++p) {
          impl_->keep_inside(free_ports[p]);
        }
        FitMinimizer minimizer;
        minimizer.set_parameter_ports(free_ports);
        if (impl_->maxfev > 0) minimizer.set_maxfev(impl_->maxfev);
        minimizer.set_objective(selected.objective);
        impl_->last_status = minimizer.run();
        if (impl_->last_status < 1 || impl_->last_status > 4) {
          if (attempt + 1 < starts.size()) continue;
          if (!have_best) {
            throw ModelSearchConfigurationError(
                "initial model-search structure did not converge");
          }
          break;
        }
      } else {
        selected.objective->update();
        impl_->last_status = impl_->cancelled.load() ? -1 : 1;
      }
      const std::pair<double, bool> attempt_score =
          impl_->score_current(selected);
      if (!have_best || attempt_score.first > best_reward) {
        have_best = true;
        best_reward = attempt_score.first;
        best_acceptable = attempt_score.second;
        best_reduced_chi2 = impl_->last_reduced_chi2;
        best_p_value = impl_->last_chi2_p_value;
        best_snapshot = impl_->capture();
      }
    }
    impl_->restore_values(best_snapshot);
    impl_->active_structure = impl_->initial_structure;
    impl_->last_reduced_chi2 = best_reduced_chi2;
    impl_->last_chi2_p_value = best_p_value;
    selected.objective->update();
    const std::pair<double, bool> score =
        std::make_pair(best_reward, best_acceptable);
    best_snapshot.residual = impl_->residual_of(selected);
    best_snapshot.blocks = impl_->blocks_of(selected);
    impl_->snapshots[impl_->initial_structure] = best_snapshot;
    impl_->snapshot_structures[impl_->initial_structure] =
        impl_->initial_structure;
    return ModelSearchState(impl_->initial_structure,
                            impl_->initial_structure, score.first,
                            score.second);
  } catch (...) {
    impl_->restore_values(previous);
    impl_->active_structure = previous_structure;
    throw;
  }
}

ModelSearchActions FittingModelSearchProblem::get_actions(
    const ModelSearchState& state) {
  const std::map<std::string, ModelSearchActions>::const_iterator found =
      impl_->actions.find(state.get_structure_key());
  if (found == impl_->actions.end()) return ModelSearchActions();
  if (!impl_->action_policy.active()) return found->second;
  const std::map<std::string, FitSearchSnapshot>::const_iterator snapshot =
      impl_->snapshots.find(state.get_key());
  if (snapshot == impl_->snapshots.end() || snapshot->second.residual.empty()) {
    return found->second;
  }
  return impl_->action_policy.apply(
      found->second, get_policy_rows(state.get_structure_key(), snapshot->second.residual,
                                     found->second, snapshot->second.blocks));
}

ModelSearchState FittingModelSearchProblem::evaluate(
    const ModelSearchState& parent, const ModelSearchAction& action) {
  impl_->validate_registry();
  impl_->last_status = 0;
  impl_->last_failure.clear();
  const std::map<std::string, FitSearchSnapshot>::const_iterator parent_snapshot =
      impl_->snapshots.find(parent.get_key());
  if (parent_snapshot == impl_->snapshots.end()) {
    throw ModelSearchConfigurationError(
        "parent model-search snapshot is not cached");
  }
  const std::map<std::string, std::string>::const_iterator parent_structure =
      impl_->snapshot_structures.find(parent.get_key());
  if (parent_structure == impl_->snapshot_structures.end() ||
      parent_structure->second != parent.get_structure_key()) {
    throw ModelSearchConfigurationError(
        "parent state structure does not match its cached snapshot");
  }
  const std::string target_key = action.get_predicted_state_key();
  const StructureRecord& source =
      impl_->structure(parent.get_structure_key());
  const StructureRecord& target = impl_->structure(target_key);
  try {
    // Every declared start is tried and the best kept. A topology's score is
    // meant to be the best fit it admits, so one seed that happens to land in
    // a poor basin must not be allowed to speak for the model.
    const std::vector<std::vector<double> > starts = impl_->starts_for(target);
    bool have_best = false;
    double best_reward = 0.0;
    bool best_acceptable = false;
    FitSearchSnapshot best_snapshot;
    int best_status = 0;
    // The diagnostics have to describe the state that is returned. Keeping
    // only the last start's would report one fit's goodness beside another
    // fit's answer, which is worse than reporting none.
    double best_reduced_chi2 = 0.0;
    double best_p_value = 0.0;
    std::string last_failure;

    for (std::size_t attempt = 0; attempt < starts.size(); ++attempt) {
      impl_->restore_values(parent_snapshot->second);
      std::vector<std::shared_ptr<GraphPort> > free_ports =
          impl_->apply_transition(source, target, starts[attempt]);
      impl_->active_structure = target_key;
      if (impl_->cancelled.load()) {
        impl_->last_status = -1;
        impl_->last_failure = "cancelled before minimization";
        impl_->rollback(parent_snapshot->second, parent.get_structure_key());
        return parent;
      }
      int status = 1;
      if (!free_ports.empty()) {
        for (std::size_t p = 0; p < free_ports.size(); ++p) {
          impl_->keep_inside(free_ports[p]);
        }
        FitMinimizer minimizer;
        minimizer.set_parameter_ports(free_ports);
        if (impl_->maxfev > 0) minimizer.set_maxfev(impl_->maxfev);
        minimizer.set_objective(target.objective);
        IMP::Pointer<FitSearchCancelObserver> observer(
            new FitSearchCancelObserver(&impl_->cancelled));
        minimizer.set_observer(observer.get());
        status = minimizer.run();
        if (minimizer.get_cancelled() || impl_->cancelled.load()) {
          impl_->last_status = -1;
          impl_->last_failure = "minimization cancelled";
          impl_->rollback(parent_snapshot->second, parent.get_structure_key());
          return parent;
        }
        if (status < 1 || status > 4) {
          std::ostringstream message;
          message << "minimizer did not converge (status " << status << ")";
          last_failure = message.str();
          continue;  // another start may still reach a usable optimum
        }
      } else {
        target.objective->update();
      }
      const std::pair<double, bool> score = impl_->score_current(target);
      if (!have_best || score.first > best_reward) {
        have_best = true;
        best_reward = score.first;
        best_acceptable = score.second;
        best_status = status;
        best_reduced_chi2 = impl_->last_reduced_chi2;
        best_p_value = impl_->last_chi2_p_value;
        best_snapshot = impl_->capture();
      }
    }

    if (!have_best) {
      impl_->last_status = 0;
      impl_->last_failure = last_failure.empty()
                                ? std::string("no declared start converged")
                                : last_failure;
      impl_->rollback(parent_snapshot->second, parent.get_structure_key());
      return parent;
    }

    // Leave the graph standing at the best start, not at the last one tried.
    impl_->restore_values(best_snapshot);
    impl_->active_structure = target_key;
    impl_->last_status = best_status;
    impl_->last_reduced_chi2 = best_reduced_chi2;
    impl_->last_chi2_p_value = best_p_value;
    target.objective->update();
    std::ostringstream state_key;
    state_key << target_key << "@" << impl_->next_snapshot++;
    const std::string key = state_key.str();
    best_snapshot.residual = impl_->residual_of(target);
    best_snapshot.blocks = impl_->blocks_of(target);
    impl_->snapshots[key] = best_snapshot;
    impl_->snapshot_structures[key] = target_key;
    return ModelSearchState(key, target_key, best_reward, best_acceptable);
  } catch (const std::exception& error) {
    impl_->last_failure = error.what();
    impl_->last_status = 0;
    impl_->rollback(parent_snapshot->second, parent.get_structure_key());
    return parent;
  }
}

void FittingModelSearchProblem::request_cancel() {
  impl_->cancelled.store(true);
}

void FittingModelSearchProblem::clear_cancel() {
  impl_->cancelled.store(false);
}

void FittingModelSearchProblem::activate_state(
    const ModelSearchState& state) {
  const std::map<std::string, std::string>::const_iterator found =
      impl_->snapshot_structures.find(state.get_key());
  if (found == impl_->snapshot_structures.end() ||
      found->second != state.get_structure_key()) {
    throw ModelSearchConfigurationError(
        "model-search state does not match a cached structure");
  }
  restore_state(state.get_key());
}

bool FittingModelSearchProblem::has_cached_state(
    const std::string& state_key) const {
  return impl_->snapshots.count(state_key) != 0;
}

std::vector<double> FittingModelSearchProblem::get_cached_values(
    const std::string& state_key) const {
  const std::map<std::string, FitSearchSnapshot>::const_iterator found =
      impl_->snapshots.find(state_key);
  if (found == impl_->snapshots.end()) return std::vector<double>();
  return found->second.values;
}

std::vector<int> FittingModelSearchProblem::get_cached_residual_blocks(
    const std::string& state_key) const {
  const std::map<std::string, FitSearchSnapshot>::const_iterator found =
      impl_->snapshots.find(state_key);
  if (found == impl_->snapshots.end()) return std::vector<int>();
  return found->second.blocks;
}

std::vector<double> FittingModelSearchProblem::get_cached_residual(
    const std::string& state_key) const {
  const std::map<std::string, FitSearchSnapshot>::const_iterator found =
      impl_->snapshots.find(state_key);
  if (found == impl_->snapshots.end()) return std::vector<double>();
  return found->second.residual;
}
std::vector<int> FittingModelSearchProblem::get_cached_fixed(
    const std::string& state_key) const {
  const std::map<std::string, FitSearchSnapshot>::const_iterator found =
      impl_->snapshots.find(state_key);
  if (found == impl_->snapshots.end()) return std::vector<int>();
  return found->second.fixed;
}

void FittingModelSearchProblem::restore_state(
    const std::string& state_key) {
  const std::map<std::string, FitSearchSnapshot>::const_iterator found =
      impl_->snapshots.find(state_key);
  const std::map<std::string, std::string>::const_iterator structure =
      impl_->snapshot_structures.find(state_key);
  if (found == impl_->snapshots.end() ||
      structure == impl_->snapshot_structures.end()) {
    throw ModelSearchConfigurationError("unknown cached model state '" +
                                         state_key + "'");
  }
  impl_->restore(found->second, structure->second);
}

void FittingModelSearchProblem::activate_structure(
    const std::string& key) {
  const StructureRecord& selected = impl_->structure(key);
  impl_->apply_initial(selected, selected.initial_values);
  impl_->select_and_update(key);
}

void FittingModelSearchProblem::select_structure(
    const std::string& key) {
  const StructureRecord& selected = impl_->structure(key);
  for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
    const std::shared_ptr<GraphPort>& port =
        impl_->parameters.find(impl_->parameter_order[i])->second;
    port->set_fixed(!impl_->is_free(selected, i));
  }
  impl_->select_and_update(key);
}

std::vector<std::string>
FittingModelSearchProblem::get_structure_parameter_ids(
    const std::string& key) const {
  const StructureRecord& selected = impl_->structure(key);
  std::vector<std::string> ids;
  for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
    if (impl_->uses(selected, i)) ids.push_back(impl_->parameter_order[i]);
  }
  return ids;
}

void FittingModelSearchProblem::set_structure_parameter_uses(
    const std::string& key, const std::vector<std::string>& canonical_ids) {
  StructureRecord& selected = impl_->structure(key);
  const std::set<std::string> wanted(canonical_ids.begin(),
                                     canonical_ids.end());
  for (std::set<std::string>::const_iterator it = wanted.begin();
       it != wanted.end(); ++it) {
    if (!impl_->parameters.count(*it)) {
      throw ModelSearchConfigurationError("topology '" + key +
                                          "' uses unknown parameter '" + *it +
                                          "'");
    }
  }
  selected.uses.assign(impl_->parameter_order.size(), 0);
  for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
    const bool used = wanted.count(impl_->parameter_order[i]) != 0;
    if (!used && selected.fixed[i] == 0) {
      throw ModelSearchConfigurationError(
          "topology '" + key + "' frees '" + impl_->parameter_order[i] +
          "' but its graph does not read it");
    }
    selected.uses[i] = used ? 1 : 0;
  }
}

void FittingModelSearchProblem::set_parameter_locked(
    const std::string& canonical_id, bool locked) {
  if (!impl_->parameters.count(canonical_id)) {
    throw ModelSearchConfigurationError("unknown canonical parameter '" +
                                        canonical_id + "'");
  }
  const bool was = impl_->locked.count(canonical_id) != 0;
  if (was == locked) return;
  if (locked) {
    impl_->locked.insert(canonical_id);
    impl_->released.erase(canonical_id);
  } else {
    impl_->locked.erase(canonical_id);
  }
  // Every cached state was fitted and scored under the previous lock.
  impl_->snapshots.clear();
  impl_->snapshot_structures.clear();
  if (!impl_->active_structure.empty()) {
    select_structure(impl_->active_structure);
  }
}

void FittingModelSearchProblem::set_parameter_released(
    const std::string& canonical_id, bool released) {
  if (!impl_->parameters.count(canonical_id)) {
    throw ModelSearchConfigurationError("unknown canonical parameter '" +
                                        canonical_id + "'");
  }
  const bool was = impl_->released.count(canonical_id) != 0;
  if (was == released) return;
  if (released) {
    impl_->released.insert(canonical_id);
    impl_->locked.erase(canonical_id);
  } else {
    impl_->released.erase(canonical_id);
  }
  impl_->snapshots.clear();
  impl_->snapshot_structures.clear();
  if (!impl_->active_structure.empty()) {
    select_structure(impl_->active_structure);
  }
}

bool FittingModelSearchProblem::get_parameter_released(
    const std::string& canonical_id) const {
  if (!impl_->parameters.count(canonical_id)) {
    throw ModelSearchConfigurationError("unknown canonical parameter '" +
                                        canonical_id + "'");
  }
  return impl_->released.count(canonical_id) != 0;
}

bool FittingModelSearchProblem::get_parameter_locked(
    const std::string& canonical_id) const {
  if (!impl_->parameters.count(canonical_id)) {
    throw ModelSearchConfigurationError("unknown canonical parameter '" +
                                        canonical_id + "'");
  }
  return impl_->locked.count(canonical_id) != 0;
}

int FittingModelSearchProblem::fit_active_structure() {
  impl_->validate_registry();
  if (impl_->active_structure.empty()) {
    throw ModelSearchConfigurationError(
        "no topology is selected; select one before fitting");
  }
  const std::string key = impl_->active_structure;
  StructureRecord& selected = impl_->structure(key);
  select_structure(key);
  std::vector<std::shared_ptr<GraphPort> > free_ports;
  for (std::size_t i = 0; i < impl_->parameter_order.size(); ++i) {
    const std::shared_ptr<GraphPort>& port =
        impl_->parameters.find(impl_->parameter_order[i])->second;
    if (!port->get_fixed()) free_ports.push_back(port);
  }
  impl_->last_failure.clear();
  impl_->last_status = 1;
  if (!free_ports.empty()) {
    for (std::size_t p = 0; p < free_ports.size(); ++p) {
      impl_->keep_inside(free_ports[p]);
    }
    FitMinimizer minimizer;
    minimizer.set_parameter_ports(free_ports);
    if (impl_->maxfev > 0) minimizer.set_maxfev(impl_->maxfev);
    minimizer.set_objective(selected.objective);
    impl_->last_status = minimizer.run();
  }
  selected.objective->update();
  if (selected.use_bic && selected.score_output.empty()) {
    impl_->score_current(selected);
  }
  return impl_->last_status;
}

void FittingModelSearchProblem::set_structure_curve(
    const std::string& structure_key, const std::string& dataset_name,
    const std::string& node_name) {
  StructureRecord& selected = impl_->structure(structure_key);
  if (!selected.curves.count(dataset_name)) {
    selected.curve_order.push_back(dataset_name);
  }
  selected.curves[dataset_name] = node_name;
}

std::vector<std::string>
FittingModelSearchProblem::get_structure_curve_datasets(
    const std::string& structure_key) const {
  return impl_->structure(structure_key).curve_order;
}

std::string FittingModelSearchProblem::get_structure_curve_node(
    const std::string& structure_key, const std::string& dataset_name) const {
  const StructureRecord& selected = impl_->structure(structure_key);
  const std::map<std::string, std::string>::const_iterator found =
      selected.curves.find(dataset_name);
  if (found == selected.curves.end()) {
    throw ModelSearchConfigurationError("structure '" + structure_key +
                                        "' produces no curve for '" +
                                        dataset_name + "'");
  }
  return found->second;
}

std::vector<double> FittingModelSearchProblem::get_structure_output(
    const std::string& structure_key, const std::string& node_name) {
  const StructureRecord& selected = impl_->structure(structure_key);
  impl_->select_and_update(structure_key);
  std::shared_ptr<GraphNode> found;
  if (selected.objective && selected.objective->get_name() == node_name) {
    found = selected.objective;
  }
  for (std::size_t i = 0; i < selected.graph_nodes.size() && !found; ++i) {
    if (selected.graph_nodes[i] &&
        selected.graph_nodes[i]->get_name() == node_name) {
      found = selected.graph_nodes[i];
    }
  }
  if (!found) {
    throw ModelSearchConfigurationError("structure '" + structure_key +
                                        "' has no node named '" + node_name +
                                        "'");
  }
  found->update();
  const std::shared_ptr<GraphPort> out =
      found->get_output_port(found->get_name());
  if (!out) {
    throw ModelSearchConfigurationError(
        "node '" + node_name + "' publishes no output under its own name");
  }
  return out->get_values_ref();
}

void FittingModelSearchProblem::publish_output(
    const std::string& name, const std::string& node_name,
    const std::string& port_name) {
  if (name.empty() || node_name.empty() || port_name.empty()) {
    throw ModelSearchConfigurationError(
        "a published output needs a name, a node and a port");
  }
  PublishedOutput& published = impl_->outputs[name];
  published.node = node_name;
  published.port = port_name;
  published.source = nullptr;
  if (!published.relay) {
    published.relay = std::make_shared<OutputRelay>("output." + name);
    published.relay->add_input_port(
        "in", std::make_shared<GraphPort>(std::vector<double>(1, 0.0)));
    published.relay->add_output_port(
        "out", std::make_shared<GraphPort>(std::vector<double>(1, 0.0), false,
                                           true, false, false, 0.0, 0.0,
                                           GRAPH_PORT_FLOAT_VECTOR, "out"));
  } else {
    published.relay->get_input_port("in")->unlink();
  }
  impl_->relink_outputs();
}

std::shared_ptr<GraphPort> FittingModelSearchProblem::get_output_port(
    const std::string& name) const {
  std::map<std::string, PublishedOutput>::const_iterator found =
      impl_->outputs.find(name);
  if (found == impl_->outputs.end()) {
    std::ostringstream known;
    for (std::map<std::string, PublishedOutput>::const_iterator it =
             impl_->outputs.begin();
         it != impl_->outputs.end(); ++it) {
      known << (it == impl_->outputs.begin() ? " " : ", ") << it->first;
    }
    throw ModelSearchConfigurationError(
        "the model publishes no output '" + name + "'; it publishes" +
        (impl_->outputs.empty() ? std::string(" none") : known.str()));
  }
  return found->second.relay->get_output_port("out");
}

std::vector<double> FittingModelSearchProblem::get_output(
    const std::string& name) {
  const std::shared_ptr<GraphPort> port = get_output_port(name);
  const std::shared_ptr<GraphNode> relay = port->get_node();
  if (relay) relay->update();
  return port->get_values_ref();
}

std::vector<std::string> FittingModelSearchProblem::get_output_names()
    const {
  std::vector<std::string> names;
  for (std::map<std::string, PublishedOutput>::const_iterator it =
           impl_->outputs.begin();
       it != impl_->outputs.end(); ++it) {
    names.push_back(it->first);
  }
  return names;
}

void FittingModelSearchProblem::adopt_output_ports(
    const FittingModelSearchProblem& previous) {
  for (std::map<std::string, PublishedOutput>::const_iterator it =
           previous.impl_->outputs.begin();
       it != previous.impl_->outputs.end(); ++it) {
    PublishedOutput& published = impl_->outputs[it->first];
    if (published.relay) {
      // Declared here already: keep this model's node and port, and the
      // previous relay that everything follows.
      const std::string node = published.node;
      const std::string port = published.port;
      published = it->second;
      published.node = node;
      published.port = port;
    } else {
      published = it->second;
    }
    published.source = nullptr;
    published.relay->get_input_port("in")->unlink();
  }
  impl_->relink_outputs();
}

std::vector<double> FittingModelSearchProblem::get_structure_port(
    const std::string& structure_key, const std::string& node_name,
    const std::string& port_name) {
  const StructureRecord& selected = impl_->structure(structure_key);
  impl_->select_and_update(structure_key);
  std::shared_ptr<GraphNode> found;
  if (selected.objective && selected.objective->get_name() == node_name) {
    found = selected.objective;
  }
  for (std::size_t i = 0; i < selected.graph_nodes.size() && !found; ++i) {
    if (selected.graph_nodes[i] && selected.graph_nodes[i]->get_name() == node_name) {
      found = selected.graph_nodes[i];
    }
  }
  if (!found) {
    throw ModelSearchConfigurationError("structure '" + structure_key +
                                        "' has no node named '" + node_name + "'");
  }
  found->update();
  const std::shared_ptr<GraphPort> out = found->get_output_port(port_name);
  if (!out) {
    throw ModelSearchConfigurationError("node '" + node_name +
                                        "' has no output '" + port_name + "'");
  }
  return out->get_values_ref();
}

const std::string& FittingModelSearchProblem::get_active_structure()
    const {
  return impl_->active_structure;
}

const std::string& FittingModelSearchProblem::get_initial_structure()
    const {
  return impl_->initial_structure;
}

std::shared_ptr<FitObjective>
FittingModelSearchProblem::get_active_objective() const {
  if (impl_->active_structure.empty()) return std::shared_ptr<FitObjective>();
  return impl_->structure(impl_->active_structure).objective;
}

std::vector<double> FittingModelSearchProblem::get_active_residual() {
  if (impl_->active_structure.empty()) {
    throw ModelSearchConfigurationError("no model structure is active");
  }
  const StructureRecord& selected = impl_->structure(impl_->active_structure);
  selected.objective->update();
  return impl_->residual_of(selected);
}

int FittingModelSearchProblem::get_last_fit_status() const {
  return impl_->last_status;
}

const std::string& FittingModelSearchProblem::get_last_failure() const {
  return impl_->last_failure;
}

ModelSearchConfig::ModelSearchConfig()
    : n_simulations_(200), c_puct_(1.5), reward_scale_(4.0),
      dirichlet_alpha_(0.15), dirichlet_fraction_(0.25), seed_(42u) {}

void ModelSearchConfig::set_number_of_simulations(int value) {
  n_simulations_ = value;
}
int ModelSearchConfig::get_number_of_simulations() const {
  return n_simulations_;
}
void ModelSearchConfig::set_c_puct(double value) { c_puct_ = value; }
double ModelSearchConfig::get_c_puct() const { return c_puct_; }
void ModelSearchConfig::set_reward_scale(double value) {
  reward_scale_ = value;
}
double ModelSearchConfig::get_reward_scale() const { return reward_scale_; }
void ModelSearchConfig::set_dirichlet_alpha(double value) {
  dirichlet_alpha_ = value;
}
double ModelSearchConfig::get_dirichlet_alpha() const {
  return dirichlet_alpha_;
}
void ModelSearchConfig::set_dirichlet_fraction(double value) {
  dirichlet_fraction_ = value;
}
double ModelSearchConfig::get_dirichlet_fraction() const {
  return dirichlet_fraction_;
}
void ModelSearchConfig::set_seed(unsigned int value) { seed_ = value; }
unsigned int ModelSearchConfig::get_seed() const { return seed_; }

ModelSearchResult::ModelSearchResult()
    : n_states_evaluated_(0), n_simulations_(0), improvement_(0.0),
      acceptable_(false), cancelled_(false) {}

const ModelSearchState& ModelSearchResult::get_root_state() const {
  return root_state_;
}
const ModelSearchState& ModelSearchResult::get_best_state() const {
  return best_state_;
}
const std::vector<std::string>& ModelSearchResult::get_best_path() const {
  return best_path_;
}
int ModelSearchResult::get_number_of_states_evaluated() const {
  return n_states_evaluated_;
}
int ModelSearchResult::get_number_of_simulations() const {
  return n_simulations_;
}
double ModelSearchResult::get_improvement() const { return improvement_; }
bool ModelSearchResult::get_acceptable() const { return acceptable_; }
bool ModelSearchResult::get_cancelled() const { return cancelled_; }

struct ModelSearch::Impl {
  std::shared_ptr<ModelSearchProblem> problem;
  ModelSearchConfig config;
  std::atomic<bool> cancel_requested;
  Impl() : cancel_requested(false) {}
};

ModelSearch::ModelSearch() : impl_(new Impl) {}
ModelSearch::ModelSearch(std::shared_ptr<ModelSearchProblem> problem)
    : impl_(new Impl) {
  impl_->problem = std::move(problem);
}
ModelSearch::~ModelSearch() {}

void ModelSearch::set_problem(
    std::shared_ptr<ModelSearchProblem> problem) {
  impl_->problem = std::move(problem);
}
std::shared_ptr<ModelSearchProblem> ModelSearch::get_problem() const {
  return impl_->problem;
}
void ModelSearch::set_config(const ModelSearchConfig& config) {
  impl_->config = config;
}
ModelSearchConfig ModelSearch::get_config() const { return impl_->config; }
void ModelSearch::request_cancel() {
  impl_->cancel_requested.store(true);
  if (impl_->problem) impl_->problem->request_cancel();
}
void ModelSearch::clear_cancel() {
  impl_->cancel_requested.store(false);
  if (impl_->problem) impl_->problem->clear_cancel();
}
bool ModelSearch::get_cancel_requested() const {
  return impl_->cancel_requested.load();
}

ModelSearchResult ModelSearch::run() {
  if (!impl_->problem) {
    throw ModelSearchConfigurationError("model search has no problem");
  }
  const ModelSearchConfig cfg = impl_->config;
  if (cfg.get_number_of_simulations() < 0 ||
      !std::isfinite(cfg.get_c_puct()) || cfg.get_c_puct() < 0.0 ||
      !std::isfinite(cfg.get_reward_scale()) ||
      cfg.get_reward_scale() <= 0.0 ||
      !std::isfinite(cfg.get_dirichlet_fraction()) ||
      cfg.get_dirichlet_fraction() < 0.0 ||
      cfg.get_dirichlet_fraction() > 1.0 ||
      (cfg.get_dirichlet_fraction() > 0.0 &&
       (!std::isfinite(cfg.get_dirichlet_alpha()) ||
        cfg.get_dirichlet_alpha() <= 0.0))) {
    throw ModelSearchConfigurationError("invalid model-search controls");
  }

  const ModelSearchState initial = impl_->problem->get_initial_state();
  require_state(initial, "initial state");
  const double root_reward = initial.get_reward();
  std::mt19937_64 rng(cfg.get_seed());
  int evaluator_calls = 0;

  std::unique_ptr<TreeNode> root(new TreeNode);
  root->state = initial;
  root->structure_key = initial.get_structure_key();
  root->evaluated = true;
  root->visits = 1;

  const auto expand = [&](TreeNode* node) {
    ModelSearchActions actions =
        impl_->problem->get_actions(node->state);
    std::set<std::string> action_keys;
    double prior_sum = 0.0;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      require_key(actions[i].get_key(), "action");
      require_key(actions[i].get_predicted_state_key(), "predicted state");
      if (!action_keys.insert(actions[i].get_key()).second) {
        throw ModelSearchConfigurationError(
            "action keys must be unique within a state");
      }
      if (!std::isfinite(actions[i].get_prior()) ||
          actions[i].get_prior() < 0.0) {
        throw ModelSearchConfigurationError(
            "action prior must be finite and non-negative");
      }
      prior_sum += actions[i].get_prior();
    }
    const bool uniform = !actions.empty() && prior_sum == 0.0;

    std::vector<double> priors(actions.size(), 0.0);
    for (std::size_t i = 0; i < actions.size(); ++i) {
      priors[i] = uniform ? 1.0 / actions.size()
                          : actions[i].get_prior() / prior_sum;
    }
    if (node == root.get() && actions.size() > 1 &&
        cfg.get_dirichlet_fraction() > 0.0) {
      std::gamma_distribution<double> gamma(cfg.get_dirichlet_alpha(), 1.0);
      std::vector<double> noise(actions.size());
      double noise_sum = 0.0;
      for (std::size_t i = 0; i < noise.size(); ++i) {
        noise[i] = gamma(rng);
        noise_sum += noise[i];
      }
      if (noise_sum > 0.0) {
        for (std::size_t i = 0; i < priors.size(); ++i) {
          priors[i] = (1.0 - cfg.get_dirichlet_fraction()) * priors[i] +
                      cfg.get_dirichlet_fraction() * noise[i] / noise_sum;
        }
      }
    }

    std::set<std::string> ancestor_keys;
    for (TreeNode* ancestor = node->parent; ancestor;
         ancestor = ancestor->parent) {
      ancestor_keys.insert(ancestor->structure_key);
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (ancestor_keys.count(actions[i].get_predicted_state_key())) continue;
      std::unique_ptr<TreeNode> child(new TreeNode(priors[i], node));
      child->parent_action = actions[i];
      child->terminal = actions[i].get_terminal();
      child->structure_key = actions[i].get_predicted_state_key();
      node->children.push_back(std::move(child));
    }
    node->expanded = true;
  };

  expand(root.get());
  int done = 0;
  while (done < cfg.get_number_of_simulations() &&
         !impl_->cancel_requested.load()) {
    std::vector<TreeNode*> path(1, root.get());
    TreeNode* node = root.get();
    while (node->expanded && !node->children.empty()) {
      TreeNode* best = node->children[0].get();
      double best_score = puct(*best, node->visits, cfg.get_c_puct());
      for (std::size_t i = 1; i < node->children.size(); ++i) {
        TreeNode* candidate = node->children[i].get();
        const double score = puct(*candidate, node->visits,
                                  cfg.get_c_puct());
        if (score > best_score) {
          best = candidate;
          best_score = score;
        }
      }
      node = best;
      path.push_back(node);
      if (node->terminal) break;
    }

    double value = 0.0;
    if (node->terminal) {
      node->state = node->parent->state;
      node->evaluated = true;
      value = reward_value(node->state, root_reward, cfg.get_reward_scale());
    } else if (!node->evaluated) {
      node->state = impl_->problem->evaluate(node->parent->state,
                                             node->parent_action);
      ++evaluator_calls;
      require_state(node->state, "evaluated state");
      node->structure_key = node->state.get_structure_key();
      node->evaluated = true;
      for (TreeNode* ancestor = node->parent; ancestor;
           ancestor = ancestor->parent) {
        if (ancestor->structure_key == node->structure_key) {
          node->terminal = true;
          node->expanded = true;
          break;
        }
      }
      value = reward_value(node->state, root_reward, cfg.get_reward_scale());
    } else if (node->expanded && node->children.empty()) {
      value = reward_value(node->state, root_reward, cfg.get_reward_scale());
    } else {
      expand(node);
      value = reward_value(node->state, root_reward, cfg.get_reward_scale());
    }

    for (std::size_t i = 0; i < path.size(); ++i) {
      ++path[i]->visits;
      path[i]->value_sum += value;
    }
    ++done;
  }

  TreeNode* best = root.get();
  std::vector<TreeNode*> stack(1, root.get());
  while (!stack.empty()) {
    TreeNode* node = stack.back();
    stack.pop_back();
    if (node->evaluated &&
        node->state.get_reward() > best->state.get_reward()) {
      best = node;
    }
    for (std::size_t i = 0; i < node->children.size(); ++i) {
      stack.push_back(node->children[i].get());
    }
  }

  ModelSearchResult result;
  result.root_state_ = initial;
  result.best_state_ = best->state;
  result.n_states_evaluated_ = 1 + evaluator_calls;
  result.n_simulations_ = done;
  result.improvement_ = best->state.get_reward() - root_reward;
  result.acceptable_ = best->state.get_acceptable();
  result.cancelled_ = done < cfg.get_number_of_simulations() &&
                      impl_->cancel_requested.load();

  TreeNode* path_node = root.get();
  while (path_node->expanded && !path_node->children.empty()) {
    TreeNode* most_visited = path_node->children[0].get();
    for (std::size_t i = 1; i < path_node->children.size(); ++i) {
      if (path_node->children[i]->visits > most_visited->visits) {
        most_visited = path_node->children[i].get();
      }
    }
    if (most_visited->terminal || most_visited->visits == 0) break;
    result.best_path_.push_back(most_visited->parent_action.get_key());
    if (!most_visited->evaluated) break;
    path_node = most_visited;
  }
  impl_->problem->activate_state(result.cancelled_ ? result.root_state_
                                                  : result.best_state_);
  return result;
}

IMPBFF_END_NAMESPACE

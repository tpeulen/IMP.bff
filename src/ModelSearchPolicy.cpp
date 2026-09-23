/**
 * \file ModelSearchPolicy.cpp
 * \brief Episodes for, and training of, one action policy across model families.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ModelSearchPolicy.h>
#include <IMP/bff/ModelSearch.h>
#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/AdamUpdate.h>

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

int policy_row_width() {
  return get_policy_state_width() + get_policy_action_width();
}

//! Index of the best-scoring move; the first one on a tie.
int best_of(const std::vector<double>& score, int begin, int end) {
  int best = begin;
  for (int i = begin + 1; i < end; ++i) {
    if (score[static_cast<std::size_t>(i)] > score[static_cast<std::size_t>(best)]) best = i;
  }
  return best - begin;
}

double log_prior(double prior) { return std::log(std::max(prior, 1e-12)); }

}  // namespace

ModelSearchPolicyData::ModelSearchPolicyData() : width_(policy_row_width()) {
  offsets_.push_back(0);
}

void ModelSearchPolicyData::add_episode(const std::vector<double>& rows,
                                        const std::vector<double>& priors,
                                        int label, const std::string& family,
                                        bool stop) {
  const std::size_t n = priors.size();
  if (n == 0 || rows.size() != n * static_cast<std::size_t>(width_)) {
    std::ostringstream message;
    message << "a policy episode needs one row of " << width_
            << " features per move; got " << rows.size() << " values for " << n
            << " moves";
    throw ModelSearchConfigurationError(message.str());
  }
  if (label < 0 || static_cast<std::size_t>(label) >= n) {
    throw ModelSearchConfigurationError("a policy episode's label must index one of its moves");
  }
  rows_.insert(rows_.end(), rows.begin(), rows.end());
  priors_.insert(priors_.end(), priors.begin(), priors.end());
  offsets_.push_back(offsets_.back() + static_cast<int>(n));
  labels_.push_back(label);
  families_.push_back(family);
  stop_.push_back(stop ? 1 : 0);
}

void ModelSearchPolicyData::extend(const ModelSearchPolicyData& other) {
  if (other.width_ != width_) {
    throw ModelSearchConfigurationError("policy data sets disagree on the row width");
  }
  const int shift = offsets_.back();
  rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end());
  priors_.insert(priors_.end(), other.priors_.begin(), other.priors_.end());
  for (std::size_t e = 1; e < other.offsets_.size(); ++e) {
    offsets_.push_back(shift + other.offsets_[e]);
  }
  labels_.insert(labels_.end(), other.labels_.begin(), other.labels_.end());
  families_.insert(families_.end(), other.families_.begin(), other.families_.end());
  stop_.insert(stop_.end(), other.stop_.begin(), other.stop_.end());
}

int ModelSearchPolicyData::get_number_of_episodes() const {
  return static_cast<int>(labels_.size());
}
int ModelSearchPolicyData::get_number_of_rows() const { return offsets_.back(); }
int ModelSearchPolicyData::get_row_width() const { return width_; }

std::vector<std::string> ModelSearchPolicyData::get_families() const {
  const std::set<std::string> unique(families_.begin(), families_.end());
  return std::vector<std::string>(unique.begin(), unique.end());
}

std::vector<int> ModelSearchPolicyData::get_family_counts() const {
  const std::vector<std::string> names = get_families();
  std::vector<int> counts(names.size(), 0);
  for (const std::string& family : families_) {
    counts[static_cast<std::size_t>(
        std::lower_bound(names.begin(), names.end(), family) - names.begin())] += 1;
  }
  return counts;
}

int ModelSearchPolicyData::get_number_of_stop_labels() const {
  int count = 0;
  for (int s : stop_) count += s;
  return count;
}

std::string ModelSearchPolicyData::to_json() const {
  nlohmann::json doc;
  doc["format"] = "bff.model_search.policy_data.v1";
  doc["width"] = width_;
  doc["rows"] = rows_;
  doc["priors"] = priors_;
  doc["offsets"] = offsets_;
  doc["labels"] = labels_;
  doc["families"] = families_;
  doc["stop"] = stop_;
  return doc.dump();
}

ModelSearchPolicyData ModelSearchPolicyData::from_json(const std::string& text) {
  const nlohmann::json doc = nlohmann::json::parse(text);
  if (doc.value("format", std::string()) != "bff.model_search.policy_data.v1") {
    throw ModelSearchConfigurationError("not a bff.model_search.policy_data.v1 document");
  }
  ModelSearchPolicyData data;
  if (doc.at("width").get<int>() != data.width_) {
    throw ModelSearchConfigurationError(
        "policy data was recorded with a different feature width");
  }
  data.rows_ = doc.at("rows").get<std::vector<double> >();
  data.priors_ = doc.at("priors").get<std::vector<double> >();
  data.offsets_ = doc.at("offsets").get<std::vector<int> >();
  data.labels_ = doc.at("labels").get<std::vector<int> >();
  data.families_ = doc.at("families").get<std::vector<std::string> >();
  data.stop_ = doc.at("stop").get<std::vector<int> >();
  if (data.offsets_.empty() || data.offsets_.size() != data.labels_.size() + 1 ||
      data.families_.size() != data.labels_.size() ||
      data.stop_.size() != data.labels_.size() ||
      data.priors_.size() != static_cast<std::size_t>(data.offsets_.back()) ||
      data.rows_.size() != data.priors_.size() * static_cast<std::size_t>(data.width_)) {
    throw ModelSearchConfigurationError("policy data document is inconsistent");
  }
  return data;
}

ModelSearchPolicyTraining::ModelSearchPolicyTraining()
    : training_loss_(std::numeric_limits<double>::quiet_NaN()),
      training_accuracy_(std::numeric_limits<double>::quiet_NaN()),
      n_validation_(0),
      validation_loss_(std::numeric_limits<double>::quiet_NaN()),
      validation_accuracy_(std::numeric_limits<double>::quiet_NaN()),
      baseline_loss_(std::numeric_limits<double>::quiet_NaN()),
      baseline_accuracy_(std::numeric_limits<double>::quiet_NaN()),
      best_epoch_(0) {}

std::vector<std::string> ModelSearchPolicyTraining::get_families() const {
  std::vector<std::string> names;
  for (const auto& entry : family_accuracy_) names.push_back(entry.first);
  return names;
}

std::vector<double> ModelSearchPolicyTraining::get_family_validation_accuracy() const {
  std::vector<double> values;
  for (const auto& entry : family_accuracy_) values.push_back(entry.second);
  return values;
}

namespace {

//! One subset of episodes with its rows gathered contiguously.
struct EpisodeSet {
  std::vector<double> x;
  std::vector<double> log_prior;
  std::vector<int> offsets{0};
  std::vector<int> labels;
  std::vector<std::string> families;

  int rows() const { return offsets.back(); }
  int episodes() const { return static_cast<int>(labels.size()); }
};

EpisodeSet gather(const ModelSearchPolicyData& data, const std::vector<int>& which,
                  const std::vector<std::string>& families) {
  EpisodeSet set;
  const int width = data.get_row_width();
  for (int e : which) {
    const int begin = data.get_offsets()[static_cast<std::size_t>(e)];
    const int end = data.get_offsets()[static_cast<std::size_t>(e) + 1];
    set.x.insert(set.x.end(), data.get_rows().begin() + begin * width,
                 data.get_rows().begin() + end * width);
    for (int r = begin; r < end; ++r) {
      set.log_prior.push_back(log_prior(data.get_priors()[static_cast<std::size_t>(r)]));
    }
    set.offsets.push_back(set.offsets.back() + end - begin);
    set.labels.push_back(data.get_labels()[static_cast<std::size_t>(e)]);
    set.families.push_back(families[static_cast<std::size_t>(e)]);
  }
  return set;
}

//! Mean cross-entropy and top-1 accuracy; fills `gradient` when asked.
std::pair<double, double> listwise(const EpisodeSet& set, const std::vector<double>& z,
                                   std::vector<double>* gradient,
                                   std::map<std::string, std::pair<int, int> >* per_family) {
  double loss = 0.0;
  int correct = 0;
  if (gradient) gradient->assign(z.size(), 0.0);
  std::vector<double> score(z.size());
  for (std::size_t i = 0; i < z.size(); ++i) score[i] = set.log_prior[i] + z[i];
  for (int e = 0; e < set.episodes(); ++e) {
    const int begin = set.offsets[static_cast<std::size_t>(e)];
    const int end = set.offsets[static_cast<std::size_t>(e) + 1];
    double maximum = -std::numeric_limits<double>::infinity();
    for (int i = begin; i < end; ++i) maximum = std::max(maximum, score[static_cast<std::size_t>(i)]);
    double total = 0.0;
    for (int i = begin; i < end; ++i) total += std::exp(score[static_cast<std::size_t>(i)] - maximum);
    const int label = begin + set.labels[static_cast<std::size_t>(e)];
    loss -= score[static_cast<std::size_t>(label)] - maximum - std::log(total);
    const bool hit = best_of(score, begin, end) == set.labels[static_cast<std::size_t>(e)];
    if (hit) ++correct;
    if (per_family) {
      std::pair<int, int>& tally = (*per_family)[set.families[static_cast<std::size_t>(e)]];
      tally.first += hit ? 1 : 0;
      tally.second += 1;
    }
    if (gradient) {
      for (int i = begin; i < end; ++i) {
        const double p = std::exp(score[static_cast<std::size_t>(i)] - maximum) / total;
        (*gradient)[static_cast<std::size_t>(i)] =
            (p - (i == label ? 1.0 : 0.0)) / set.episodes();
      }
    }
  }
  return std::make_pair(loss / std::max(1, set.episodes()),
                        static_cast<double>(correct) / std::max(1, set.episodes()));
}

}  // namespace

ModelSearchPolicyTraining train_action_policy(const ModelSearchPolicyData& data,
                                              const std::vector<int>& hidden,
                                              int epochs, double learning_rate,
                                              unsigned int seed,
                                              double validation_fraction,
                                              double weight_decay) {
  if (data.get_number_of_episodes() == 0 || epochs <= 0 || !(learning_rate > 0.0)) {
    throw ModelSearchConfigurationError(
        "policy training needs episodes, epochs and a learning rate");
  }
  if (!(validation_fraction >= 0.0) || !(validation_fraction < 1.0)) {
    throw ModelSearchConfigurationError("validation fraction must lie in [0, 1)");
  }
  if (!(weight_decay >= 0.0)) {
    throw ModelSearchConfigurationError("weight decay must not be negative");
  }
  const int width = data.get_row_width();
  const int n = data.get_number_of_episodes();
  std::mt19937 rng(seed);
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::shuffle(order.begin(), order.end(), rng);
  // Held-out episodes split in two: one half chooses when to stop, the other
  // is only ever scored, so the reported accuracy is not the one optimised.
  const int n_held = static_cast<int>(validation_fraction * n);
  const int n_select = n_held / 2;
  const int n_test = n_held - n_select;
  if (n - n_held <= 0) throw ModelSearchConfigurationError("no policy episodes left to train on");
  const std::vector<std::string>& families = data.get_episode_families();
  const EpisodeSet test = gather(
      data, std::vector<int>(order.begin(), order.begin() + n_test), families);
  const EpisodeSet select = gather(
      data, std::vector<int>(order.begin() + n_test, order.begin() + n_held), families);
  const EpisodeSet train = gather(
      data, std::vector<int>(order.begin() + n_held, order.end()), families);

  internal::MlpModel model;
  int previous = width;
  std::vector<int> widths(hidden);
  widths.push_back(1);
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
    // The output starts at zero, so an untrained policy is the declared priors.
    if (l + 1 == widths.size()) std::fill(layer.weight.begin(), layer.weight.end(), 0.0);
    layer.bias.assign(layer.n_out, 0.0);
    model.layers.push_back(layer);
    previous = layer.n_out;
  }
  model.x_scaler.mean.assign(width, 0.0);
  model.x_scaler.scale.assign(width, 1.0);
  const int rows = train.rows();
  for (int j = 0; j < width; ++j) {
    double sum = 0.0, variance = 0.0;
    for (int r = 0; r < rows; ++r) sum += train.x[static_cast<std::size_t>(r) * width + j];
    const double mean = sum / rows;
    for (int r = 0; r < rows; ++r) {
      const double d = train.x[static_cast<std::size_t>(r) * width + j] - mean;
      variance += d * d;
    }
    const double scale = std::sqrt(variance / rows);
    model.x_scaler.mean[j] = mean;
    // A feature that never varies (a family without that kind of move) is
    // passed through rather than divided by nothing.
    model.x_scaler.scale[j] = scale > 1e-9 ? scale : 1.0;
  }
  model.validate();

  std::vector<double> flat, z, dy, d2y, gradient, dparams, dX, dV;
  internal::mlpcore::flatten(model.layers, flat);
  internal::AdamState adam;
  adam.reset(flat.size());
  std::vector<double> best = flat;
  int best_epoch = 0;
  double best_loss = std::numeric_limits<double>::infinity();
  if (n_select > 0) {
    internal::mlpcore::model_predict(model, select.x.data(), select.rows(), 0, nullptr, z, dy, d2y);
    best_loss = listwise(select, z, nullptr, nullptr).first;
  }
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    internal::mlpcore::model_predict(model, train.x.data(), rows, 0, nullptr, z, dy, d2y);
    listwise(train, z, &gradient, nullptr);
    internal::mlpcore::model_backward(model, train.x.data(), rows, nullptr,
                                      gradient.data(), nullptr, nullptr, dparams, dX, dV);
    for (std::size_t i = 0; i < flat.size(); ++i) dparams[i] += weight_decay * flat[i];
    internal::adam_update(flat.data(), dparams.data(), flat.size(), adam, learning_rate);
    internal::mlpcore::unflatten(model.layers, flat.data(), flat.size());
    if (n_select > 0) {
      internal::mlpcore::model_predict(model, select.x.data(), select.rows(), 0, nullptr, z,
                                       dy, d2y);
      const double loss = listwise(select, z, nullptr, nullptr).first;
      if (loss < best_loss) {
        best_loss = loss;
        best = flat;
        best_epoch = epoch;
      }
    } else {
      best = flat;
      best_epoch = epoch;
    }
  }
  internal::mlpcore::unflatten(model.layers, best.data(), best.size());

  ModelSearchPolicyTraining result;
  result.best_epoch_ = best_epoch;
  internal::mlpcore::model_predict(model, train.x.data(), rows, 0, nullptr, z, dy, d2y);
  const std::pair<double, double> fitted = listwise(train, z, nullptr, nullptr);
  result.training_loss_ = fitted.first;
  result.training_accuracy_ = fitted.second;
  result.n_validation_ = n_test;
  if (n_test > 0) {
    std::map<std::string, std::pair<int, int> > per_family, per_family_baseline;
    internal::mlpcore::model_predict(model, test.x.data(), test.rows(), 0, nullptr, z, dy, d2y);
    const std::pair<double, double> held = listwise(test, z, nullptr, &per_family);
    result.validation_loss_ = held.first;
    result.validation_accuracy_ = held.second;
    const std::vector<double> none(static_cast<std::size_t>(test.rows()), 0.0);
    const std::pair<double, double> baseline = listwise(test, none, nullptr, &per_family_baseline);
    result.baseline_loss_ = baseline.first;
    result.baseline_accuracy_ = baseline.second;
    for (const auto& entry : per_family) {
      result.family_accuracy_[entry.first] =
          static_cast<double>(entry.second.first) / entry.second.second;
      const std::pair<int, int>& prior = per_family_baseline[entry.first];
      result.family_baseline_[entry.first] = static_cast<double>(prior.first) / prior.second;
    }
  }
  result.network_ = internal::mlpcore::model_to_json<nlohmann::json>(model).dump();
  return result;
}

std::vector<double> ModelSearchPolicyTraining::get_family_baseline_accuracy() const {
  std::vector<double> values;
  for (const auto& entry : family_baseline_) values.push_back(entry.second);
  return values;
}

std::string get_shipped_action_policy() {
  std::string path;
  try {
    path = get_data_path("model_search/action_policy.json");
  } catch (...) {
    return std::string();
  }
  std::ifstream in(path.c_str());
  if (!in) return std::string();
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

IMPBFF_END_NAMESPACE

/**
 *  \file IMP/bff/ModelSearchPolicy.h
 *  \brief Episodes for, and training of, one action policy across model families.
 *
 *  A search episode is a fitted structure, the moves it offered, and which of
 *  them led towards the structure that generated the data. Every move is
 *  described by family-agnostic features (get_policy_state_features,
 *  get_policy_action_features), so episodes from FCS, TCSPC, anisotropy or a
 *  kinetic scheme fitted to several measurements at once pool into one data
 *  set, and one network learns to rank moves for all of them.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_MODELSEARCHPOLICY_H
#define IMPBFF_MODELSEARCHPOLICY_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Grouped (state, move) feature rows with the move that was right, per episode.
class IMPBFFEXPORT ModelSearchPolicyData {
 public:
  ModelSearchPolicyData();

  //! Record one episode: one row per move, flattened, and the correct move.
  /*! `priors` are the moves' declared priors, one per row. Training scores a
      move as `log(prior) + network(row)`, exactly as a search weights it,
      so the network learns a correction to what the family declares.
      \throws ModelSearchConfigurationError unless every row has
      get_policy_state_width() + get_policy_action_width() features and
      `label` indexes one of them. */
  void add_episode(const std::vector<double>& rows,
                   const std::vector<double>& priors, int label,
                   const std::string& family, bool stop = false);
  //! Append another data set's episodes, e.g. another family's.
  void extend(const ModelSearchPolicyData& other);

  int get_number_of_episodes() const;
  int get_number_of_rows() const;
  int get_row_width() const;
  //! The families episodes came from, sorted.
  std::vector<std::string> get_families() const;
  //! Episodes per family, in #get_families order.
  std::vector<int> get_family_counts() const;
  //! How often the correct move was terminal, i.e. the right answer was to stop.
  int get_number_of_stop_labels() const;

  const std::vector<double>& get_rows() const { return rows_; }
  const std::vector<int>& get_offsets() const { return offsets_; }
  const std::vector<int>& get_labels() const { return labels_; }
  const std::vector<double>& get_priors() const { return priors_; }
  const std::vector<std::string>& get_episode_families() const { return families_; }

  std::string to_json() const;
  static ModelSearchPolicyData from_json(const std::string& text);

  IMP_SHOWABLE_INLINE(ModelSearchPolicyData,
                      out << "ModelSearchPolicyData(" << labels_.size()
                          << " episodes)");

 private:
  int width_;
  std::vector<double> rows_;
  std::vector<double> priors_;
  //! Row index at which each episode starts; one more entry than episodes.
  std::vector<int> offsets_;
  std::vector<int> labels_;
  std::vector<std::string> families_;
  std::vector<int> stop_;
};
IMP_VALUES(ModelSearchPolicyData, ModelSearchPolicyDatas);

//! What training produced: the network and how well it ranks held-out moves.
class IMPBFFEXPORT ModelSearchPolicyTraining {
 public:
  ModelSearchPolicyTraining();
  //! A `NeuralNet` document for FittingModelSearchProblem::set_action_policy.
  const std::string& get_network() const { return network_; }
  double get_training_loss() const { return training_loss_; }
  double get_training_accuracy() const { return training_accuracy_; }
  //! Held-out episodes scored for the report; as many again chose the epoch.
  int get_number_of_validation_episodes() const { return n_validation_; }
  //! The epoch whose network was kept: lowest loss on the selection half.
  int get_best_epoch() const { return best_epoch_; }
  //! NaN when nothing was held out.
  double get_validation_loss() const { return validation_loss_; }
  double get_validation_accuracy() const { return validation_accuracy_; }
  //! Share of held-out episodes where the declared priors alone ranked the
  //! right move first -- the bar a policy has to clear to be worth shipping.
  double get_baseline_accuracy() const { return baseline_accuracy_; }
  //! The declared priors' own cross-entropy on the held-out episodes.
  double get_baseline_loss() const { return baseline_loss_; }
  std::vector<std::string> get_families() const;
  //! Held-out top-1 accuracy per family, in #get_families order.
  std::vector<double> get_family_validation_accuracy() const;
  //! The declared priors' held-out accuracy per family, in #get_families order.
  std::vector<double> get_family_baseline_accuracy() const;

  IMP_SHOWABLE_INLINE(ModelSearchPolicyTraining,
                      out << "ModelSearchPolicyTraining(validation accuracy "
                          << validation_accuracy_ << ")");

 private:
#ifndef SWIG
  friend ModelSearchPolicyTraining train_action_policy(
      const ModelSearchPolicyData&, const std::vector<int>&, int, double,
      unsigned int, double, double);
#endif
  std::string network_;
  double training_loss_;
  double training_accuracy_;
  int n_validation_;
  double validation_loss_;
  double validation_accuracy_;
  double baseline_loss_;
  double baseline_accuracy_;
  int best_epoch_;
  std::map<std::string, double> family_accuracy_;
  std::map<std::string, double> family_baseline_;
};
IMP_VALUES(ModelSearchPolicyTraining, ModelSearchPolicyTrainings);

//! Train one move-scoring network over every episode, whatever its family.
/*! Listwise: within an episode the moves' scores `log(prior) + network(row)`
    are softmaxed and the cross-entropy against the correct move is minimised
    with Adam over bff's MlpCore, plus `weight_decay` times the squared
    weights. Hidden layers are tanh, the output is linear and starts at zero,
    inputs are standardised. `seed` draws the weights and the split. A
    `validation_fraction` of episodes is never trained on: half of it picks
    the epoch to keep (early stopping), the other half is what the reported
    accuracies are measured on. */
IMPBFFEXPORT ModelSearchPolicyTraining train_action_policy(
    const ModelSearchPolicyData& data, const std::vector<int>& hidden,
    int epochs, double learning_rate, unsigned int seed = 67890,
    double validation_fraction = 0.3, double weight_decay = 1e-4);

//! The action policy bff ships, trained on every family it carries.
/*! `data/model_search/policy/action_policy.msgpack` -- a `bff.neural_net`
    document stored as msgpack, so the float64 weights are kept exactly and
    compactly -- returned as the JSON text
    FittingModelSearchProblem::set_action_policy reads; empty when
    this build ships none, which searches on the declared priors alone. */
IMPBFFEXPORT std::string get_shipped_action_policy();

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_MODELSEARCHPOLICY_H

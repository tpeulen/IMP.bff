/**
 *  \file IMP/bff/ModelSearchSelfPlay.h
 *  \brief A model family teaching itself where its parameters start.
 *
 *  Measurement, not intuition, decided what this learns. Once a candidate has
 *  a score of its own, the tree search already agrees with an exhaustive walk
 *  on every fixture and every seed -- *choosing which topology to try next is
 *  not the bottleneck*. What fails is reaching a topology's optimum: on a
 *  two-species FCS curve the generating model scored -915 from its declared
 *  seed and -12.6 from a good one, and the twenty-four-parameter VV/VH/VM
 *  family selects the generating topology in three noise draws out of seven,
 *  purely because generic seeds miss the basin. Seeded where the optimiser
 *  can reach it, the answer is right every time.
 *
 *  This class learns both **where to start parameters** from simulated curves
 *  and which structural action repairs a fitted residual. The latter samples
 *  a richer child topology, fits its declared parent to that synthetic data,
 *  and learns from the parent's weighted residual. `FittingModelSearchProblem`
 *  and `FittingModelSearchProblem` consume the resulting policy in the
 *  same native C++ process.
 *
 *  The loop needs no new physics and no hand-written simulator, which is the
 *  point: a family is a description, a description can already produce the
 *  curve it predicts (FittingModelSearchProblem::get_structure_output),
 *  so *any* family -- including one added tomorrow as a file -- can generate
 *  its own training data. Sample parameters, simulate the measurement they
 *  imply, and remember the pair. The features are what a fitter could compute
 *  before it knows the answer; the targets are the answer.
 *
 *  Training uses the vendored `MlpCore` kernels, which already carry the
 *  reverse-mode backward pass, and writes the same JSON `NeuralNet` loads --
 *  so there is one MLP implementation, tttrlib owns it, and a trained
 *  proposer is read back through the ordinary inference path.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_MODELSEARCHSELFPLAY_H
#define IMPBFF_MODELSEARCHSELFPLAY_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/ModelSearchSpec.h>

#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Episodes a model family generates about itself, and the proposer they train.
class IMPBFFEXPORT ModelSearchSelfPlay {
 public:
  //! Play a family against itself.
  /*! \param[in] spec the family, with its measurements already bound -- the
      bound data fixes the shape and sampling of what gets simulated, not its
      values, which every episode replaces. */
  explicit ModelSearchSelfPlay(const ModelSearchSpec& spec);
  ~ModelSearchSelfPlay();

  //! The topology episodes are generated for.
  void set_structure(const std::string& structure_key);
  const std::string& get_structure() const;

  //! How far a sampled parameter may stray from its declared start, in decades.
  /*! Wide enough that the proposer sees basins a fixed seed misses, narrow
      enough that the episodes remain measurements someone could have taken. */
  void set_spread(double decades);
  double get_spread() const;

  //! Sample counting noise into each simulated measurement.
  void set_poisson(bool value);
  bool get_poisson() const;

  //! Generate episodes: features a fitter could compute, and the answers.
  /*!
      \param[in] episodes how many to play
      \param[in] seed the generator's seed; the same seed replays exactly

      Fills #get_features and #get_targets, row-major, one row per episode.
  */
  void generate(int episodes, unsigned int seed);

  int get_number_of_episodes() const;
  int get_number_of_features() const;
  int get_number_of_targets() const;
  //! Row-major, `episodes * get_number_of_features()`.
  const std::vector<double>& get_features() const;
  //! Row-major, `episodes * get_number_of_targets()`, in registry order.
  const std::vector<double>& get_targets() const;
  //! The canonical ids the targets are for, in the order they appear.
  std::vector<std::string> get_target_ids() const;

  //! Train a proposer on the generated episodes.
  /*!
      \param[in] hidden units in each hidden layer
      \param[in] epochs passes over the episodes
      \param[in] learning_rate the Adam step
      \return the trained network as `tttrlib.neural_net` JSON, which
              IMP::bff::NeuralNet reads

      Gradients come from the vendored MlpCore backward pass, so this is one
      implementation of the mathematics driven from a second place, not a
      second implementation of it.
  */
  std::string train(const std::vector<int>& hidden, int epochs,
                    double learning_rate);

  //! Mean squared error per target on the generated episodes, after training.
  std::vector<double> get_training_error() const;

  //! Ask a trained proposer where to start, given one real measurement.
  /*! \param[in] network the JSON #train returned
      \return one value per canonical parameter, in registry order, ready for
              FittingModelSearchProblem::add_structure_start. */
  std::vector<double> propose(const std::string& network) const;

  //! Simulate structural defects, fit their smaller parent, and record residuals.
  /*! Each episode draws an action uniformly, then one declared transition
      carrying it, so an action reachable from many parents does not dominate
      the labels. A structural episode samples the destination topology,
      simulates its bound measurements (with matching noise), then fits the
      transition's parent; a terminal action's episode simulates and fits the
      same topology, which teaches the policy when to stop. The input is the
      fitted topology's #get_residual_profile; the one-hot target is the
      action. Only transitions whose source and destination expose the same
      measurement curves participate, so a label always means a corrective
      structural move rather than an artefact of missing data. */
  void generate_policy(int episodes, unsigned int seed);
  int get_number_of_policy_episodes() const;
  int get_number_of_policy_features() const;
  std::vector<std::string> get_policy_action_keys() const;
  const std::vector<double>& get_policy_features() const;
  const std::vector<double>& get_policy_targets() const;

  //! Train a softmax action policy for #set_residual_action_policy.
  /*! Returns a `NeuralNet` JSON document whose input width is the residual
      profile width and whose outputs are ordered by #get_policy_action_keys. */
  /*! `seed` draws the initial weights and the split; a `validation_fraction`
      of the episodes is held out, never trained on, and scored afterwards. */
  std::string train_policy(const std::vector<int>& hidden, int epochs,
                           double learning_rate, unsigned int seed = 67890,
                           double validation_fraction = 0.0);
  //! Mean cross-entropy over the training episodes after training.
  double get_policy_training_loss() const;
  int get_number_of_policy_validation_episodes() const;
  //! Mean cross-entropy on the held-out episodes; NaN when none were held out.
  double get_policy_validation_loss() const;
  //! Share of held-out episodes whose most probable action is the label.
  double get_policy_validation_accuracy() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_MODELSEARCHSELFPLAY_H

/**
 *  \file IMP/bff/ModelSearch.h
 *  \brief Model-independent Monte Carlo tree search for fitted structures.
 *
 *  The tree knows only opaque state/action keys and scalar rewards.  The
 *  problem owns every model-specific object and performs each transition in
 *  C++; this keeps fitting data and the inner optimiser on the same side of
 *  the Python boundary.  A future adapter may therefore keep one state as a
 *  single fit and another as a heterogeneous global fit without teaching the
 *  search either representation.
 */
#ifndef IMPBFF_MODELSEARCH_H
#define IMPBFF_MODELSEARCH_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

class IMPBFFEXPORT ModelSearchConfigurationError
    : public std::domain_error {
 public:
  explicit ModelSearchConfigurationError(const std::string& what)
      : std::domain_error(what) {}
};

//! One fully evaluated model state; its key is opaque to the tree.
//! How two competing models are compared.
/*!
    Not by chi-square alone. A richer model always fits at least as well, so
    a comparison that only counts misfit always prefers the richer one --
    measured here: dropping the sample size from a description makes the
    analytical FCS family choose a spurious relaxation term on a curve built
    from one diffusing species. Both criteria below are that misfit plus a
    price for each free parameter; they differ in the price.

    The reward the search maximises is a monotone transform of the criterion
    -- `-BIC/2` or `-AIC/2` -- so maximising reward is minimising the
    criterion, and the numbers a result carries can be read as either.
*/
enum ModelSelectionCriterion {
  //! chi2 + k ln n. The price grows with the data, so on a long measurement
  //! BIC is the more reluctant of the two to buy another parameter.
  MODEL_SELECTION_BIC = 0,
  //! chi2 + 2k. A fixed price per parameter, and the more permissive.
  MODEL_SELECTION_AIC = 1
};

class IMPBFFEXPORT ModelSearchState {
 public:
  ModelSearchState();
  ModelSearchState(const std::string& key, double reward,
                      bool acceptable = false);
  ModelSearchState(const std::string& key,
                      const std::string& structure_key, double reward,
                      bool acceptable = false);
  const std::string& get_key() const;
  //! Canonical structural identity; several fitted snapshots may share it.
  const std::string& get_structure_key() const;
  double get_reward() const;
  bool get_acceptable() const;
  IMP_SHOWABLE_INLINE(ModelSearchState,
                      out << "ModelSearchState(" << key_ << ", "
                          << reward_ << ")");

 private:
  std::string key_;
  std::string structure_key_;
  double reward_;
  bool acceptable_;
};
IMP_VALUES(ModelSearchState, ModelSearchStates);

//! A structural move offered by a problem at an evaluated state.
class IMPBFFEXPORT ModelSearchAction {
 public:
  ModelSearchAction();
  ModelSearchAction(const std::string& key,
                       const std::string& predicted_state_key,
                       double prior = 1.0, bool terminal = false);
  const std::string& get_key() const;
  const std::string& get_predicted_state_key() const;
  double get_prior() const;
  bool get_terminal() const;
  IMP_SHOWABLE_INLINE(ModelSearchAction,
                      out << "ModelSearchAction(" << key_ << " -> "
                          << predicted_state_key_ << ")");

 private:
  std::string key_;
  std::string predicted_state_key_;
  double prior_;
  bool terminal_;
};
IMP_VALUES(ModelSearchAction, ModelSearchActions);

//! Model-specific topology and evaluation, implemented entirely in C++.
/*!
  `evaluate()` may return a key different from the action's predicted key.
  This is how an optimiser reports that a nominally larger model collapsed
  onto a canonical ancestor.  The tree records its reward, then makes that
  node a dead end so the collapsed structure is not searched repeatedly.
*/
class IMPBFFEXPORT ModelSearchProblem {
 public:
  virtual ~ModelSearchProblem();
  virtual ModelSearchState get_initial_state() = 0;
  virtual ModelSearchActions get_actions(
      const ModelSearchState& state) = 0;
  virtual ModelSearchState evaluate(
      const ModelSearchState& parent,
      const ModelSearchAction& action) = 0;
  //! Propagate a cooperative cancellation request into an active evaluator.
  virtual void request_cancel();
  virtual void clear_cancel();
  //! Make a cached state current after search (the root when cancelled).
  virtual void activate_state(const ModelSearchState& state);
};

//! A callback-free finite problem, useful for persisted/pre-scored graphs.
/*!
  This is also the executable contract fixture for the SWIG surface.  Real
  fitting adapters subclass #ModelSearchProblem in C++ and keep their
  heterogeneous model states behind the same opaque keys.
*/
class IMPBFFEXPORT TabularModelSearchProblem : public ModelSearchProblem {
 public:
  TabularModelSearchProblem();
  void add_state(const std::string& key, double reward,
                 bool acceptable = false);
  void set_initial_state(const std::string& key);
  void add_action(const std::string& parent_key, const std::string& action_key,
                  const std::string& result_key, double prior = 1.0,
                  bool terminal = false);
  ModelSearchState get_initial_state() override;
  ModelSearchActions get_actions(
      const ModelSearchState& state) override;
  ModelSearchState evaluate(const ModelSearchState& parent,
                               const ModelSearchAction& action) override;
  int get_number_of_evaluations() const;
  void reset_number_of_evaluations();

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

class GraphNode;
class GraphPort;

//! A live, callback-free adapter from declarative structures to FitMinimizer.
/*!
  Structures name the parameter groups that are free.  Transitions restore
  their parent's cached snapshot, apply the target fix/free mask and any seed
  for a newly enabled group, then run #FitMinimizer on the resulting free
  owner ports.  Every successful state caches its own values and fixed mask.

  The graph is shared and is therefore evaluated sequentially.  There is no
  graph clone and no hidden parallelism: each transition is a transaction over
  one graph, and the next transition begins by restoring its own parent.  On
  cancellation, non-convergence or an exception the parent snapshot is
  restored and the transition collapses to that parent.

  Ports must be scalar, unlinked canonical owners.  A caller with linked ports
  supplies each link target once; accepting followers as independent state
  would make snapshot restoration order-dependent.
*/
class IMPBFFEXPORT FittingModelSearchProblem : public ModelSearchProblem {
 public:
  FittingModelSearchProblem();
  FittingModelSearchProblem(std::shared_ptr<GraphNode> objective,
                           const std::string& residual_key = "residuals");
  ~FittingModelSearchProblem();

  void set_objective(std::shared_ptr<GraphNode> objective,
                     const std::string& residual_key = "residuals");
  std::shared_ptr<GraphNode> get_objective() const;

  //! Declare one independently fixable group and optional enable-time seeds.
  void add_parameter_group(
      const std::string& key,
      const std::vector<std::shared_ptr<GraphPort> >& ports,
      const std::vector<double>& enable_values = std::vector<double>());
  std::vector<std::string> get_parameter_group_keys() const;

  //! Declare a structure by the groups that are free in it.
  void add_structure(const std::string& key,
                     const std::vector<std::string>& free_groups);
  void set_initial_structure(const std::string& key);
  void add_action(const std::string& parent_structure,
                  const std::string& action_key,
                  const std::string& result_structure, double prior = 1.0,
                  bool terminal = false);

  //! Use this scalar objective output as reward (higher is better).
  void set_score_output(const std::string& key);
  void clear_score_output();
  const std::string& get_score_output() const;
  //! Otherwise reward is -chi2/2 - this penalty times the free-port count.
  void set_complexity_penalty(double value);
  double get_complexity_penalty() const;
  //! Optional scalar/bool output defining result.acceptable.
  void set_acceptable_output(const std::string& key);
  void clear_acceptable_output();

  ModelSearchState get_initial_state() override;
  ModelSearchActions get_actions(
      const ModelSearchState& state) override;
  ModelSearchState evaluate(const ModelSearchState& parent,
                               const ModelSearchAction& action) override;
  void request_cancel() override;
  void clear_cancel() override;
  void activate_state(const ModelSearchState& state) override;

  //! Snapshot inspection and explicit winner application.
  bool has_cached_state(const std::string& state_key) const;
  std::vector<double> get_cached_values(const std::string& state_key) const;
  std::vector<int> get_cached_fixed(const std::string& state_key) const;
  void restore_state(const std::string& state_key);
  int get_last_fit_status() const;
  const std::string& get_last_failure() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  FittingModelSearchProblem(const FittingModelSearchProblem&) = delete;
  FittingModelSearchProblem& operator=(const FittingModelSearchProblem&) = delete;
};

//! A callback-free fitting search whose structures have different graphs.
/*!
  The problem owns one canonical parameter registry.  Every structure names
  every registered parameter and must supply the exact same owner #GraphPort
  for that canonical id.  A structure may omit a parameter from its objective
  topology by marking it fixed, but it does not get a private copy.  Switching
  structures therefore selects an objective graph while values, bounds, links,
  and snapshots remain properties of one shared native parameter state.

  Each structure also declares its starting value and fixed mask.  Values of
  parameters that remain free across a transition are warm-started in place.
  Newly enabled parameters and fixed structural constants take the target
  structure's declared starting value.  No value is copied between graphs and
  no parameter is matched by a port or node name.

  The default reward is `-chi2/2`.  Optional BIC metadata changes it to
  `-(chi2 + complexity * log(effective_sample_size))/2`; an explicit scalar
  score output overrides both.  A #FitJointChiSquared is simply another
  objective graph, so single and joint fits use the same path.
*/
class IMPBFFEXPORT MultiStructureModelSearchProblem
    : public ModelSearchProblem {
 public:
  MultiStructureModelSearchProblem();
  ~MultiStructureModelSearchProblem();

  //! Register one stable canonical id and its scalar, unlinked owner port.
  void add_parameter(const std::string& canonical_id,
                     std::shared_ptr<GraphPort> owner);
  std::vector<std::string> get_parameter_ids() const;
  std::shared_ptr<GraphPort> get_parameter(
      const std::string& canonical_id) const;

  //! Register a complete native objective topology and parameter state.
  /*!
    `parameter_ids`, `parameter_ports`, `initial_values`, and `fixed_mask`
    describe the complete canonical registry. IDs may be in any order, but
    none may be duplicated or omitted. Every supplied port must be the exact
    owner already registered for its id.
  */
  void add_structure(
      const std::string& key, std::shared_ptr<GraphNode> objective,
      const std::vector<std::string>& parameter_ids,
      const std::vector<std::shared_ptr<GraphPort> >& parameter_ports,
      const std::vector<double>& initial_values,
      const std::vector<int>& fixed_mask,
      const std::string& residual_key = "residuals");
  //! Declare another starting point for a structure, tried alongside its own.
  /*!
      \param[in] structure_key the structure this start belongs to
      \param[in] initial_values one value per canonical parameter, in
      registry order, exactly as #add_structure takes them

      Every declared start is fitted and the best result kept, so a
      topology's score is the best fit it admits rather than whatever the one
      seed happened to find. Measured on a two-species FCS curve, the
      generating topology scores -915 from its declared seed and -12.6 from a
      good one -- a difference large enough to lose the model selection.

      Because the starts are *declared*, the best of them still depends only
      on the model and the data, not on the route the search took to get
      here. That is the property #set_warm_start would otherwise cost.
  */
  void add_structure_start(const std::string& structure_key,
                           const std::vector<double>& initial_values);
  std::vector<std::string> get_structure_keys() const;
  std::shared_ptr<GraphNode> get_structure_objective(
      const std::string& key) const;
  //! Retain one upstream node that belongs to a structure's complete graph.
  /*!
    Graph links retain ports while a port retains its attached node weakly.
    Registering upstream nodes here gives the problem ownership of the full
    topology. The objective passed to #add_structure is retained already.
  */
  void add_structure_node(const std::string& structure_key,
                          std::shared_ptr<GraphNode> node);

  void set_initial_structure(const std::string& key);
  void add_action(const std::string& parent_structure,
                  const std::string& action_key,
                  const std::string& result_structure, double prior = 1.0,
                  bool terminal = false);

  //! Carry a parent's fitted values into the parameters a child also frees.
  /*!
      Off by default, and that default is the important part.

      Warm starting is cheap and usually helps, but it makes a topology
      inherit whichever optimum its *route* landed in: the same structure,
      the same data and the same free parameters score differently depending
      on the order of the moves that reached them -- measured at 2528.9 in
      reward across two orderings of two commuting moves
      (okf/validation/model-search-strategy.md). A candidate that has no
      score of its own cannot be compared with another, which is the one
      thing model selection does.

      With warm starting off, every structure is fitted from the seeds it
      declares, so its score is a property of the model and the data and
      nothing else. Turn it on only where speed matters more than
      comparability -- refining a single known topology, not choosing between
      topologies.
  */
  void set_warm_start(bool value);
  bool get_warm_start() const;

  //! Use this structure's scalar output directly as reward.
  void set_structure_score_output(const std::string& structure_key,
                                  const std::string& output_key);
  void clear_structure_score_output(const std::string& structure_key);
  //! Optional scalar/bool output defining result.acceptable.
  void set_structure_acceptable_output(const std::string& structure_key,
                                       const std::string& output_key);
  void clear_structure_acceptable_output(const std::string& structure_key);
  //! How this structure is compared with the others, when it has no explicit
  //! score output.
  /*!
      \param[in] criterion BIC or AIC
      \param[in] effective_sample_size observations the fit actually used
      \param[in] complexity free parameters, counted once each

      A structure must have either this or a score output. There is
      deliberately no default: scoring by misfit alone is not a comparison
      between models, it is a preference for the larger one, and a family
      that forgot to say how it should be judged must fail rather than
      quietly overfit.
  */
  void set_structure_selection(const std::string& structure_key,
                               ModelSelectionCriterion criterion,
                               double effective_sample_size,
                               double complexity);
  void clear_structure_selection(const std::string& structure_key);

  //! Refuse a structure whose chi-square test falls below this probability.
  /*! Acceptability then means "describes the data", which is separate from
      winning the comparison. */
  void set_structure_acceptance(const std::string& structure_key,
                                double least_probability);

  //! Goodness of fit of the last evaluation: P(chi2 >= observed).
  /*! The chi-square test, which asks whether the winner describes the data
      at all -- a question no comparison between candidates answers, because
      the best of a bad family is still bad. NaN when the fit left no degrees
      of freedom. */
  double get_last_chi2_p_value() const;

  //! Reduced chi-square of the last evaluation: chi2 / (n - k - 1).
  /*! The goodness-of-fit question, which is separate from the comparison:
      a model can be the best of a family and still not describe the data.
      Near one is the expectation. */
  double get_last_reduced_chi2() const;

  ModelSearchState get_initial_state() override;
  ModelSearchActions get_actions(
      const ModelSearchState& state) override;
  ModelSearchState evaluate(const ModelSearchState& parent,
                            const ModelSearchAction& action) override;
  void request_cancel() override;
  void clear_cancel() override;
  void activate_state(const ModelSearchState& state) override;

  bool has_cached_state(const std::string& state_key) const;
  std::vector<double> get_cached_values(const std::string& state_key) const;
  std::vector<int> get_cached_fixed(const std::string& state_key) const;
  void restore_state(const std::string& state_key);
  //! Make one topology current, with its declared seeds and its fixed mask.
  /*! Evaluating a structure is not the same as selecting it: reading a
      node's output leaves the registry as it stands, while this puts the
      registry into the state the structure declares -- which is also what
      says *which* parameters that topology frees. */
  void activate_structure(const std::string& key);

  //! Record which node produces the curve compared against one measurement.
  /*! A described family knows this -- the objective bound to a measurement
      reads its model from exactly one node -- and it is what lets a family
      be run backwards, as its own generator, without anyone guessing at a
      naming convention. */
  void set_structure_curve(const std::string& structure_key,
                           const std::string& dataset_name,
                           const std::string& node_name);
  //! The measurements this structure produces a curve for.
  std::vector<std::string> get_structure_curve_datasets(
      const std::string& structure_key) const;
  //! The node producing the curve for one measurement.
  std::string get_structure_curve_node(const std::string& structure_key,
                                       const std::string& dataset_name) const;

  //! Evaluate one topology and read what a node in it produced.
  /*!
      \param[in] structure_key the topology to make current and evaluate
      \param[in] node_name the node's full name, which for a described
      family is `"<structure>.<node>"`

      The curve a topology predicts at the parameters currently in the
      registry. Fitting runs this graph already; this only makes its output
      readable, which is what turns a model family into its own data
      generator -- set the parameters, read the curve, and a fixture or a
      self-play episode has a measurement no hand-written simulator had to
      reproduce. A hand-written one is a second opinion about the model, and
      the first one is right here.
  */
  std::vector<double> get_structure_output(const std::string& structure_key,
                                           const std::string& node_name);

  const std::string& get_active_structure() const;
  std::shared_ptr<GraphNode> get_active_objective() const;
  int get_last_fit_status() const;
  const std::string& get_last_failure() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  MultiStructureModelSearchProblem(
      const MultiStructureModelSearchProblem&) = delete;
  MultiStructureModelSearchProblem& operator=(
      const MultiStructureModelSearchProblem&) = delete;
};

//! Search controls, independent of any fitting model or data family.
class IMPBFFEXPORT ModelSearchConfig {
 public:
  ModelSearchConfig();
  void set_number_of_simulations(int value);
  int get_number_of_simulations() const;
  void set_c_puct(double value);
  double get_c_puct() const;
  void set_reward_scale(double value);
  double get_reward_scale() const;
  void set_dirichlet_alpha(double value);
  double get_dirichlet_alpha() const;
  void set_dirichlet_fraction(double value);
  double get_dirichlet_fraction() const;
  void set_seed(unsigned int value);
  unsigned int get_seed() const;
  IMP_SHOWABLE_INLINE(ModelSearchConfig,
                      out << "ModelSearchConfig(" << n_simulations_
                          << " simulations)");

 private:
  int n_simulations_;
  double c_puct_;
  double reward_scale_;
  double dirichlet_alpha_;
  double dirichlet_fraction_;
  unsigned int seed_;
};
IMP_VALUES(ModelSearchConfig, ModelSearchConfigs);

//! The best evaluated state and an auditable summary of one run.
class IMPBFFEXPORT ModelSearchResult {
 public:
  ModelSearchResult();
  const ModelSearchState& get_root_state() const;
  const ModelSearchState& get_best_state() const;
  const std::vector<std::string>& get_best_path() const;
  int get_number_of_states_evaluated() const;
  int get_number_of_simulations() const;
  double get_improvement() const;
  bool get_acceptable() const;
  bool get_cancelled() const;
  IMP_SHOWABLE_INLINE(ModelSearchResult,
                      out << "ModelSearchResult(" << n_simulations_
                          << " simulations, best=" << best_state_.get_key()
                          << ")");

 private:
  friend class ModelSearch;
  ModelSearchState root_state_;
  ModelSearchState best_state_;
  std::vector<std::string> best_path_;
  int n_states_evaluated_;
  int n_simulations_;
  double improvement_;
  bool acceptable_;
  bool cancelled_;
};
IMP_VALUES(ModelSearchResult, ModelSearchResults);

//! PUCT traversal, lazy expansion, cancellation, and bookkeeping in C++.
class IMPBFFEXPORT ModelSearch {
 public:
  ModelSearch();
  explicit ModelSearch(std::shared_ptr<ModelSearchProblem> problem);
  ~ModelSearch();
  void set_problem(std::shared_ptr<ModelSearchProblem> problem);
  std::shared_ptr<ModelSearchProblem> get_problem() const;
  void set_config(const ModelSearchConfig& config);
  ModelSearchConfig get_config() const;

  //! Thread-safe cooperative cancellation, checked between simulations.
  void request_cancel();
  void clear_cancel();
  bool get_cancel_requested() const;

  //! Run from the problem's evaluated initial state.
  ModelSearchResult run();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ModelSearch(const ModelSearch&) = delete;
  ModelSearch& operator=(const ModelSearch&) = delete;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_MODELSEARCH_H

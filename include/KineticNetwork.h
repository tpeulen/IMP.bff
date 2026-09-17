/**
 *  \file IMP/bff/KineticNetwork.h
 *  \brief Kinetic schemes composed from factors: conditional intensity
 *         matrices and their amalgamation into the joint generator.
 *
 *  A kinetic scheme that is a product of processes -- a conformation crossed
 *  with a dye's photophysics, a conformation crossed with an emission mode, a
 *  donor's label states crossed with an acceptor's -- has a joint generator
 *  whose size is the product of the parts, and which is written out by hand
 *  with index arithmetic today (chisurf `burst/simulate.py`
 *  `exchange_matrix_ms`, `2 * target + mode`). This header builds it from the
 *  parts instead: each variable gets a **conditional intensity matrix** (CIM),
 *  its rates conditioned on the states of its parents, and **amalgamation**
 *  sums them into the joint generator. That is a continuous-time Bayesian
 *  network (Nodelman, Shelton & Koller, UAI 2002), harvested from aGrUM's
 *  `pyagrum.ctbn` (`CIM.py` `amalgamate`, `CTBNInference.py`
 *  `SimpleInference`; dual LGPL-3.0-or-later / MIT; chisurf
 *  `junk/aGrUM` at 9f2905b60).
 *
 *  \par What amalgamation is
 *  For variables X (parents U_X) and Y (parents U_Y), the joint rate from
 *  (x, y) to (x', y') is
 *
 *      Q_X(x -> x' | u_X)            if x' != x and y' == y,
 *      Q_Y(y -> y' | u_Y)            if x' == x and y' != y,
 *      Q_X(x -> x | u_X) + Q_Y(y -> y | u_Y)   on the diagonal,
 *      0                             if both change at once,
 *
 *  where a parent of X that is Y is read at Y's **source** state y (and vice
 *  versa). Without parents this is the Kronecker sum K_X (+) K_Y. Parents that
 *  are not amalgamated stay parents of the result, so amalgamation can be done
 *  a pair at a time, as aGrUM does. **Simultaneous changes are zero** -- the
 *  one assumption, and the one to check a scheme against: an energy transfer
 *  D*A -> DA* changes two variables in one event and does NOT factor this way.
 *
 *  \par Conventions -- the classic silent bug, so all of them here
 *  - **Matrices are `K[target, source]`**, flat row-major, columns summing to
 *    zero: `dp/dt = K p` with p a column vector. The same convention as
 *    `fcs_bunching_factor` / `fcs_saturated_curve_shape` (FCS.h),
 *    PhotophysicsTransferKinetics.h and chisurf `core/fitting/kinetics.py`.
 *    aGrUM's `CIM.toMatrix` is the **transpose** (row = from state `#i`,
 *    column = to state `#j`, rows sum to zero); tttrlib's `SimSystem` /
 *    `SimKinetics` rates are row-major source -> target, also the transpose.
 *  - **Joint state index is C-order over the variables in declaration
 *    order, the first variable most significant**: for variables with sizes
 *    (n_0, ..., n_{m-1}) and states (s_0, ..., s_{m-1}) the index is
 *    `((s_0 n_1 + s_1) n_2 + s_2) ...`, which is `numpy.ravel_multi_index`
 *    and the index order of `numpy.kron(K_0, I) + numpy.kron(I, K_1)`.
 *    aGrUM's `toMatrix` instead sorts variable names and increments the
 *    first name fastest (little-endian).
 *  - **Parent configurations are C-order over the parents** in the order
 *    they were added.
 *  - **The diagonal is derived**, never stored: a CIM holds off-diagonal
 *    rates, and every read of a diagonal returns minus the exit rate of that
 *    source under that parent configuration. aGrUM stores whatever the caller
 *    writes there (its generator writes -sum); deriving it makes a
 *    column-sum error unrepresentable.
 *
 *  \par What is here, and what is not
 *  #KineticIntensityMatrix (CIM, `extract`, `amalgamate`), #KineticNetwork
 *  (variables, arcs, rates, the joint generator, marginals), and the exact
 *  inference aGrUM's `SimpleInference` does -- the transient distribution
 *  through the matrix exponential -- plus the stationary distribution.
 *  aGrUM's forward-sampling inference is not ported: a Gillespie walk on the
 *  amalgamated generator is tttrlib `SimKinetics` (transpose the matrix).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_KINETICNETWORK_H
#define IMPBFF_KINETICNETWORK_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A conditional intensity matrix: rates of one or more variables, given parents.
/**
    Created over one variable; amalgamation produces one over several. The
    rates are stored per parent configuration as `K[target, source]` over the
    joint states of the variables (see the file's conventions).
 */
class IMPBFFEXPORT KineticIntensityMatrix {
 public:
  //! The empty CIM: amalgamating with it returns the other operand.
  KineticIntensityMatrix();

  //! One variable with \p n_states, conditioned on \p parents.
  /** All rates start at zero. \p parent_states gives each parent's number of
      states; names must be unique and a variable cannot be its own parent. */
  KineticIntensityMatrix(const std::string& variable, int n_states,
                         const std::vector<std::string>& parents =
                                 std::vector<std::string>(),
                         const std::vector<int>& parent_states =
                                 std::vector<int>());

  //! The variables, most significant first.
  std::vector<std::string> get_variables() const { return variables_; }
  std::vector<int> get_variable_states() const { return variable_states_; }
  std::vector<std::string> get_parents() const { return parents_; }
  std::vector<int> get_parent_states() const { return parent_states_; }

  //! The number of joint states, the product of the variables' states.
  int get_number_of_states() const { return n_; }
  //! The number of parent configurations, 1 without parents.
  int get_number_of_parent_configurations() const { return n_configurations_; }
  //! True when there are parents (aGrUM's `not isIM()`).
  bool get_is_conditional() const { return !parents_.empty(); }

  //! Joint state index of per-variable states (C-order).
  int get_state_index(const std::vector<int>& states) const;
  //! Per-variable states of a joint index.
  std::vector<int> get_states(int index) const;
  //! Parent configuration index of per-parent states (C-order).
  int get_parent_configuration(const std::vector<int>& parent_values) const;

  //! Set the rate from joint state \p source to \p target (off-diagonal, >= 0).
  void set_rate(int source, int target, double rate,
                int parent_configuration = 0);
  //! The rate; on the diagonal minus the exit rate of \p source.
  double get_rate(int source, int target, int parent_configuration = 0) const;

  //! Set every rate of one parent configuration from `K[target, source]`.
  /** Flat row-major, `n * n` values; the diagonal is ignored (derived). */
  void set_matrix(const std::vector<double>& matrix,
                  int parent_configuration = 0);
  //! `K[target, source]` of one parent configuration, diagonal derived.
  void get_matrix(int parent_configuration, double** out_view,
                  int* n_out_view) const;

  //! aGrUM's `extract`: fix some parents, keep the rest as parents.
  /** Every name must be a parent of this CIM. Fixing all parents gives an
      unconditional CIM whose matrix is that configuration's. */
  KineticIntensityMatrix extract(const std::vector<std::string>& parents,
                                 const std::vector<int>& values) const;

  //! aGrUM's `amalgamate` (spelled `*` there): the joint CIM of two.
  /** The variables are this CIM's followed by \p other 's (so this one is
      more significant); the two must not share a variable. A parent of one
      that is a variable of the other is resolved at the other's source
      state; the remaining parents, this CIM's first, stay parents. */
  KineticIntensityMatrix amalgamate(const KineticIntensityMatrix& other) const;

  //! The joint generator: `K[target, source]` of an unconditional CIM.
  /** Throws when parents remain -- as aGrUM's `toMatrix` does. */
  void get_generator(double** out_view, int* n_out_view) const;

  //! Marginal distribution of one variable from a joint distribution.
  void get_marginal(const std::vector<double>& joint,
                    const std::string& variable, double** out_view,
                    int* n_out_view) const;

 private:
  int variable_position(const std::string& name) const;
  int parent_position(const std::string& name) const;
  void check_configuration(int parent_configuration) const;
  void check_state(int state) const;
  std::size_t at(int configuration, int target, int source) const;

  std::vector<std::string> variables_;
  std::vector<int> variable_states_;
  std::vector<std::string> parents_;
  std::vector<int> parent_states_;
  int n_ = 1;
  int n_configurations_ = 1;
  //! Off-diagonal rates, `[configuration][target][source]`; diagonal kept 0.
  std::vector<double> rates_;
};

//! A continuous-time Bayesian network: variables, arcs, one CIM per variable.
/**
    aGrUM's `CTBN`. Arcs may form cycles (a CTBN allows them: a conformation
    that gates a dye's quenching, and a dye state that gates a
    photo-isomerisation). The joint generator amalgamates the CIMs in
    declaration order, so the first variable added is the most significant
    digit of the joint index.

    A scheme with two clocks -- spontaneous rates and excitation-scaled rates,
    `K = K_dark + k_exc K_exc` in FCS.h, `k_nrad` / `k_rad` in tttrlib's
    simulator -- is two networks over the same variables: amalgamation is
    linear in the rates, so the joint generators add.
 */
class IMPBFFEXPORT KineticNetwork {
 public:
  KineticNetwork();

  //! Add a variable; returns its position. Names must be unique.
  int add_variable(const std::string& name, int n_states);
  //! Make \p parent condition \p child 's rates.
  /** Rates already set on \p child are copied to every state of the new
      parent, so a rate set before the arc holds for all of them. */
  void add_arc(const std::string& parent, const std::string& child);

  //! Set the rate of \p variable from \p source to \p target, given its parents.
  /** \p parent_values holds one state per parent, in the order the arcs were
      added (see get_parents); empty is allowed only without parents. */
  void set_rate(const std::string& variable, int source, int target,
                double rate,
                const std::vector<int>& parent_values = std::vector<int>());
  double get_rate(const std::string& variable, int source, int target,
                  const std::vector<int>& parent_values =
                          std::vector<int>()) const;

  std::vector<std::string> get_variables() const;
  std::vector<int> get_variable_states() const;
  //! The parents of \p variable, in arc order.
  std::vector<std::string> get_parents(const std::string& variable) const;
  //! The CIM of one variable.
  KineticIntensityMatrix get_intensity_matrix(const std::string& variable) const;
  //! Replace the CIM of one variable; its variable and parents must match.
  void set_intensity_matrix(const KineticIntensityMatrix& cim);

  //! The number of joint states.
  int get_number_of_states() const;
  //! The amalgamation of every CIM, in declaration order.
  KineticIntensityMatrix get_joint() const;
  //! The joint generator `K[target, source]`, flat row-major.
  void get_generator(double** out_view, int* n_out_view) const;

  int get_state_index(const std::vector<int>& states) const;
  std::vector<int> get_states(int index) const;
  //! Marginal distribution of one variable from a joint distribution.
  void get_marginal(const std::vector<double>& joint,
                    const std::string& variable, double** out_view,
                    int* n_out_view) const;

 private:
  int position(const std::string& name) const;
  std::vector<KineticIntensityMatrix> cims_;
};

//! The stationary distribution of a generator `K[target, source]`.
/**
    Solves `K p = 0` with `sum(p) = 1`. Throws when the stationary
    distribution is not unique (more than one closed class: the rank of K is
    below n - 1).

    \param[in] generator flat row-major `n * n`; the diagonal is ignored and
               derived from the columns
    \param[out] out_view,n_out_view the n populations
 */
IMPBFFEXPORT void kinetic_stationary_distribution(
        const std::vector<double>& generator, double** out_view,
        int* n_out_view);

//! The distribution at \p time: `exp(K t) p0` for `K[target, source]`.
/**
    aGrUM's `SimpleInference.makeInference(t)` with a given initial
    distribution instead of the uniform one. The matrix exponential is Eigen's
    scaling and squaring with Pade approximants, so stiff schemes (nanosecond
    decay beside second-scale exchange) need no step control.

    \param[in] generator flat row-major `n * n`; the diagonal is derived
    \param[in] p0 the n initial populations
    \param[in] time in the reciprocal of the rate unit, >= 0
    \param[out] out_view,n_out_view the n populations at \p time
 */
IMPBFFEXPORT void kinetic_transient_distribution(
        const std::vector<double>& generator, const std::vector<double>& p0,
        double time, double** out_view, int* n_out_view);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_KINETICNETWORK_H */

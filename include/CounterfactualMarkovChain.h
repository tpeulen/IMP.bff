/**
 *  \file IMP/bff/CounterfactualMarkovChain.h
 *  \brief Counterfactual trajectories of a discrete Markov chain
 *         (Gumbel-max structural causal model).
 *
 *  A Markov chain observed in frames -- a conformational trace from smFRET
 *  TIRF, an HMM/H2MM state path -- is written as a structural causal model
 *  (Oberst & Sontag, ICML 2019): the next state is
 *
 *      s_{t+1} = argmax_k ( log P[s_t, k] + g_{t,k} ),   g_{t,k} ~ Gumbel(0, 1),
 *
 *  with one Gumbel vector g_t per step that does not depend on the current
 *  state. That noise is what the factual and the counterfactual world share.
 *
 *  - **abduction**: given the observed step s_t -> s_{t+1} under the factual
 *    matrix, g_t is drawn from its posterior by top-down sampling: the winner's
 *    value from Gumbel(logsumexp), the others truncated below it. A state the
 *    factual row makes impossible keeps its prior noise, since the observation
 *    says nothing about it.
 *  - **action**: from a chosen step on, the transition matrix is replaced -- a
 *    ligand that did not bind, a mutation, an intermediate that is blocked.
 *  - **prediction**: the chain is replayed with the abducted noise.
 *
 *  A chain observed in frames of a continuous-time process is exactly a
 *  discrete chain with P = expm(K dt), so the construction is exact for framed
 *  data. Typical questions: "would this molecule have closed without the
 *  ligand?" (the probability of necessity, per binding event), "would it have
 *  reached B had the intermediate been blocked?".
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_COUNTERFACTUALMARKOVCHAIN_H
#define IMPBFF_COUNTERFACTUALMARKOVCHAIN_H

#include <IMP/bff/bff_config.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Gumbel-max counterfactuals for a discrete-time Markov chain.
class IMPBFFEXPORT CounterfactualMarkovChain {
 public:
  //! \param[in] transitions row-major (n_states, n_states) factual transition matrix
  CounterfactualMarkovChain(const std::vector<double>& transitions, int n_states);

  int get_number_of_states() const;

  //! Counterfactual trajectories of an observed one.
  /*!
      \param[in] trajectory the observed states s_0 .. s_T
      \param[in] counterfactual_transitions row-major matrix used from `from_step` on
      \param[in] n_samples abducted noise draws
      \param[in] seed random seed
      \param[in] from_step the counterfactual matrix governs steps from_step -> from_step+1
                 onwards; earlier steps follow the observed trajectory
      \return row-major (n_samples, T + 1) states
  */
  std::vector<int> get_counterfactual_trajectories(
      const std::vector<int>& trajectory,
      const std::vector<double>& counterfactual_transitions, int n_samples,
      unsigned int seed, int from_step = 0) const;

  //! Fraction of counterfactual trajectories in each state at each step,
  //! row-major (T + 1, n_states).
  std::vector<double> get_counterfactual_occupancy(
      const std::vector<int>& trajectory,
      const std::vector<double>& counterfactual_transitions, int n_samples,
      unsigned int seed, int from_step = 0) const;

  //! Probability that the last state would have been `state` in the counterfactual world.
  double get_counterfactual_probability(
      const std::vector<int>& trajectory,
      const std::vector<double>& counterfactual_transitions, int state, int n_samples,
      unsigned int seed, int from_step = 0) const;

 private:
  void check_matrix(const std::vector<double>& m, const char* what) const;
  int n_;
  std::vector<double> log_p_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_COUNTERFACTUALMARKOVCHAIN_H

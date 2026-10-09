/**
 *  \file IMP/bff/CausalLinearGaussian.h
 *  \brief A linear-Gaussian structural causal model: interventions,
 *         abduction and counterfactuals in closed form.
 *
 *  Every variable has one structural equation
 *
 *      X_i = sum_j w_ij X_j + U_i,     U_i ~ N(mu_i, sd_i^2) independent,
 *
 *  with the arrows j -> i acyclic. A variable without parents is exogenous: its
 *  value is its noise. The three rungs of Pearl's ladder are then linear
 *  algebra on the map X = A U + a:
 *
 *  - **association**: the joint Gaussian of X, conditioned on observations;
 *  - **intervention**, do(X_k = v): the equation of X_k is replaced by the
 *    constant v (its incoming arrows and its noise are cut);
 *  - **counterfactual**, "this unit showed y; what would it have shown under
 *    do(X_k = v)?": *abduction* conditions the noise U on the observation,
 *    *action* replaces the equations, *prediction* pushes the abducted noise
 *    through the changed model. The unit's noise is shared by both worlds,
 *    which is the twin network of InferenceFactorGraph::get_twin in closed form.
 *
 *  Natural direct and indirect effects (Pearl 2001) are nested
 *  counterfactuals -- Y(t1, M(t0)) mixes the world where the treatment is t1
 *  with the mediator value of the world where it is t0 -- and are computed
 *  here exactly, for the population or for one observed unit.
 *
 *  The functional form decides counterfactual answers: two models with the same
 *  arrows and the same interventional distributions can disagree about one
 *  unit. A restraint on a counterfactual is therefore an assumption, and
 *  belongs in a model as a prior, never as a likelihood.
 *
 *  Zero noise widths are allowed (a deterministic equation); conditioning uses
 *  a pseudo-inverse, so observations that are consistent but redundant are
 *  fine. See okf/counterfactuals.md.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_CAUSALLINEARGAUSSIAN_H
#define IMPBFF_CAUSALLINEARGAUSSIAN_H

#include <IMP/bff/bff_config.h>

#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A Gaussian over named variables: a world of a causal model.
/*! `covariance` is row-major, `names.size()` squared. */
struct IMPBFFEXPORT CausalGaussianWorld {
  std::vector<std::string> names;
  std::vector<double> mean;
  std::vector<double> covariance;
  //! Mean of one variable; throws for an unknown name.
  double get_mean(const std::string& name) const;
  //! Standard deviation of one variable; throws for an unknown name.
  double get_sd(const std::string& name) const;
};

//! A linear-Gaussian structural causal model with closed-form counterfactuals.
class IMPBFFEXPORT CausalLinearGaussian {
 public:
  CausalLinearGaussian();

  //! Add a variable X = (sum of parents) + U with U ~ N(noise_mean, noise_sd^2).
  void add_variable(const std::string& name, double noise_mean = 0.0,
                    double noise_sd = 1.0);
  //! Add the arrow parent -> child with a weight. Throws if it closes a cycle.
  void add_edge(const std::string& parent, const std::string& child, double weight);

  unsigned int get_number_of_variables() const;
  //! Variable names in the order they were added.
  std::vector<std::string> get_variable_names() const;
  //! The parents of a variable.
  std::vector<std::string> get_parents(const std::string& name) const;

  //! The distribution of the world do(do_names = do_values) (empty: observational).
  CausalGaussianWorld get_interventional(
      const std::vector<std::string>& do_names = std::vector<std::string>(),
      const std::vector<double>& do_values = std::vector<double>()) const;

  //! The observational distribution conditioned on observed values (association).
  CausalGaussianWorld get_conditional(const std::vector<std::string>& obs_names,
                                      const std::vector<double>& obs_values) const;

  //! Abduction: the posterior of every variable's noise U given observations.
  /*! The returned world is over the noise terms, named "U[<variable>]". */
  CausalGaussianWorld get_abducted_noise(const std::vector<std::string>& obs_names,
                                         const std::vector<double>& obs_values) const;

  //! The counterfactual world: abduct from the observations, then do(), then predict.
  /*! With no observations this is the interventional distribution; with no
      intervention it is the conditional one. */
  CausalGaussianWorld get_counterfactual(const std::vector<std::string>& obs_names,
                                         const std::vector<double>& obs_values,
                                         const std::vector<std::string>& do_names,
                                         const std::vector<double>& do_values) const;

  //! Natural effects of a treatment on an outcome through mediators.
  /*! Returns {total, natural direct, natural indirect} effect of changing the
      treatment from t0 to t1:
        total = E[Y(t1)] - E[Y(t0)],
        NDE   = E[Y(t1, M(t0))] - E[Y(t0)],
        NIE   = E[Y(t1)] - E[Y(t1, M(t0))],
      so total = NDE + NIE. Expectations are over the noise prior, or -- when
      observations are given -- over the abducted noise of that unit. */
  std::vector<double> get_natural_effects(
      const std::string& treatment, const std::vector<std::string>& mediators,
      const std::string& outcome, double t0, double t1,
      const std::vector<std::string>& obs_names = std::vector<std::string>(),
      const std::vector<double>& obs_values = std::vector<double>()) const;

  //! Draw n samples of the world do(...), row-major (n, number of variables).
  std::vector<double> get_samples(
      unsigned int n, unsigned int seed,
      const std::vector<std::string>& do_names = std::vector<std::string>(),
      const std::vector<double>& do_values = std::vector<double>()) const;

 private:
  int index_of(const std::string& name) const;
  std::vector<std::string> names_;
  std::map<std::string, int> index_;
  std::vector<double> noise_mean_, noise_sd_;
  std::vector<std::vector<std::pair<int, double> > > parents_;  // child -> (parent, weight)
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_CAUSALLINEARGAUSSIAN_H

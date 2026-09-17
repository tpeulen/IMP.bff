/**
 *  \file IMP/bff/InferenceGaussianElimination.h
 *  \brief Exact linear-Gaussian inference over an InferenceFactorGraph:
 *         variable elimination on canonical-form factors.
 *
 *  Give each factor of an InferenceFactorGraph a Gaussian in canonical form
 *  (InferenceCanonicalForm.h), enter evidence, and ask for the posterior of
 *  any set of variables. The answer is exact for the product of those
 *  Gaussians -- nothing is sampled and nothing is re-fitted: evidence is a
 *  slice of every factor, eliminating a variable is the Schur complement of
 *  the product of the factors it appears in, and the cost is set by the
 *  graph's treewidth in the variables' dimensions, not by their number.
 *
 *  The algorithm is aGrUM's `CLGVariableElimination.canonicalPosterior`
 *  (pyAgrum `clg/variableElimination.py`, dual LGPL-3.0-or-later OR MIT):
 *  reduce every factor by the evidence, eliminate the non-targets in the
 *  graph's elimination order, multiply what is left, and -- for a normalised
 *  posterior -- divide by the same list with the targets eliminated too. The
 *  default order is aGrUM's default triangulation (the "weighted" heuristic of
 *  InferenceFactorGraph::get_elimination_order).
 *
 *  One thing is added: **relevance pruning**, which aGrUM's CLG code does not
 *  do. A factor that reaches no target except through observed variables
 *  multiplies the normalised posterior by a constant, so it is dropped before
 *  any algebra. That is exact, and it is the undirected form of what aGrUM's
 *  Bayes-ball does on a DAG; Bayes-ball itself does not apply, because it
 *  also removes barren nodes, which is exact only for normalised conditionals,
 *  and a likelihood factor is not one.
 *
 *  What makes the answer Gaussian is the factors, not this class: a
 *  fluorescence likelihood is Gaussian exactly in the parameters it is linear
 *  in, and approximately (Laplace) near an optimum in the rest. See
 *  okf/prds/prd-151.md.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_INFERENCEGAUSSIANELIMINATION_H
#define IMPBFF_INFERENCEGAUSSIANELIMINATION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/InferenceCanonicalForm.h>
#include <IMP/bff/InferenceFactorGraph.h>

#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Variable elimination over an InferenceFactorGraph with Gaussian factors.
/*!
    The graph fixes the structure: which variables exist, how many numbers
    each holds, and which variables each factor may read. A factor's canonical
    form may read fewer variables than the factor's scope, never more.
*/
class IMPBFFEXPORT InferenceGaussianElimination {
 public:
  //! Bind to a graph; the graph is copied, so later edits to it are not seen.
  explicit InferenceGaussianElimination(const InferenceFactorGraph& graph);

  //! The graph the elimination runs over.
  InferenceFactorGraph get_graph() const { return graph_; }

  //! Give a factor its Gaussian.
  /*!
      \throw ValueException when the factor is unknown, the form reads a
             variable outside the factor's scope, or a variable's size
             disagrees with the graph
  */
  void set_factor(const std::string& factor_key,
                  const InferenceCanonicalForm& form);
  //! Whether a factor has been given a Gaussian.
  bool has_factor(const std::string& factor_key) const;
  //! The Gaussian of a factor. \throw ValueException when it has none
  InferenceCanonicalForm get_factor(const std::string& factor_key) const;

  //! Observe a variable (as many numbers as it holds).
  void set_evidence(const std::string& variable,
                    const std::vector<double>& value);
  //! Forget one observation.
  void erase_evidence(const std::string& variable);
  //! Forget every observation.
  void erase_all_evidence();
  //! Whether a variable is observed.
  bool has_evidence(const std::string& variable) const;
  //! Number of observed variables.
  unsigned int get_number_of_evidence() const {
    return static_cast<unsigned int>(evidence_.size());
  }

  //! The factors a normalised posterior of \p targets depends on.
  /*!
      Those whose scope reaches a target through unobserved variables; every
      other factor is a constant once the evidence is in. Graph order.
  */
  std::vector<std::string> get_relevant_factors(
      const std::vector<std::string>& targets) const;

  //! The posterior of \p targets given the evidence, in canonical form.
  /*!
      \param[in] targets unobserved variables; the result is ordered as given
      \param[in] normalized divide by the total mass (aGrUM's default); when
                 false the form's #InferenceCanonicalForm::get_log_normalizer
                 is the log evidence, and every factor takes part
      \param[in] heuristic elimination order, as
                 InferenceFactorGraph::get_elimination_order
      \throw ValueException on an observed or unknown target, a relevant
             factor without a Gaussian, or a posterior that cannot be
             normalised because a direction of it is unconstrained
  */
  InferenceCanonicalForm get_posterior(
      const std::vector<std::string>& targets, bool normalized = true,
      const std::string& heuristic = "weighted") const;

  //! log of the integral of the product of all factors given the evidence.
  /*!
      For normalised factors this is `log p(evidence)`; for a posterior built
      from a likelihood and priors it is the log marginal likelihood.
  */
  double get_log_evidence(const std::string& heuristic = "weighted") const;

 private:
  InferenceFactorGraph graph_;
  std::map<std::string, InferenceCanonicalForm> forms_;
  std::map<std::string, std::vector<double> > evidence_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INFERENCEGAUSSIANELIMINATION_H

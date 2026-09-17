/**
 *  \file InferenceGaussianElimination.cpp
 *  \brief Variable elimination on Gaussian factors
 *         (see InferenceGaussianElimination.h).
 *
 *  Line references are to ../chisurf/junk/aGrUM at 9f2905b60,
 *  wrappers/pyagrum/pyLibs/clg/variableElimination.py.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/InferenceGaussianElimination.h>
#include <IMP/bff/IMPCompatibility.h>

#include <algorithm>
#include <set>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

InferenceGaussianElimination::InferenceGaussianElimination(
    const InferenceFactorGraph& graph)
    : graph_(graph) {}

void InferenceGaussianElimination::set_factor(
    const std::string& factor_key, const InferenceCanonicalForm& form) {
  const std::vector<std::string> keys = graph_.get_factor_keys();
  if (std::find(keys.begin(), keys.end(), factor_key) == keys.end()) {
    IMP_THROW("no factor '" << factor_key << "' in the graph",
              IMP::ValueException);
  }
  const std::vector<std::string> scope = graph_.variables_of(factor_key);
  const std::vector<std::string> names = form.get_names();
  const std::vector<int> sizes = form.get_sizes();
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (std::find(scope.begin(), scope.end(), names[i]) == scope.end()) {
      IMP_THROW("factor '" << factor_key << "' does not read '" << names[i]
                           << "', so its Gaussian cannot either",
                IMP::ValueException);
    }
    if (graph_.get_variable_size(names[i]) != sizes[i]) {
      IMP_THROW("variable '" << names[i] << "' holds "
                             << graph_.get_variable_size(names[i])
                             << " numbers in the graph, " << sizes[i]
                             << " in the Gaussian of '" << factor_key << "'",
                IMP::ValueException);
    }
  }
  forms_.erase(factor_key);
  forms_.insert(std::make_pair(factor_key, form));
}

bool InferenceGaussianElimination::has_factor(
    const std::string& factor_key) const {
  return forms_.count(factor_key) > 0;
}

InferenceCanonicalForm InferenceGaussianElimination::get_factor(
    const std::string& factor_key) const {
  auto it = forms_.find(factor_key);
  if (it == forms_.end()) {
    IMP_THROW("factor '" << factor_key << "' has no Gaussian",
              IMP::ValueException);
  }
  return it->second;
}

void InferenceGaussianElimination::set_evidence(
    const std::string& variable, const std::vector<double>& value) {
  if (graph_.index_of(variable) < 0) {
    IMP_THROW("no variable '" << variable << "' in the graph",
              IMP::ValueException);
  }
  if (int(value.size()) != graph_.get_variable_size(variable)) {
    IMP_THROW("variable '" << variable << "' holds "
                           << graph_.get_variable_size(variable)
                           << " numbers, the evidence gives " << value.size(),
              IMP::ValueException);
  }
  evidence_[variable] = value;
}

void InferenceGaussianElimination::erase_evidence(const std::string& variable) {
  evidence_.erase(variable);
}

void InferenceGaussianElimination::erase_all_evidence() { evidence_.clear(); }

bool InferenceGaussianElimination::has_evidence(
    const std::string& variable) const {
  return evidence_.count(variable) > 0;
}

std::vector<std::string> InferenceGaussianElimination::get_relevant_factors(
    const std::vector<std::string>& targets) const {
  // Breadth-first from the targets through unobserved variables: a factor is
  // relevant when its scope holds a variable reached that way. Evidence
  // separates -- conditioning on a separator makes the two sides independent,
  // so what is beyond it only rescales the posterior.
  std::set<std::string> reached;
  std::vector<std::string> frontier;
  for (const auto& t : targets) {
    if (graph_.index_of(t) < 0) {
      IMP_THROW("no variable '" << t << "' in the graph", IMP::ValueException);
    }
    if (!evidence_.count(t) && reached.insert(t).second) frontier.push_back(t);
  }
  std::set<std::string> relevant;
  while (!frontier.empty()) {
    const std::string v = frontier.back();
    frontier.pop_back();
    for (const auto& f : graph_.factors_of(v)) {
      if (!relevant.insert(f).second) continue;
      for (const auto& u : graph_.variables_of(f)) {
        if (evidence_.count(u)) continue;
        if (reached.insert(u).second) frontier.push_back(u);
      }
    }
  }
  std::vector<std::string> out;
  for (const auto& f : graph_.get_factor_keys())
    if (relevant.count(f)) out.push_back(f);
  return out;
}

namespace {

//! The list elimination of variableElimination.py 227-259, in place.
void inference_gaussian_eliminate(std::vector<InferenceCanonicalForm>* forms,
                                  const std::string& variable) {
  std::vector<InferenceCanonicalForm> containing;
  std::vector<InferenceCanonicalForm> rest;
  for (auto& f : *forms) {
    if (f.has_variable(variable))
      containing.push_back(f);
    else
      rest.push_back(f);
  }
  if (containing.empty()) return;
  InferenceCanonicalForm product = containing.front();
  for (std::size_t i = 1; i < containing.size(); ++i)
    product = product.product(containing[i]);
  rest.push_back(product.marginalize(std::vector<std::string>{variable}));
  forms->swap(rest);
}

InferenceCanonicalForm inference_gaussian_product(
    const std::vector<InferenceCanonicalForm>& forms) {
  InferenceCanonicalForm out;
  for (const auto& f : forms) out = out.product(f);
  return out;
}

}  // namespace

InferenceCanonicalForm InferenceGaussianElimination::get_posterior(
    const std::vector<std::string>& targets, bool normalized,
    const std::string& heuristic) const {
  // variableElimination.py 140-145: an observed target is refused.
  std::set<std::string> target_set;
  for (const auto& t : targets) {
    if (graph_.index_of(t) < 0) {
      IMP_THROW("no variable '" << t << "' in the graph", IMP::ValueException);
    }
    if (evidence_.count(t)) {
      IMP_THROW("the variable " << t << " is observed", IMP::ValueException);
    }
    if (!target_set.insert(t).second) {
      IMP_THROW("target '" << t << "' is named twice", IMP::ValueException);
    }
  }

  // Which factors take part: all of them for an unnormalised posterior, whose
  // mass is the evidence; only the relevant ones for a normalised one.
  const std::vector<std::string> factors =
      normalized ? get_relevant_factors(targets) : graph_.get_factor_keys();
  std::vector<std::string> ev_names;
  std::vector<double> ev_values;
  std::vector<int> ev_sizes;
  for (const auto& e : evidence_) {
    ev_names.push_back(e.first);
    ev_sizes.push_back(static_cast<int>(e.second.size()));
    ev_values.insert(ev_values.end(), e.second.begin(), e.second.end());
  }
  // _sum_product_ve 216-219: every factor reduced by the evidence.
  std::vector<InferenceCanonicalForm> forms;
  std::vector<std::string> missing;
  for (const auto& f : factors) {
    auto it = forms_.find(f);
    if (it == forms_.end()) {
      missing.push_back(f);
      continue;
    }
    forms.push_back(it->second.reduce(ev_names, ev_values, ev_sizes));
  }
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "no Gaussian for factor(s)";
    for (const auto& m : missing) msg << " '" << m << "'";
    msg << ", which the posterior depends on";
    IMP_THROW(msg.str(), IMP::ValueException);
  }

  // canonicalPosterior 150-161: the order, split into removed and kept.
  const std::vector<std::string> order = graph_.get_elimination_order(heuristic);
  std::vector<std::string> kept;
  for (const auto& v : order) {
    if (target_set.count(v)) {
      kept.push_back(v);
    } else if (!evidence_.count(v)) {
      inference_gaussian_eliminate(&forms, v);
    }
  }

  // canonicalPosterior 163-164. A target no factor reads is flat: it enters
  // with K = 0, which is the honest answer, and the form says so.
  std::vector<int> target_sizes;
  for (const auto& t : targets)
    target_sizes.push_back(graph_.get_variable_size(t));
  InferenceCanonicalForm posterior =
      inference_gaussian_product(forms).extend(targets, target_sizes);
  if (normalized) {
    // canonicalPosterior 165-167: divide by the targets eliminated as well.
    for (const auto& v : kept) inference_gaussian_eliminate(&forms, v);
    posterior = posterior.divide(inference_gaussian_product(forms));
  }
  return posterior.marginal(targets);
}

double InferenceGaussianElimination::get_log_evidence(
    const std::string& heuristic) const {
  return get_posterior(std::vector<std::string>(), false, heuristic)
      .get_log_constant();
}

IMPBFF_END_NAMESPACE

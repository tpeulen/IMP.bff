/**
 *  \file CounterfactualMarkovChain.cpp
 *  \brief Gumbel-max counterfactuals for a Markov chain
 *         (see CounterfactualMarkovChain.h).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/CounterfactualMarkovChain.h>
#include <IMP/bff/IMPCompatibility.h>

#include <cmath>
#include <limits>
#include <random>

IMPBFF_BEGIN_NAMESPACE

namespace {

const double kNegInf = -std::numeric_limits<double>::infinity();

double gumbel(std::mt19937_64& rng, double location) {
  std::uniform_real_distribution<double> u(std::nextafter(0.0, 1.0), 1.0);
  return location - std::log(-std::log(u(rng)));
}

std::vector<double> log_matrix(const std::vector<double>& p) {
  std::vector<double> out(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) out[i] = p[i] > 0.0 ? std::log(p[i]) : kNegInf;
  return out;
}

}  // namespace

CounterfactualMarkovChain::CounterfactualMarkovChain(const std::vector<double>& transitions,
                                                     int n_states)
    : n_(n_states) {
  if (n_states < 2) IMP_THROW("CounterfactualMarkovChain: need at least two states", IMP::ValueException);
  check_matrix(transitions, "CounterfactualMarkovChain");
  log_p_ = log_matrix(transitions);
}

int CounterfactualMarkovChain::get_number_of_states() const { return n_; }

void CounterfactualMarkovChain::check_matrix(const std::vector<double>& m, const char* what) const {
  if (m.size() != static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_)) {
    IMP_THROW(what << ": the transition matrix needs " << n_ << " x " << n_ << " entries",
              IMP::ValueException);
  }
  for (int i = 0; i < n_; ++i) {
    double row = 0.0;
    for (int j = 0; j < n_; ++j) {
      const double v = m[static_cast<std::size_t>(i * n_ + j)];
      if (!(v >= 0.0)) IMP_THROW(what << ": transition probabilities must be >= 0", IMP::ValueException);
      row += v;
    }
    if (std::abs(row - 1.0) > 1e-6) {
      IMP_THROW(what << ": row " << i << " sums to " << row << ", not 1", IMP::ValueException);
    }
  }
}

std::vector<int> CounterfactualMarkovChain::get_counterfactual_trajectories(
    const std::vector<int>& trajectory, const std::vector<double>& counterfactual_transitions,
    int n_samples, unsigned int seed, int from_step) const {
  check_matrix(counterfactual_transitions, "get_counterfactual_trajectories");
  const int T = static_cast<int>(trajectory.size()) - 1;
  if (T < 0) IMP_THROW("get_counterfactual_trajectories: empty trajectory", IMP::ValueException);
  if (n_samples < 1) IMP_THROW("get_counterfactual_trajectories: need n_samples >= 1", IMP::ValueException);
  if (from_step < 0 || from_step > T) {
    IMP_THROW("get_counterfactual_trajectories: from_step outside [0, " << T << "]", IMP::ValueException);
  }
  for (int s : trajectory) {
    if (s < 0 || s >= n_) IMP_THROW("get_counterfactual_trajectories: state " << s << " out of range", IMP::ValueException);
  }
  for (int t = 0; t < T; ++t) {
    if (log_p_[static_cast<std::size_t>(trajectory[t] * n_ + trajectory[t + 1])] == kNegInf) {
      IMP_THROW("get_counterfactual_trajectories: the step " << trajectory[t] << " -> "
                    << trajectory[t + 1] << " at " << t << " is impossible under the factual matrix",
                IMP::ValueException);
    }
  }
  const std::vector<double> log_q = log_matrix(counterfactual_transitions);
  std::mt19937_64 rng(seed);
  std::vector<int> out(static_cast<std::size_t>(n_samples) * static_cast<std::size_t>(T + 1));
  std::vector<double> noise(static_cast<std::size_t>(n_));
  for (int k = 0; k < n_samples; ++k) {
    int* row = out.data() + static_cast<std::size_t>(k) * (T + 1);
    for (int t = 0; t <= from_step; ++t) row[t] = trajectory[static_cast<std::size_t>(t)];
    for (int t = from_step; t < T; ++t) {
      // Abduction: g_t given the observed factual step trajectory[t] -> trajectory[t+1].
      const double* lp = log_p_.data() + static_cast<std::size_t>(trajectory[t] * n_);
      const int won = trajectory[static_cast<std::size_t>(t + 1)];
      double lse = kNegInf;
      for (int j = 0; j < n_; ++j) {
        if (lp[j] == kNegInf) continue;
        lse = lse == kNegInf ? lp[j] : std::max(lse, lp[j]) + std::log1p(std::exp(-std::abs(lse - lp[j])));
      }
      const double top = gumbel(rng, lse);
      for (int j = 0; j < n_; ++j) {
        if (j == won) {
          noise[static_cast<std::size_t>(j)] = top - lp[j];
        } else if (lp[j] == kNegInf) {
          noise[static_cast<std::size_t>(j)] = gumbel(rng, 0.0);  // unconstrained by the observation
        } else {
          const double g = gumbel(rng, lp[j]);
          const double truncated = -std::log(std::exp(-top) + std::exp(-g));  // below the winner
          noise[static_cast<std::size_t>(j)] = truncated - lp[j];
        }
      }
      // Action and prediction: the counterfactual matrix, the same noise.
      const double* lq = log_q.data() + static_cast<std::size_t>(row[t] * n_);
      int best = 0;
      double best_value = kNegInf;
      for (int j = 0; j < n_; ++j) {
        const double v = lq[j] + noise[static_cast<std::size_t>(j)];
        if (v > best_value) {
          best_value = v;
          best = j;
        }
      }
      row[t + 1] = best;
    }
  }
  return out;
}

std::vector<double> CounterfactualMarkovChain::get_counterfactual_occupancy(
    const std::vector<int>& trajectory, const std::vector<double>& counterfactual_transitions,
    int n_samples, unsigned int seed, int from_step) const {
  const std::vector<int> paths =
      get_counterfactual_trajectories(trajectory, counterfactual_transitions, n_samples, seed, from_step);
  const std::size_t L = trajectory.size();
  std::vector<double> occ(L * static_cast<std::size_t>(n_), 0.0);
  for (int k = 0; k < n_samples; ++k) {
    for (std::size_t t = 0; t < L; ++t) {
      occ[t * static_cast<std::size_t>(n_) + static_cast<std::size_t>(paths[static_cast<std::size_t>(k) * L + t])] += 1.0 / n_samples;
    }
  }
  return occ;
}

double CounterfactualMarkovChain::get_counterfactual_probability(
    const std::vector<int>& trajectory, const std::vector<double>& counterfactual_transitions,
    int state, int n_samples, unsigned int seed, int from_step) const {
  if (state < 0 || state >= n_) IMP_THROW("get_counterfactual_probability: state out of range", IMP::ValueException);
  const std::vector<double> occ =
      get_counterfactual_occupancy(trajectory, counterfactual_transitions, n_samples, seed, from_step);
  return occ[(trajectory.size() - 1) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(state)];
}

IMPBFF_END_NAMESPACE

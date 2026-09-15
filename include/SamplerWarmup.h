/**
 * \file IMP/bff/SamplerWarmup.h
 * \brief Warm-up adaptation shared by bff's samplers: the windowed schedule and dual averaging.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * One implementation of each, used by every kernel that adapts (PRD-147 step 3):
 * - **warmup_windows**: Stan's windowed schedule -- an initial buffer (75), slow windows starting at 25
 *   and doubling, a terminal buffer (50), the last window stretched to the terminal buffer. Checked
 *   against CmdStan 2.39.0's `windowed_adaptation` compiled from its source: identical window ends for
 *   warm-ups of 150 and of 200 or more (500, 1000, 5000, 20000 tested). Below that Stan's own code
 *   differs, and this keeps adapting on purpose: for 20-149 iterations Stan's reduced-stage branch
 *   returns without restarting its counters, so it never closes a window (no metric adaptation at
 *   all); for 151-199 it closes an extra window of one iteration. Here short warm-ups use 15 %/75 %/10 %
 *   stages (rounded) with one window, and fewer than 20 iterations adapt nothing.
 * - **DualAveragingStepSize**: Nesterov dual averaging of a log step size towards a target acceptance
 *   statistic (Hoffman & Gelman, JMLR 15, 1593 (2014), Algorithm 5), with Stan's constants (gamma 0.05,
 *   t0 10, kappa 0.75) and `stepsize_adaptation::learn_stepsize`'s update order; checked against that
 *   header at 1e-15. `restart(step)` centres on log(10 step) as Stan's NUTS does; `restart_log(mu)`
 *   centres anywhere (the blocked Metropolis centres its per-block log scale on 2.38/sqrt(k)).
 */

#ifndef IMPBFF_SAMPLERWARMUP_H
#define IMPBFF_SAMPLERWARMUP_H

#include <IMP/bff/bff_config.h>

#include <algorithm>
#include <cmath>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The windowed warm-up schedule for n_warmup iterations.
struct WarmupWindows {
  int init_buffer = 0;          //!< iterations before the first window
  int term_buffer = 0;          //!< iterations after the last window
  std::vector<int> ends;        //!< iteration counts (1-based) at which a window closes
  //! whether iteration count i (1-based) closes a window
  bool closes(int i) const { return std::find(ends.begin(), ends.end(), i) != ends.end(); }
};

inline WarmupWindows warmup_windows(int n_warmup, int init_buffer = 75, int term_buffer = 50, int base_window = 25) {
  WarmupWindows w;
  w.init_buffer = init_buffer;
  w.term_buffer = term_buffer;
  if (n_warmup < 20) {
    w.init_buffer = n_warmup;
    w.term_buffer = 0;
    return w;
  }
  if (init_buffer + base_window + term_buffer > n_warmup) {
    w.init_buffer = int(std::lround(0.15 * n_warmup));
    w.term_buffer = int(std::lround(0.10 * n_warmup));
    base_window = n_warmup - w.init_buffer - w.term_buffer;
    if (base_window < 2) {
      w.init_buffer = n_warmup;
      w.term_buffer = 0;
      return w;
    }
  }
  int start = w.init_buffer, window = base_window;
  const int last = n_warmup - w.term_buffer;
  while (start + window <= last) {
    int end = start + window;
    if (end + 2 * window > last) end = last;
    w.ends.push_back(end);
    start = end;
    window *= 2;
  }
  return w;
}

//! Dual averaging of a step size.
class DualAveragingStepSize {
 public:
  explicit DualAveragingStepSize(double target = 0.8, double gamma = 0.05, double t0 = 10.0, double kappa = 0.75)
      : delta_(target), gamma_(gamma), t0_(t0), kappa_(kappa) {}

  //! Start adapting around log(10 step), Stan's mu.
  void restart(double step) { restart_log(std::log(10.0 * step)); }
  //! Start adapting around the log value mu.
  void restart_log(double mu) {
    mu_ = mu;
    s_bar_ = 0.0;
    x_bar_ = mu;
    counter_ = 0.0;
  }
  //! One update from the transition's acceptance statistic; returns the log value to use next.
  double learn_log(double adapt_stat) {
    counter_ += 1.0;
    adapt_stat = std::min(1.0, adapt_stat);
    const double eta = 1.0 / (counter_ + t0_);
    s_bar_ = (1.0 - eta) * s_bar_ + eta * (delta_ - adapt_stat);
    const double x = mu_ - s_bar_ * std::sqrt(counter_) / gamma_;
    const double x_eta = std::pow(counter_, -kappa_);
    x_bar_ = (1.0 - x_eta) * x_bar_ + x_eta * x;
    return x;
  }
  //! learn_log(), as a step size.
  double learn(double adapt_stat) { return std::exp(learn_log(adapt_stat)); }
  //! The averaged log value to freeze at the end of warm-up (mu when nothing was learned).
  double final_log() const { return x_bar_; }
  //! The averaged step to freeze at the end of warm-up (the last step when nothing was learned).
  double final_step(double current) const { return counter_ > 0.0 ? std::exp(x_bar_) : current; }
  double target() const { return delta_; }
  void set_target(double delta) { delta_ = delta; }

 private:
  double delta_, gamma_, t0_, kappa_;
  double mu_ = 0.0, s_bar_ = 0.0, x_bar_ = 0.0, counter_ = 0.0;
};

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SAMPLERWARMUP_H */

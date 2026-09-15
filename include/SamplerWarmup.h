/**
 * \file IMP/bff/SamplerWarmup.h
 * \brief Warm-up adaptation shared by bff's samplers: dual averaging of a step size.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * Nesterov dual averaging of log step size towards a target acceptance statistic, as Hoffman &
 * Gelman, JMLR 15, 1593 (2014), Algorithm 5, with Stan's constants (gamma 0.05, t0 10, kappa 0.75)
 * and Stan's `stepsize_adaptation::learn_stepsize` update order. One implementation (PRD-147
 * step 3 moves the blocked Metropolis onto it; the windowed schedule and the covariance estimator
 * join this header then).
 */

#ifndef IMPBFF_SAMPLERWARMUP_H
#define IMPBFF_SAMPLERWARMUP_H

#include <IMP/bff/bff_config.h>

#include <algorithm>
#include <cmath>

IMPBFF_BEGIN_NAMESPACE

//! Dual averaging of a step size.
class DualAveragingStepSize {
 public:
  explicit DualAveragingStepSize(double target = 0.8, double gamma = 0.05, double t0 = 10.0, double kappa = 0.75)
      : delta_(target), gamma_(gamma), t0_(t0), kappa_(kappa) {}

  //! Start adapting around log(10 step), Stan's mu.
  void restart(double step) {
    mu_ = std::log(10.0 * step);
    s_bar_ = 0.0;
    x_bar_ = 0.0;
    counter_ = 0.0;
  }
  //! One update from the transition's acceptance statistic; returns the step to use next.
  double learn(double adapt_stat) {
    counter_ += 1.0;
    adapt_stat = std::min(1.0, adapt_stat);
    const double eta = 1.0 / (counter_ + t0_);
    s_bar_ = (1.0 - eta) * s_bar_ + eta * (delta_ - adapt_stat);
    const double x = mu_ - s_bar_ * std::sqrt(counter_) / gamma_;
    const double x_eta = std::pow(counter_, -kappa_);
    x_bar_ = (1.0 - x_eta) * x_bar_ + x_eta * x;
    return std::exp(x);
  }
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

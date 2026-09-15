/**
 * \file IMP/bff/BayesianFisherScoring.h
 * \brief The smooth positive floor a Poisson mean needs (the score itself is
 *        tttrlib's, `internal/PoissonScore.h`).
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_BAYESIANFISHERSCORING_H
#define IMPBFF_BAYESIANFISHERSCORING_H

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/internal/PoissonScore.h>
#include <cmath>
#include <limits>
#include <cstddef>
#include <thread>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name Fisher scoring for count models
//! @{

//! The Poisson score and information matrices are tttrlib's (`internal/PoissonScore.h`:
//! `tttrlib::poisson_score`, `tttrlib::poisson_score_blocks`, `tttrlib::PoissonInformationKind`),
//! moved there 2026-09-15 (PRD-143 A4.1) so that fit2x and this model share them.

/**
 * \brief `s log(1 + exp(x/s))`: a mean that stays positive without a kink.
 *
 * A Poisson mean must be positive -- the likelihood has `log m` in it -- but a
 * model built as a sum of terms can go slightly negative where there is no
 * signal. Clamping at zero puts a kink in the objective exactly where the
 * optimiser works, and a zero derivative beyond it, so the fit stops seeing the
 * data there. Bending instead costs nothing above a few `s` and never reaches
 * zero. Its derivative is `sigmoid(x/s)`, which the chain rule needs and which
 * is why the floor has to be smooth rather than clamped.
 *
 * `threshold`: above it (in units of `s`) the value is `x` exactly -- torch's
 * convention (`BAYESIAN_TORCH_SOFTPLUS_THRESHOLD`); the default never cuts. The two
 * differ by under `s * 2e-9`.
 */
inline double bayesian_soft_positive(double x, double s = 0.05,
                                     double threshold = std::numeric_limits<double>::infinity()) {
  const double z = x / s;
  if (z > threshold) return x;
  return s * (z > 0.0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z)));
}

//! torch's `softplus` threshold: above `x/s = 20` it returns `x` itself. A model
//! gated against a torch prototype passes it as `threshold` to both functions.
constexpr double BAYESIAN_TORCH_SOFTPLUS_THRESHOLD = 20.0;

//! `d bayesian_soft_positive / dx`.
inline double bayesian_soft_positive_derivative(double x, double s = 0.05,
                                                double threshold = std::numeric_limits<double>::infinity()) {
  const double z = x / s;
  if (z > threshold) return 1.0;
  return z > 0.0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_BAYESIANFISHERSCORING_H

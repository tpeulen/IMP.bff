/**
 * \file IMP/bff/Optimization.h
 * \brief Given a curvature and a gradient, find a step that actually improves
 *        the objective.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * The algorithms live in tttrlib (`modules/math/include/DampedNewton.h`), carried
 * here as the verbatim copy `internal/DampedNewton.h` (tpeulen, 2026-09-15:
 * "Optimizer (the algos) should sit in tttrlib, if header only. I do not want
 * duplication"). This header keeps the names imp.bff callers use; it holds no
 * arithmetic of its own. See the copy for `CholeskyFactor`, `cholesky_solve`,
 * `log_det_spd` and `DampedNewton` (with `line_search_below`).
 */

#ifndef IMPBFF_OPTIMIZATION_H
#define IMPBFF_OPTIMIZATION_H

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/internal/DampedNewton.h>

IMPBFF_BEGIN_NAMESPACE

//! \name Damped Newton steps (tttrlib's, vendored)
//! @{
using CholeskyFactor = ::tttrlib::CholeskyFactor;
using OptimizationStepResult = ::tttrlib::OptimizationStepResult;
using ::tttrlib::cholesky_solve;
using ::tttrlib::log_det_spd;
template <typename Objective>
using DampedNewton = ::tttrlib::DampedNewton<Objective>;
//! @}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_OPTIMIZATION_H

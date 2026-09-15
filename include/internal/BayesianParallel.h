/**
 * \file IMP/bff/internal/BayesianParallel.h
 * \brief The one thread pool the Bayesian decay headers run their loops on.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * Moved out of `BayesianDecayModel.h` (2026-09-15, PRD-143 #18d) so that the
 * transfer tensors share the pool without including the model.
 */

#ifndef IMPBFF_INTERNAL_BAYESIAN_PARALLEL_H
#define IMPBFF_INTERNAL_BAYESIAN_PARALLEL_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/ThreadPool.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>

IMPBFF_BEGIN_NAMESPACE

namespace internal {

//! The pool the decay model's loops run on: bff's `ThreadPool`, one per process,
//! `BFF_BAYESIAN_THREADS` threads in all (default 4).
inline ThreadPool& bayesian_thread_pool() {
    static ThreadPool pool([] { const char* e = std::getenv("BFF_BAYESIAN_THREADS"); return e ? std::max(1, std::atoi(e)) : 4; }());
    return pool;
}
//! Set on a thread that is itself one of several concurrent fits: its loops then
//! run inline instead of contending for the one pool (which is not re-entrant).
inline bool& bayesian_serial_here() { static thread_local bool s = false; return s; }
//! body(i) for i in [0, n), on the pool unless this thread runs serially.
template <typename Body>
inline void bayesian_parallel_for(std::size_t n, Body body) {
    if (n == 0) return;
    if (bayesian_serial_here() || n == 1) { for (std::size_t i = 0; i < n; ++i) body(i); return; }
    const std::function<void(std::size_t)> fn = body;
    bayesian_thread_pool().run(n, fn);
}

}  // namespace internal

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INTERNAL_BAYESIAN_PARALLEL_H */

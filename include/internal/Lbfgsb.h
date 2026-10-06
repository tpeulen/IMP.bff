/**
 *  \file IMP/bff/internal/Lbfgsb.h
 *  \brief L-BFGS-B 3.0 reverse-communication driver (`setulb`).
 *
 * A port of SciPy's C translation of L-BFGS-B 3.0 (Byrd, Lu, Nocedal and Zhu
 * 1995; Morales and Nocedal 2011), `scipy/optimize/src/lbfgsb.c`, so that
 * bounded quasi-Newton minimisation is available without SciPy. The
 * algorithm text is SciPy's, unchanged; only the BLAS/LAPACK it calls
 * (`dcopy`, `ddot`, `daxpy`, `dscal`, upper-triangular `dpotrf`/`dtrtrs` on
 * matrices no larger than `2 m x 2 m`) is supplied here, in the unit-stride,
 * reference-BLAS order SciPy's code uses.
 *
 * License: BSD-3-Clause (SciPy; L-BFGS-B by Ciyou Zhu, Richard Byrd, Peihuang
 * Lu and Jorge Nocedal, distributed under the "New BSD License"). The full
 * notice is at the head of src/internal/Lbfgsb.cpp.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_LBFGSB_H
#define IMPBFF_INTERNAL_LBFGSB_H

#include <IMP/bff/bff_config.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

namespace lbfgsb {

//! One reverse-communication step of L-BFGS-B.
/*! Arguments exactly as SciPy's `setulb` (`wa` holds
    `2 m n + 5 n + 11 m^2 + 8 m` doubles, `iwa` `3 n` ints, `lsave` 4,
    `isave` 44, `dsave` 29). `nbd[i]`: 0 unbounded, 1 lower only, 2 both,
    3 upper only. On return `task[0]` says what the caller must do next:
    FG -- evaluate f and g at x and call again; NEW_X -- an iteration
    finished; anything else -- stop, with `task[1]` the detailed reason.
    Task codes are SciPy's: 0 START, 1 NEW_X, 2 RESTART, 3 FG, 4 CONVERGENCE,
    5 STOP, 6 WARNING, 7 ERROR, 8 ABNORMAL. */
void setulb(int n, int m, double* x, double* l, double* u, int* nbd,
            double* f, double* g, double factr, double pgtol, double* wa,
            int* iwa, int* task, int* lsave, int* isave, double* dsave,
            int maxls, int* ln_task);

}  // namespace lbfgsb

IMPBFF_END_INTERNAL_NAMESPACE

#endif /* IMPBFF_INTERNAL_LBFGSB_H */

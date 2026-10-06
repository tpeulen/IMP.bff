/**
 *  \file IMP/bff/internal/Lsoda.h
 *  \brief LSODA (Hindmarsh and Petzold), the integrator behind scipy's odeint.
 *
 * A port of SciPy's C translation of ODEPACK's LSODA,
 * `scipy/integrate/src/lsoda.{c,h}`: automatic switching between Adams
 * (non-stiff) and BDF (stiff) methods. The algorithm text is SciPy's; see
 * the head of src/internal/Lsoda.cpp for the changes and the notice
 * (BSD-3-Clause, SciPy; ODEPACK is public domain).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_LSODA_H
#define IMPBFF_INTERNAL_LSODA_H

#include <IMP/bff/bff_config.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

namespace lsoda {

/**
 * @brief Struct to hold the LSODA common block variables.
 *
 * This struct serves as a C representation of the Fortran common blocks used in LSODA.
 * Moreover, original Fortran LSODA code, type puns doubles and ints in the same common
 * block making it impossible to decipher which variables are used in which way. Hence,
 * those punned variables are replicated as separate variables. While this slightly
 * increases the memory usage, it greatly improves code clarity and maintainability.
 *
 * NOTE: The struct is organized with all doubles first, then all ints, to enable
 * efficient serialization via memcpy for state persistence between Python calls.
 */
typedef struct {
    /*
     * All double precision variables (240 total)
     * Combining double common blocks ls0001 and lsa001
     */
    double conit, crate, el[13], elco[156], hold, rmax, tesco[36], ccmax, el0, h, hmin, hmxi, hu, rc, tn, uround;
    double tsw, pdest, pdlast, ratio, cm1[12], cm2[5], pdnorm;

    /*
     * All integer variables (48 total)
     * Combining integer common blocks ls0001 and lsa001
     */
    int illin, init, lyh, lewt, lacor, lsavf, lwm, liwm, mxstep, mxhnil,
        nhnil, ntrep, nslast, nyh, ialth, ipup, lmax, meo, nqnyh, nslp,
        icf, ierpj, iersl, jcur, jstart, kflag, l, meth, miter, maxord,
        maxcor, msbp, mxncf, n, nq, nst, nfe, nje, nqu, /* lsa001 part*/ insufr,
        insufi, ixpr, icount, irflag, jtyp, mused, mxordn, mxords;
} lsoda_common_struct_t;

typedef void (*lsoda_func_t)(int* neq, double* t, double* y, double* ydot);
typedef void (*lsoda_jac_t)(int* neq, double* t, double* y, int* ml, int* mu,
                            double* pd, int* nrowpd);

//! One call of LSODA, arguments exactly as SciPy's `lsoda()`.
void lsoda(lsoda_func_t f, int neq, double* y, double* t, double* tout,
           int itol, double* rtol, double* atol, int* itask, int* istate,
           int* iopt, double* rwork, int lrw, int* iwork, int liw,
           lsoda_jac_t jac, const int jt, lsoda_common_struct_t* S);

}  // namespace lsoda

IMPBFF_END_INTERNAL_NAMESPACE

#endif /* IMPBFF_INTERNAL_LSODA_H */

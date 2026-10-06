/**
 *  \file IMP/bff/Numerics.h
 *  \brief Special functions, distribution functions and the matrix
 *         exponential, elementwise over numpy arrays.
 *
 * These are the scalar numerics ChiSurf used to take from scipy
 * (`scipy.special`, `scipy.stats`, `scipy.linalg.expm`). They are here so a
 * consumer of bff needs no scipy, and they are computed **by the same code
 * scipy runs**, not by a second implementation of the same mathematics:
 *
 * - the special functions call SciPy's own `xsf` library (Cephes ports and
 *   friends, vendored privately in `thirdparty/xsf`, never exposed through a
 *   public header), so they agree with `scipy.special` to the last bit;
 * - the distribution functions scipy delegates to Boost.Math (`betainc`,
 *   `betaincinv`, the F distribution, the binomial pmf, Student's t, the
 *   non-central chi-square)
 *   use Boost.Math here too, with scipy's own error policies, so the two agree
 *   to Boost's round-off (and to the bit when both use the same Boost);
 * - expm() is Eigen's scaling-and-squaring Padé exponential (Higham 2005),
 *   the algorithm family scipy uses; it agrees to round-off, not to the bit.
 *
 * Every function is elementwise over 1-D arrays of **equal length** -- the
 * Python shim (`chisurf.core.math.numerics`) does numpy broadcasting and
 * reshaping, so the C++ only ever sees flat contiguous float64. Results come
 * back as owned numpy views (no copy, see internal/OutputView.h).
 *
 * Domain errors give NaN and poles give +-inf, as scipy does (scipy also
 * emits a warning; this does not).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_NUMERICS_H
#define IMPBFF_NUMERICS_H

#include <IMP/bff/bff_config.h>

IMPBFF_BEGIN_NAMESPACE

/** \name Special functions (scipy.special, via xsf)
    Each takes `x` (and a second argument where named) and publishes one
    array of the same length.
*/
//!@{
//! \f$\ln|\Gamma(x)|\f$ -- `scipy.special.gammaln`.
IMPBFFEXPORT void gammaln_array(const double* in_x, int n_x,
                                double** out_view, int* n_out_view);
//! Error function -- `scipy.special.erf`.
IMPBFFEXPORT void erf_array(const double* in_x, int n_x,
                            double** out_view, int* n_out_view);
//! Complementary error function -- `scipy.special.erfc`.
IMPBFFEXPORT void erfc_array(const double* in_x, int n_x,
                             double** out_view, int* n_out_view);
//! Digamma \f$\psi(x)\f$ -- `scipy.special.digamma`.
IMPBFFEXPORT void digamma_array(const double* in_x, int n_x,
                                double** out_view, int* n_out_view);
//! Exponentially scaled \f$e^{-|x|} I_0(x)\f$ -- `scipy.special.i0e`.
IMPBFFEXPORT void i0e_array(const double* in_x, int n_x,
                            double** out_view, int* n_out_view);
//! Bessel \f$J_0(x)\f$ -- `scipy.special.j0`.
IMPBFFEXPORT void j0_array(const double* in_x, int n_x,
                           double** out_view, int* n_out_view);
//! Bessel \f$J_1(x)\f$ -- `scipy.special.j1`.
IMPBFFEXPORT void j1_array(const double* in_x, int n_x,
                           double** out_view, int* n_out_view);
//! Standard normal CDF \f$\Phi(x)\f$ -- `scipy.special.ndtr`, `norm.cdf`.
IMPBFFEXPORT void ndtr_array(const double* in_x, int n_x,
                             double** out_view, int* n_out_view);
//! Regularised lower incomplete gamma \f$P(a, x)\f$ -- `scipy.special.gammainc`.
IMPBFFEXPORT void gammainc_array(const double* in_a, int n_a,
                                 const double* in_x, int n_x,
                                 double** out_view, int* n_out_view);
//! Regularised upper incomplete gamma \f$Q(a, x)\f$ -- `scipy.special.gammaincc`.
IMPBFFEXPORT void gammaincc_array(const double* in_a, int n_a,
                                  const double* in_x, int n_x,
                                  double** out_view, int* n_out_view);
//! Hurwitz zeta \f$\zeta(s, q)\f$ -- `scipy.special.zeta(s, q)`.
/*! polygamma is built from it in the shim exactly as scipy builds it:
    \f$\psi^{(n)}(x) = (-1)^{n+1} n!\, \zeta(n+1, x)\f$, digamma for n = 0. */
IMPBFFEXPORT void zeta_array(const double* in_s, int n_s,
                             const double* in_q, int n_q,
                             double** out_view, int* n_out_view);
//! Fresnel integrals -- `scipy.special.fresnel`, which returns `(S, C)`.
/*! \param[out] out_view,n_out_view \f$S(x)\f$
    \param[out] out_second,n_out_second \f$C(x)\f$ */
IMPBFFEXPORT void fresnel_array(const double* in_x, int n_x,
                                double** out_view, int* n_out_view,
                                double** out_second, int* n_out_second);
//!@}

/** \name Distribution functions (scipy.stats internals)
    Argument order follows the scipy.special function each one is, which is
    not always the scipy.stats order: `fdtr(dfn, dfd, x)`, `chdtri(df, q)`.
*/
//!@{
//! Regularised incomplete beta \f$I_x(a, b)\f$ -- `betainc`, `beta.cdf`.
IMPBFFEXPORT void betainc_array(const double* in_a, int n_a,
                                const double* in_b, int n_b,
                                const double* in_x, int n_x,
                                double** out_view, int* n_out_view);
//! Inverse of betainc in x -- `betaincinv`, `beta.ppf`.
IMPBFFEXPORT void betaincinv_array(const double* in_a, int n_a,
                                   const double* in_b, int n_b,
                                   const double* in_p, int n_p,
                                   double** out_view, int* n_out_view);
//! F distribution CDF -- `fdtr(dfn, dfd, x)`, `f.cdf`.
IMPBFFEXPORT void fdtr_array(const double* in_dfn, int n_dfn,
                             const double* in_dfd, int n_dfd,
                             const double* in_x, int n_x,
                             double** out_view, int* n_out_view);
//! F distribution survival -- `fdtrc(dfn, dfd, x)`, `f.sf`.
IMPBFFEXPORT void fdtrc_array(const double* in_dfn, int n_dfn,
                              const double* in_dfd, int n_dfd,
                              const double* in_x, int n_x,
                              double** out_view, int* n_out_view);
//! F distribution quantile -- `fdtri(dfn, dfd, p)`, `f.ppf`.
IMPBFFEXPORT void fdtri_array(const double* in_dfn, int n_dfn,
                              const double* in_dfd, int n_dfd,
                              const double* in_p, int n_p,
                              double** out_view, int* n_out_view);
//! Chi-square survival -- `chdtrc(df, x)`, `chi2.sf`.
IMPBFFEXPORT void chdtrc_array(const double* in_df, int n_df,
                               const double* in_x, int n_x,
                               double** out_view, int* n_out_view);
//! Inverse chi-square survival -- `chdtri(df, q)`, `chi2.isf`.
IMPBFFEXPORT void chdtri_array(const double* in_df, int n_df,
                               const double* in_q, int n_q,
                               double** out_view, int* n_out_view);
//! Student's t CDF -- `stdtr(df, t)`, `t.cdf`; `t.sf(x) = stdtr(df, -x)`.
IMPBFFEXPORT void stdtr_array(const double* in_df, int n_df,
                              const double* in_t, int n_t,
                              double** out_view, int* n_out_view);
//! Binomial pmf -- `binom.pmf(k, n, p)`.
IMPBFFEXPORT void binom_pmf_array(const double* in_k, int n_k,
                                  const double* in_n, int n_n,
                                  const double* in_p, int n_p,
                                  double** out_view, int* n_out_view);
//! Non-central chi-square pdf -- `ncx2.pdf(x, df, nc)`.
IMPBFFEXPORT void ncx2_pdf_array(const double* in_x, int n_x,
                                 const double* in_df, int n_df,
                                 const double* in_nc, int n_nc,
                                 double** out_view, int* n_out_view);
//!@}

//! Matrix exponential -- `scipy.linalg.expm` for a square real matrix.
/*! \param[in] in_matrix row-major `n_rows x n_cols`, square
    \param[out] out_matrix,n_out_rows,n_out_cols \f$e^{A}\f$, row-major */
IMPBFFEXPORT void expm(const double* in_matrix, int n_rows, int n_cols,
                       double** out_matrix, int* n_out_rows, int* n_out_cols);

IMPBFF_END_NAMESPACE

#endif // IMPBFF_NUMERICS_H

/**
 *  \file IMP/bff/SpecialFunctions.h
 *  \brief Special functions the distance distributions need.
 *
 * Only what the polymer distributions call for. In particular the modified
 * Bessel function \f$I_0\f$, which the worm-like chain needs and which is
 * **deliberately the polynomial approximation** rather than the exact
 * function -- see i0().
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_SPECIALFUNCTIONS_H
#define IMPBFF_SPECIALFUNCTIONS_H

#include <IMP/bff/bff_config.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Modified Bessel function \f$I_0(x)\f$, for any real \p x.
/*!
    Boost.Math's exact function, to machine precision. Header-only, so it adds
    no runtime dependency: the standalone build links `Boost::headers` and
    nothing else of Boost.

    **This changed on 2026-09-14 and the change is visible in fitted
    numbers.** It was the Abramowitz & Stegun polynomial (Numerical Recipes'
    `bessi0`), kept deliberately because ChiSurf used it and parity mattered
    more than accuracy -- and kept with a transposed digit, 3.5156299 where
    A&S prints 3.5156229, because the published fits in this stack were made
    with that. Measured against `scipy.special.i0` before the swap: 9.6e-7
    maximum relative difference, and 8.5e-7 across a worm-like-chain axis,
    roughly eighty-five times what the optimiser calls converged.

    So a worm-like-chain distribution computed now differs from one computed
    before by about a part in a million, and refits are the expected
    consequence. Anything comparing against numbers produced by the older
    code should expect that difference rather than treat it as a regression.

    \f$I_0\f$ is **even**, which is the property the worm-like chain depends
    on: its argument is negative, and \f$I_0(-x) = I_0(x)\f$ grows where
    \f$\exp(-x)\f$ decays. Transcribing this as `exp` is not a small error --
    see the note on worm_like_chain().

    \param[in] x the argument
*/
IMPBFFEXPORT double i0(double x);

//! Regularised upper incomplete gamma \f$Q(a, x)\f$.
/*!
    Boost.Math's `gamma_q`, named here so there is one place in this library
    that answers for it. Boost headers are already a hard requirement of both
    builds -- the standalone CMakeLists asks for them outright -- and
    Boost.Math is header-only, so this costs no dependency and, more to the
    point, writes no new numerics. A hand-rolled series and continued fraction
    is a well-known recipe and still a second implementation of something the
    toolchain already carries correctly.

    \param[in] a shape, positive
    \param[in] x argument, non-negative
*/
IMPBFFEXPORT double gamma_q(double a, double x);

//! The goodness-of-fit probability of a chi-square: \f$P(\chi^2 \ge c)\f$.
/*!
    How often a correct model with this many degrees of freedom would produce
    a misfit at least this large. Small means the model does not describe the
    data; near one can mean the errors are overstated or the model has too
    many parameters for the question.

    This answers a different question from BIC or AIC. Those rank candidates
    against each other, and the best of a bad family is still bad -- nothing
    in a comparison says whether the winner fits. Reduced chi-square is the
    same statistic without the calibration; this is the probability.

    \param[in] chi2 the misfit, non-negative
    \param[in] dof degrees of freedom: observations less fitted parameters
*/
IMPBFFEXPORT double chi2_p_value(double chi2, double dof);

//! i0() over an axis.
/*!
    \param[in] x the arguments
    \param[out] out_view,n_out_view \f$I_0\f$ at each point, as a managed view
*/
IMPBFFEXPORT void i0_array(
        const std::vector<double>& x,
        double** out_view = 0, int* n_out_view = 0
);

IMPBFF_END_NAMESPACE

#endif //IMPBFF_SPECIALFUNCTIONS_H

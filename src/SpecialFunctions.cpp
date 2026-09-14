/**
 * \file SpecialFunctions.cpp
 * \brief Special functions the distance distributions need.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SpecialFunctions.h>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <limits>
#include <stdexcept>
#include <IMP/bff/internal/OutputView.h>

#include <cmath>

IMPBFF_BEGIN_NAMESPACE

double i0(double x) {
    // Boost.Math's exact modified Bessel function, header-only, so this adds
    // no runtime dependency -- the standalone build links Boost::headers and
    // nothing else of Boost.
    //
    // This replaced the Abramowitz & Stegun polynomial on 2026-09-14, by the
    // owner's decision, and it is a deliberate break rather than a cleanup.
    // Two things changed with it:
    //
    //   * accuracy, from about 1e-7 to machine precision;
    //   * a transposed digit. The polynomial carried 3.5156299 where A&S and
    //     Numerical Recipes both print 3.5156229, reproduced on purpose
    //     because every published worm-like-chain fit in the stack was made
    //     with it. That is now gone too.
    //
    // Measured before the change: 9.6e-7 maximum relative difference against
    // scipy.special.i0, and 8.5e-7 across a worm-like-chain axis, which is
    // about eighty-five times the optimiser's convergence tolerance. So
    // worm-like-chain fits made after this do not reproduce ones made before
    // it to better than a part in a million, and refits are the expected
    // consequence rather than a surprise.
    //
    // I0 is even, and the worm-like chain depends on that: its argument is
    // negative, and I0(-x) = I0(x) grows where exp(-x) decays. Boost takes
    // the magnitude for an integer order anyway; passing it explicitly says
    // the evenness is load-bearing rather than incidental.
    return boost::math::cyl_bessel_i(0, std::fabs(x));
}

void i0_array(const std::vector<double>& x, double** out_view, int* n_out_view) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = i0(x[i]);
    internal::copy_to_view(out, out_view, n_out_view);
}

double gamma_q(double a, double x) {
  if (!(a > 0.0)) {
    throw std::domain_error("gamma_q: the shape must be positive");
  }
  if (x < 0.0) {
    throw std::domain_error("gamma_q: the argument must be non-negative");
  }
  return boost::math::gamma_q(a, x);
}

double chi2_p_value(double chi2, double dof) {
  if (!(dof > 0.0)) {
    // No degrees of freedom left: the model can reach the data exactly and
    // the fit says nothing about whether it should. Refusing to answer is
    // more honest than returning a probability of one.
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (chi2 < 0.0) {
    throw std::domain_error("chi2_p_value: the misfit must be non-negative");
  }
  return gamma_q(0.5 * dof, 0.5 * chi2);
}

IMPBFF_END_NAMESPACE

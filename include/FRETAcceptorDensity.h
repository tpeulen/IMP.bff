/**
 *  \file IMP/bff/FRETAcceptorDensity.h
 *  \brief A donor quenched by acceptors at a density rather than at a distance.
 *
 *  When acceptors are spread at random through a volume, across a membrane or
 *  along a helix, a donor has many acceptors at many distances and what the
 *  data determine is an acceptor density. For a random distribution without
 *  diffusion or excluded volume the donor decay is known in closed form
 *  (Förster; Wolber and Hudson):
 *
 *      I_DA(t) = I_D(t) exp[-2 eta_d (t / tau0)^{d/6}],
 *      eta_d = Gamma(1 - d/6) / 2 * C/C0,
 *
 *  for dimensionality d = 1, 2, 3. C0 is the density that places, on average,
 *  one acceptor within R0 of a donor, so C/C0 is "acceptors within a Förster
 *  radius". tau0 is the donor lifetime R0 belongs to (R0 and tau0 are one
 *  pair, as in FRETSpectrumNode). kappa^2 = 2/3 is assumed in R0.
 *
 *  The decay is not a sum of exponentials, so it is built on the time axis and
 *  convolved as a curve. Moved here from ChiSurf's
 *  `core/fluorescence/fret/dimensionality.py`, which forwards to it.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_FRETACCEPTORDENSITY_H
#define IMPBFF_FRETACCEPTORDENSITY_H

#include <IMP/bff/bff_config.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! C0, the density with one acceptor within R0 on average, in length^-d.
/*! 3-D: 1/(4/3 pi R0^3); 2-D: 1/(pi R0^2); 1-D: 1/(2 R0).
    \throws std::domain_error unless R0 > 0 and d is 1, 2 or 3. */
IMPBFFEXPORT double acceptor_characteristic_density(double forster_radius, int dimension);

//! eta_d = Gamma(1 - d/6) / 2 * C/C0.
IMPBFFEXPORT double acceptor_reduced_density(double c_over_c0, int dimension);

//! exp[-2 eta_d (t/tau0)^{d/6}] on \p time (non-negative).
IMPBFFEXPORT std::vector<double> acceptor_quenching_factor(const std::vector<double>& time,
                                                           double tau0, double c_over_c0,
                                                           int dimension);

//! The transfer efficiency a density implies, 1 - int I_DA / int I_D.
/*! No closed form in one and two dimensions: the decay of a unit donor
    lifetime is integrated with the trapezoid rule on \p n_points points over
    `[0, t_max_tau]`. */
IMPBFFEXPORT double acceptor_transfer_efficiency(double c_over_c0, int dimension,
                                                 int n_points = 200001,
                                                 double t_max_tau = 200.0);

//! The quenched donor decay on `t_i = i dt`, from its donor-only lifetime spectrum.
/*! `sum_k a_k exp(-t/tau_k)` times the quenching factor. With \p period > 0 the
    decays of the \p folded_periods preceding pulses are added, each quenched at
    its own elapsed time -- folding the *decay*, which keeps the stretched term
    exact where folding a spectrum could not. */
IMPBFFEXPORT std::vector<double> acceptor_density_decay(
    const std::vector<double>& donor_spectrum, int n_points, double dt, double tau0,
    double c_over_c0, int dimension, double period = 0.0, int folded_periods = 8);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FRETACCEPTORDENSITY_H

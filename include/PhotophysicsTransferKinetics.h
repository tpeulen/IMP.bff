/**
 *  \file IMP/bff/PhotophysicsTransferKinetics.h
 *  \brief Excited-state kinetics between two chromophores that transfer both ways.
 *
 *  Two chromophores A and B, each decaying on its own (rate 1/tau), with
 *  energy transfer A->B at k_AB and B->A at k_BA, evolve their excited
 *  populations P = (A*, B*) under
 *
 *      dP/dt = K P,   K = [[-1/tauA - k_AB,  k_BA          ],
 *                          [ k_AB,          -1/tauB - k_BA ]].
 *
 *  What is detected in a channel after a pulse is m^T exp(K t) p0, with p0 the
 *  pulse's excitation of each chromophore (a row of the excitation
 *  PhotophysicsCrosstalkMatrix) and m the channel's detection of each (a column
 *  of the emission matrix). Donor-acceptor FRET is the case k_BA = 0 (the donor
 *  decays under k_FRET, the sensitised acceptor rises and decays), homo-FRET
 *  and partial donor-donor energy migration (PDDEM) are the general case. This
 *  header is THE implementation of that kinetics in bff: the PDDEM model and
 *  the transfer tensors of the Bayesian decay model both evaluate it.
 *
 *  \par Where the transfer rates come from
 *  The rates are inputs, per distance. A FRET rate is k = (1/tau0) (R0/R)^6
 *  (times 3/2 kappa^2 for a kappa^2 other than 2/3) with ONE reference donor
 *  lifetime tau0 -- the lifetime R0 was determined at. R0 and tau0 are a pair:
 *  R0^6 scales with the donor quantum yield and Q_D/tau_D is the radiative
 *  rate, so the pair fixes k for every donor lifetime component, and using
 *  1/tau_D,i per component, or R0 against whatever lifetime is at hand, is
 *  wrong. FRETSpectrumNode's `fret_rates` are computed that way, from its
 *  `forster_radius` and `tau0` ports together.
 *
 *  \par The closed form, and the degeneracy
 *  The eigenvalues are l1 >= l2 with Delta = l1 - l2 = sqrt((1/tauA - 1/tauB +
 *  k_AB - k_BA)^2 + 4 k_AB k_BA). Putzer's form is exact everywhere:
 *
 *      exp(K t) = e^{l2 t} [ I + (K - l2 I) phi(t) ],  phi(t) = expm1(Delta t)/Delta,
 *
 *  with phi -> t as Delta -> 0. Written as components, with c = m^T (K - l2 I) p0
 *  and s = m^T p0:
 *
 *  - Delta > eps |l1|: two exponentials, amplitude c/Delta at rate -l1 and
 *    s - c/Delta at rate -l2 -- no cancelling divisor;
 *  - Delta <= eps |l1| (a grid can land exactly there: 1/tauD + k_AB = 1/tauA
 *    with k_BA = 0): an exponential s at rate -l and a `t e^{-rate t}` term with
 *    coefficient c, l = (l1 + l2)/2. `eps` is a parameter.
 *
 *  Each component carries a kind (#TRANSFER_EXPONENTIAL or
 *  #TRANSFER_T_EXPONENTIAL) so a caller can give each its own convolution
 *  kernel. The mode #TRANSFER_CHISURF_REGULARISED instead reproduces ChiSurf's
 *  pddem: two exponentials with the divisor Delta + 1e-9, for parity only.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_PHOTOPHYSICSTRANSFERKINETICS_H
#define IMPBFF_PHOTOPHYSICSTRANSFERKINETICS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/PhotophysicsCrosstalkMatrix.h>

#include <cmath>
#include <string>
#include <vector>

#ifndef SWIG
#include <IMP/bff/internal/Dual.h>
#endif

IMPBFF_BEGIN_NAMESPACE

//! What a transfer-kinetics component is.
enum TransferComponentKind {
  TRANSFER_EXPONENTIAL = 0,   //!< amplitude * exp(-rate t)
  TRANSFER_T_EXPONENTIAL = 1  //!< amplitude * t * exp(-rate t), at a degeneracy
};

//! How the closed form is evaluated.
enum TransferKineticsMode {
  TRANSFER_EXACT = 0,                //!< Putzer, degeneracy as a t e^{-kt} term
  TRANSFER_CHISURF_REGULARISED = 1   //!< ChiSurf's pddem divisor Delta + 1e-9
};

#ifndef SWIG
namespace internal {

template <typename T>
struct TransferComponent {
  T amplitude;
  T rate;
  int kind;
};

//! The pieces of the closed form: rates, eigenvalues, c = m^T (K - l2 I) p0, s = m^T p0.
template <typename T>
struct TransferPair {
  T it_a, it_b, delta, l1, l2, c, s;
};

template <typename T>
inline TransferPair<T> transfer_pair_t(const T& tau_a, const T& tau_b, const T& k_ab,
                                       const T& k_ba, const T& p_a, const T& p_b,
                                       const T& m_a, const T& m_b) {
  using std::sqrt;
  TransferPair<T> q;
  q.it_a = T(1.0) / tau_a;
  q.it_b = T(1.0) / tau_b;
  const T split = q.it_a - q.it_b + k_ab - k_ba;
  q.delta = sqrt(split * split + T(4.0) * k_ab * k_ba);
  q.l1 = T(0.5) * (-q.it_a - q.it_b - k_ab - k_ba + q.delta);
  q.l2 = q.l1 - q.delta;
  q.c = m_a * (p_a * (-q.l2 - q.it_a - k_ab) + p_b * k_ba) +
        m_b * (p_a * k_ab + p_b * (-q.l2 - q.it_b - k_ba));
  q.s = m_a * p_a + m_b * p_b;
  return q;
}

//! The two components of m^T exp(K t) p0 for one chromophore pair and rate pair.
/*! Templated on the number type: `double`, or tttrlib's `Dual<GradVec<N>>` for
    exact derivatives by every argument. */
template <typename T>
inline void transfer_pair_components_t(const T& tau_a, const T& tau_b, const T& k_ab,
                                       const T& k_ba, const T& p_a, const T& p_b,
                                       const T& m_a, const T& m_b, int mode, double eps,
                                       TransferComponent<T> out[2]) {
  const TransferPair<T> q = transfer_pair_t(tau_a, tau_b, k_ab, k_ba, p_a, p_b, m_a, m_b);
  if (mode == TRANSFER_CHISURF_REGULARISED) {
    const T common = T(1.0) / (q.delta + T(1e-9));
    out[0] = {common * q.c, -q.l1, TRANSFER_EXPONENTIAL};
    out[1] = {common * (m_a * (p_a * (q.l1 + q.it_a + k_ab) - p_b * k_ba) +
                        m_b * (-p_a * k_ab + p_b * (q.l1 + q.it_b + k_ba))),
              -q.l2, TRANSFER_EXPONENTIAL};
    return;
  }
  if (tttrlib::ad_value(q.delta) > eps * std::fabs(tttrlib::ad_value(q.l1))) {
    const T a1 = q.c / q.delta;
    out[0] = {a1, -q.l1, TRANSFER_EXPONENTIAL};
    out[1] = {q.s - a1, -q.l2, TRANSFER_EXPONENTIAL};
    return;
  }
  const T l = T(0.5) * (q.l1 + q.l2);
  out[0] = {q.s, -l, TRANSFER_EXPONENTIAL};
  out[1] = {q.c, -l, TRANSFER_T_EXPONENTIAL};
}

//! phi(t) = expm1(Delta t) / Delta, which is t at Delta = 0.
inline double transfer_phi(double delta, double t) {
  const double x = delta * t;
  return x == 0.0 ? t : std::expm1(x) / delta;
}
template <typename G>
inline tttrlib::Dual<G> transfer_phi(const tttrlib::Dual<G>& delta, double t) {
  const double x = delta.val * t;
  // d phi / d Delta = t^2 (x e^x - expm1(x)) / x^2, which is t^2/2 at 0.
  const double slope = x == 0.0 ? 0.5 * t * t
                                : t * t * (x * std::exp(x) - std::expm1(x)) / (x * x);
  tttrlib::Dual<G> r(transfer_phi(delta.val, t), delta.grad);
  r.grad *= slope;
  return r;
}

//! m^T exp(K t) p0 at one time, exact at the degeneracy (Putzer's form).
template <typename T>
inline T transfer_pair_decay_t(double t, const T& tau_a, const T& tau_b, const T& k_ab,
                               const T& k_ba, const T& p_a, const T& p_b, const T& m_a,
                               const T& m_b) {
  using std::exp;
  const TransferPair<T> q = transfer_pair_t(tau_a, tau_b, k_ab, k_ba, p_a, p_b, m_a, m_b);
  return exp(q.l2 * T(t)) * (q.s + q.c * transfer_phi(q.delta, t));
}

}  // namespace internal
#endif  // SWIG

//! The two components of m^T exp(K t) p0, flat `[amplitude, rate, kind] x 2`.
/*! Per unit populations: pass the excitation `p0 = (p_a, p_b)` and detection
    `m = (m_a, m_b)` a caller wants, e.g. `(1, 0)` and `(0, 1)` for the
    sensitised acceptor after exciting the donor. \p eps is the degeneracy
    threshold on Delta relative to |l1|. */
IMPBFFEXPORT std::vector<double> transfer_pair_components(
    double tau_a, double tau_b, double k_ab, double k_ba, double p_a, double p_b,
    double m_a, double m_b, int mode = TRANSFER_EXACT, double eps = 1e-9);

//! The derivative of #transfer_pair_components by its eight arguments.
/*! Row-major `6 x 8`: rows the flat `[amplitude, rate, kind] x 2` output (the
    kind rows are 0), columns `tau_a, tau_b, k_ab, k_ba, p_a, p_b, m_a, m_b`.
    Exact: the same kernel on tttrlib's dual numbers. At a degeneracy the
    derivative of the branch taken; undefined where Delta = 0 exactly with
    transfer present, where Delta's own derivative is. */
IMPBFFEXPORT std::vector<double> transfer_pair_components_jacobian(
    double tau_a, double tau_b, double k_ab, double k_ba, double p_a, double p_b,
    double m_a, double m_b, int mode = TRANSFER_EXACT, double eps = 1e-9);

//! m^T exp(K t) p0 on a time axis, exact including the degeneracy.
IMPBFFEXPORT std::vector<double> transfer_pair_decay(
    const std::vector<double>& time, double tau_a, double tau_b, double k_ab, double k_ba,
    double p_a, double p_b, double m_a, double m_b);

//! The lifetime spectrum a pulse gives in a channel, for lifetime spectra A and B.
/*!
    For every rate `(w, k)` of \p rates (interleaved, e.g. FRETSpectrumNode's
    `fret_rates`), every component `(cA, tauA)` of \p spectrum_a and `(cB, tauB)`
    of \p spectrum_b, the pair decays under `k_AB = k f_ab`, `k_BA = k f_ba`,
    excited by the pulse's row of \p excitation and detected by the channel's
    column of \p emission (selected by label; chromophores labelled
    \p chromophore_a and \p chromophore_b). The initial populations are
    \p populations = `(pi_A, pi_B, pi_AB)`: molecules carrying only A, only B,
    and the pair. Layout, rate-major and fixed: for each rate, for each iA, jB,
    the pair's two components weighted `w pi_AB cA cB`; then the pure-A
    components `w pi_A m_A p_A cA` and the pure-B ones. Returned interleaved
    `(amplitude, lifetime)`. A degeneracy in #TRANSFER_EXACT mode, which a sum of
    exponentials cannot hold, is represented by two exponentials split by
    `eps |l|` around it (its error is O(eps)); use #transfer_pair_components for
    the exact form.
*/
IMPBFFEXPORT std::vector<double> transfer_kinetics_spectrum(
    const std::vector<double>& spectrum_a, const std::vector<double>& spectrum_b,
    const std::vector<double>& rates, double f_ab, double f_ba,
    const std::vector<double>& populations, const PhotophysicsCrosstalkMatrix& excitation,
    const PhotophysicsCrosstalkMatrix& emission, const std::string& pulse,
    const std::string& channel, const std::string& chromophore_a = "A",
    const std::string& chromophore_b = "B", int mode = TRANSFER_EXACT, double eps = 1e-9);

//! Initial populations `(pi_A, pi_B, pi_AB)` from pure fractions, as ChiSurf's PDDEM.
/*! `pi_A = pA (1 - pB) / (1 - pA pB)`, `pi_B` likewise, `pi_AB = 1 - pi_A - pi_B`. */
IMPBFFEXPORT std::vector<double> transfer_populations_from_pure_fractions(double pure_a,
                                                                          double pure_b);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PHOTOPHYSICSTRANSFERKINETICS_H

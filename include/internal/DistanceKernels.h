/**
 *  \file IMP/bff/internal/DistanceKernels.h
 *  \brief Distance-distribution kernels written once for any number type.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_DISTANCE_KERNELS_H
#define IMPBFF_INTERNAL_DISTANCE_KERNELS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/Dual.h>
#include <IMP/bff/internal/GradVec.h>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

using tttrlib::ad_value;
using tttrlib::Dual;

/*  What the kernels need beyond tttrlib's Dual.h: the special functions are
    bff's (SpecialFunctions.h), so their dual forms live beside them here. */

//! Modified Bessel function I0, even: I0'(x) = I1(x).
inline double bessel_i0(double x) { return boost::math::cyl_bessel_i(0, std::fabs(x)); }
template <typename G>
inline Dual<G> bessel_i0(const Dual<G>& a) {
  // I1 is odd, so the sign of x carries into the slope.
  const double i1 = boost::math::cyl_bessel_i(1, std::fabs(a.val));
  Dual<G> r(bessel_i0(a.val), a.grad);
  r.grad *= a.val < 0.0 ? -i1 : i1;
  return r;
}

//! The gamma function: Gamma'(x) = Gamma(x) psi(x).
inline double gamma_function(double x) { return std::tgamma(x); }
template <typename G>
inline Dual<G> gamma_function(const Dual<G>& a) {
  const double g = std::tgamma(a.val);
  Dual<G> r(g, a.grad);
  r.grad *= g * boost::math::digamma(a.val);
  return r;
}

//! `a^b` for a variable exponent: std::pow for double, exp(b log a) on duals.
inline double power(double a, double b) { return std::pow(a, b); }
template <typename G>
inline Dual<G> power(const Dual<G>& a, const Dual<G>& b) { return exp(b * log(a)); }
template <typename G>
inline Dual<G> power(double a, const Dual<G>& b) { return exp(b * std::log(a)); }

inline bool is_finite(double v) { return std::isfinite(v); }
template <typename G>
inline bool is_finite(const Dual<G>& v) { return std::isfinite(v.val); }

/*  The kernels behind PolymerChain.h and the Gaussian-cloud distance of
    Distributions.h, templated on the number type of their parameters.
    With `double` they are the published functions, operation for operation;
    with `Dual` they give the exact derivative of that same arithmetic, which
    is what PolymerDistances::get_weights_jacobian reports. The axis is always
    `double`: it is data, not a parameter. */

template <typename T>
inline void normalize_sum_t(std::vector<T>& y) {
  T total(0.0);
  for (const T& v : y) total += v;
  if (ad_value(total) > 0.0) {
    for (T& v : y) v /= total;
  }
}

//! The normal density at `x` with location `loc` and width `scale`.
template <typename T, typename X, typename L>
inline std::vector<T> normal_density_t(const std::vector<X>& x, const L& loc,
                                       const T& scale) {
  using std::exp;
  using std::sqrt;
  std::vector<T> y(x.size(), T(0.0));
  if (ad_value(scale) == 0.0) return y;
  const T a = T(1.0) / (T(std::sqrt(2.0 * M_PI)) * scale);
  const T two_s2 = T(2.0) * scale * scale;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const T d = T(x[i]) - loc;
    y[i] = a * exp(-(d * d) / two_s2);
  }
  return y;
}

//! The generalized-normal transform at `x`.
/*! With `double`, exactly Distributions.cpp's arithmetic, operation for
    operation. On dual numbers a shape with `|shape t| < 1e-3`, `t = (x - loc)
    / scale`, takes the transform's series, because `-log(1 - shape t) /
    shape` cancels there and the exact `shape == 0` branch has no shape
    dependence at all -- the function is smooth through 0 and its derivative
    is not 0. The series agrees with the closed form to 1e-12 relative there. */
inline double generalized_normal_z(double x, double loc, double scale, double shape) {
  if (shape == 0.0) return (x - loc) / scale;
  // A non-positive transform argument clamps to the step above 1.0; zero
  // instead would send the logarithm to -inf.
  const double tiny = std::nextafter(1.0, 2.0) - 1.0;
  double t = 1.0 - shape * (x - loc) / scale;
  if (t < 0.0) t = tiny;
  return -std::log(t) / shape;
}
template <typename G>
inline Dual<G> generalized_normal_z(double x, const Dual<G>& loc, const Dual<G>& scale,
                                   const Dual<G>& shape) {
  const Dual<G> t = (Dual<G>(x) - loc) / scale;
  const Dual<G> u = shape * t;
  if (std::fabs(u.val) < 1e-3) {
    return t * (1.0 + u * (0.5 + u * (1.0 / 3.0 + u * 0.25)));
  }
  if (1.0 - u.val < 0.0) {
    // Clamped: the value does not move with the parameters.
    return Dual<G>(-std::log(std::nextafter(1.0, 2.0) - 1.0) / shape.val);
  }
  return -log(1.0 - u) / shape;
}

//! The generalized normal (a normal for `shape == 0`) at `x`; see Distributions.h.
template <typename T>
inline std::vector<T> generalized_normal_density_t(const std::vector<double>& x, const T& loc,
                                                   const T& scale, const T& shape, bool norm) {
  std::vector<T> n(x.size(), T(0.0));
  if (ad_value(scale) == 0.0) return n;
  // The *standard* normal at z: loc and scale are folded into the transform.
  std::vector<T> z(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) z[i] = generalized_normal_z(x[i], loc, scale, shape);
  n = normal_density_t(z, 0.0, T(1.0));
  if (norm) normalize_sum_t(n);
  return n;
}

//! The distance between two Gaussian clouds `separation` apart, width `sigma`.
template <typename T, typename S>
inline std::vector<T> distance_between_gaussian_t(const std::vector<double>& distances,
                                                  const S& separation, const T& sigma) {
  std::vector<T> pr(distances.size(), T(0.0));
  if (ad_value(sigma) != 0.0) {
    if (ad_value(separation) > 0.0) {
      const std::vector<T> a = normal_density_t(distances, separation, sigma);
      const std::vector<T> b = normal_density_t(distances, -separation, sigma);
      for (std::size_t i = 0; i < distances.size(); ++i) {
        pr[i] = T(distances[i]) / T(separation) * (a[i] - b[i]);
      }
    } else {
      const std::vector<T> n = normal_density_t(distances, 0.0, sigma);
      const T inv_s2 = T(1.0) / (sigma * sigma);
      for (std::size_t i = 0; i < distances.size(); ++i) {
        pr[i] = T(2.0) * T(distances[i]) * T(distances[i]) * inv_s2 * n[i];
      }
    }
  }
  return pr;
}

//! The worm-like chain of Becker, Rosa and Everaers; see PolymerChain.h.
template <typename T>
inline std::vector<T> worm_like_chain_t(const std::vector<double>& distances, const T& kappa,
                                        T chain_length, bool normalize, bool distance) {
  using std::exp;
  using std::log;
  using std::pow;
  std::vector<T> pr(distances.size(), T(0.0));
  if (distances.empty()) return pr;
  if (ad_value(chain_length) == 0.0) {
    chain_length = T(*std::max_element(distances.begin(), distances.end()));
  }
  if (ad_value(chain_length) == 0.0) return pr;

  const double a = 14.054;
  const double b = 0.473;
  const T c = T(1.0) - pow(T(1.0) + pow(T(0.38) * pow(kappa, -0.95), -5.0), -0.2);
  // The branch is the paper's: below kappa = 0.125 the correction is linear,
  // above it the fitted form takes over. Written with `exp(0.783*log(x))`
  // rather than `pow(x, 0.783)`, term for term as the paper writes it.
  const T d = (ad_value(kappa) < 0.125)
                  ? kappa + T(1.0)
                  : T(1.0) - T(1.0) / (T(0.177) / (kappa - T(0.111)) +
                                       T(6.4) * exp(T(0.783) * log(kappa - T(0.111))));

  // The reference implementation `break`s at the first distance that reaches
  // the contour length, so everything past that index stays zero *whether or
  // not* those distances are themselves shorter. That is a PREFIX, not a
  // mask, and on an unsorted axis the two differ. Reproduce the prefix.
  std::size_t limit = distances.size();
  for (std::size_t i = 0; i < distances.size(); ++i) {
    if (distances[i] >= ad_value(chain_length)) { limit = i; break; }
  }

  for (std::size_t i = 0; i < limit; ++i) {
    const double r = distances[i];
    const T r_n = T(r) / chain_length;
    const T r_n2 = r_n * r_n;
    T pri = pow((T(1.0) - c * r_n2) / (T(1.0) - r_n2), 2.5);
    pri *= exp(-d * kappa * T(a) * T(b) * T(1.0 + b) /
               (T(1.0) - (T(b) * r_n) * (T(b) * r_n)) * r_n2);
    const T r_n4 = r_n2 * r_n2;
    const T r_n6 = r_n4 * r_n2;
    const T g = (T(-0.75) / kappa - T(0.5)) * r_n2 +
                (T(-0.359375) / kappa + T(1.0625)) * r_n4 +
                (T(-0.109375) / kappa - T(0.5625)) * r_n6;
    pri *= exp(g / (T(1.0) - r_n2));
    // I0, NOT exp: the argument is negative and I0 is even, so I0(-x) GROWS
    // where exp(-x) decays (see PolymerChain.cpp's history).
    pri *= bessel_i0(-d * kappa * T(a) * T(1.0 + b) * r_n /
                     (T(1.0) - (T(b) * r_n) * (T(b) * r_n)));
    pr[i] = pri;
  }
  if (distance) {
    for (std::size_t i = 0; i < pr.size(); ++i) pr[i] *= T(distances[i] * distances[i]);
  }
  if (normalize) normalize_sum_t(pr);
  return pr;
}

//! The worm-like chain broadened by the dye linkers; see PolymerChain.h.
template <typename T>
inline std::vector<T> worm_like_chain_linker_t(const std::vector<double>& distances,
                                               const T& kappa, const T& chain_length,
                                               const T& sigma, bool normalize) {
  std::vector<T> pn(distances.size(), T(0.0));
  if (ad_value(sigma) != 0.0) {
    const std::vector<T> pr =
        worm_like_chain_t(distances, kappa, chain_length, normalize, false);
    for (std::size_t i = 0; i < distances.size(); ++i) {
      if (ad_value(pr[i]) == 0.0) continue;
      const std::vector<T> broad = distance_between_gaussian_t(distances, distances[i], sigma);
      for (std::size_t j = 0; j < pn.size(); ++j) pn[j] += pr[i] * broad[j];
    }
    if (normalize) normalize_sum_t(pn);
  }
  return pn;
}

//! The des Cloizeaux self-avoiding chain; see PolymerChain.h.
template <typename T>
inline std::vector<T> saw_nu_t(const std::vector<double>& distances, const T& r_rms,
                               const T& nu, double gamma_exp) {
  using std::exp;
  using std::pow;
  using std::sqrt;
  std::vector<T> pr(distances.size(), T(0.0));
  if (!(ad_value(nu) > 0.0 && ad_value(nu) < 1.0) || ad_value(r_rms) <= 0.0) return pr;
  const T theta = T(gamma_exp - 1.0) / nu;
  const T delta = T(1.0) / (T(1.0) - nu);
  // <r^2> = r0^2 Gamma((5+theta)/delta) / Gamma((3+theta)/delta), which fixes
  // r0 from the requested RMS distance.
  const T ratio = gamma_function((T(5.0) + theta) / delta) /
                  gamma_function((T(3.0) + theta) / delta);
  const T r0 = r_rms / sqrt(ratio);
  const T norm = delta / (power(r0, T(3.0) + theta) * gamma_function((T(3.0) + theta) / delta));
  for (std::size_t i = 0; i < distances.size(); ++i) {
    const double r = distances[i];
    const T value = norm * power(r, T(2.0) + theta) * exp(-power(T(r) / r0, delta));
    // Non-finite values (an overflow at large r/r0) are squashed to zero.
    pr[i] = is_finite(value) ? value : T(0.0);
  }
  return pr;
}

//! The Ising two-state Gaussian chain; see PolymerChain.h.
template <typename T>
inline std::vector<T> ising_chain_t(const std::vector<double>& distances, int number_of_residues,
                                    const T& b_structured, const T& b_unstructured,
                                    const T& coupling, const T& field, int n_k) {
  using std::exp;
  using std::sin;
  using std::sqrt;
  const std::size_t n_r = distances.size();
  std::vector<T> pr(n_r, T(0.0));
  if (number_of_residues < 1 || n_r == 0 || n_k < 2) return pr;
  const int n = number_of_residues;

  // Per-residue bond variance in the characteristic function, b^2/6.
  const T vS = b_structured * b_structured / T(6.0);
  const T vU = b_unstructured * b_unstructured / T(6.0);

  // Ising nearest-neighbour weights W(sigma, sigma'), states 0 = S, 1 = U,
  // for the energy -J*delta(sigma,sigma') - (h/2)*(is_S(sigma)+is_S(sigma')).
  const T w00 = exp(coupling + field);
  const T w01 = exp(-coupling + T(0.5) * field);
  const T w10 = w01;
  const T w11 = exp(coupling);

  // The k grid runs to where phi has decayed, which the *smallest* bond
  // variance sets -- the stiffest state is the one still oscillating when the
  // other has died away.
  const double r_max = *std::max_element(distances.begin(), distances.end());
  const T smallest = ad_value(vS) < ad_value(vU) ? vS : vU;
  T scale = sqrt(smallest * T(static_cast<double>(n)));
  if (ad_value(scale) < r_max / n) scale = T(r_max / n);
  if (ad_value(scale) < 1e-6) scale = T(1e-6);
  const double k_min = 1e-6;
  const T k_max = T(30.0) / scale;
  const T step = (k_max - T(k_min)) / T(static_cast<double>(n_k - 1));

  std::vector<T> k(n_k), phi(n_k);
  for (int j = 0; j < n_k; ++j) k[j] = T(k_min) + step * T(static_cast<double>(j));
  k[n_k - 1] = k_max;                      // linspace pins its own endpoint

  for (int j = 0; j < n_k; ++j) {
    const T& kk = k[j];
    const T g0 = exp(-kk * kk * vS);
    const T g1 = exp(-kk * kk * vU);
    // M[s][s'] = W[s][s'] * g[s'], and v is a row vector: v <- v M.
    const T m00 = w00 * g0, m01 = w01 * g1;
    const T m10 = w10 * g0, m11 = w11 * g1;
    T v0(1.0), v1(1.0);
    for (int i = 0; i < n; ++i) {
      const T t0 = v0 * m00 + v1 * m10;
      const T t1 = v0 * m01 + v1 * m11;
      v0 = t0;
      v1 = t1;
    }
    phi[j] = v0 + v1;
  }
  const T phi0 = phi[0];
  if (ad_value(phi0) != 0.0) {
    for (int j = 0; j < n_k; ++j) phi[j] /= phi0;
  }

  // P(R) = (2R/pi) * trapz_k[ k sin(kR) phi(k) ].
  for (std::size_t i = 0; i < n_r; ++i) {
    const double r = distances[i];
    T integral(0.0);
    T f_prev = k[0] * sin(k[0] * T(r)) * phi[0];
    for (int j = 1; j < n_k; ++j) {
      const T f = k[j] * sin(k[j] * T(r)) * phi[j];
      integral += T(0.5) * (f + f_prev) * (k[j] - k[j - 1]);
      f_prev = f;
    }
    const T value = T(2.0 * r / M_PI) * integral;
    // A non-finite phi must not poison the normalisation; a negative lobe of
    // the transform is a truncation artefact, not a probability.
    pr[i] = (is_finite(value) && ad_value(value) > 0.0) ? value : T(0.0);
  }

  T area(0.0);
  for (std::size_t i = 1; i < n_r; ++i) {
    area += T(0.5) * (pr[i] + pr[i - 1]) * T(distances[i] - distances[i - 1]);
  }
  if (ad_value(area) > 0.0) {
    for (std::size_t i = 0; i < n_r; ++i) pr[i] /= area;
  }
  return pr;
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_DISTANCE_KERNELS_H

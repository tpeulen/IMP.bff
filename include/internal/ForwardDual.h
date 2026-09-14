/**
 *  \file IMP/bff/internal/ForwardDual.h
 *  \brief Forward-mode derivatives of a kernel written once for any number type.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_FORWARD_DUAL_H
#define IMPBFF_INTERNAL_FORWARD_DUAL_H

#include <IMP/bff/bff_config.h>

#include <array>
#include <cmath>
#include <cstddef>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! A value and its gradient with respect to up to `kDualSize` parameters.
/*! A kernel templated on its number type evaluates with `double` for the
    value and with `Dual` for the value *and* the exact derivative of that
    same computation -- one implementation, so the derivative cannot drift
    from what is output. Branches compare values; at a branch point the
    derivative is that of the branch taken. */
static const std::size_t kDualSize = 8;

struct Dual {
  double v = 0.0;
  std::array<double, kDualSize> d{};
  Dual() = default;
  Dual(double value) : v(value) {}  // NOLINT: a constant has no gradient
  static Dual variable(double value, std::size_t index) {
    Dual x(value);
    x.d[index] = 1.0;
    return x;
  }
};

inline Dual scaled(const Dual& a, double value, double slope) {
  Dual r(value);
  for (std::size_t i = 0; i < kDualSize; ++i) r.d[i] = slope * a.d[i];
  return r;
}

inline Dual operator+(const Dual& a, const Dual& b) {
  Dual r(a.v + b.v);
  for (std::size_t i = 0; i < kDualSize; ++i) r.d[i] = a.d[i] + b.d[i];
  return r;
}
inline Dual operator-(const Dual& a, const Dual& b) {
  Dual r(a.v - b.v);
  for (std::size_t i = 0; i < kDualSize; ++i) r.d[i] = a.d[i] - b.d[i];
  return r;
}
inline Dual operator-(const Dual& a) { return scaled(a, -a.v, -1.0); }
inline Dual operator*(const Dual& a, const Dual& b) {
  Dual r(a.v * b.v);
  for (std::size_t i = 0; i < kDualSize; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return r;
}
inline Dual operator/(const Dual& a, const Dual& b) {
  Dual r(a.v / b.v);
  const double inv = 1.0 / b.v;
  for (std::size_t i = 0; i < kDualSize; ++i) {
    r.d[i] = (a.d[i] - r.v * b.d[i]) * inv;
  }
  return r;
}
inline Dual& operator+=(Dual& a, const Dual& b) { return a = a + b; }
inline Dual& operator-=(Dual& a, const Dual& b) { return a = a - b; }
inline Dual& operator*=(Dual& a, const Dual& b) { return a = a * b; }
inline Dual& operator/=(Dual& a, const Dual& b) { return a = a / b; }

inline bool operator<(const Dual& a, const Dual& b) { return a.v < b.v; }
inline bool operator>(const Dual& a, const Dual& b) { return a.v > b.v; }
inline bool operator<=(const Dual& a, const Dual& b) { return a.v <= b.v; }
inline bool operator>=(const Dual& a, const Dual& b) { return a.v >= b.v; }
inline bool operator==(const Dual& a, const Dual& b) { return a.v == b.v; }
inline bool operator!=(const Dual& a, const Dual& b) { return a.v != b.v; }

inline Dual exp(const Dual& a) { const double e = std::exp(a.v); return scaled(a, e, e); }
inline Dual log(const Dual& a) { return scaled(a, std::log(a.v), 1.0 / a.v); }
inline Dual sqrt(const Dual& a) { const double s = std::sqrt(a.v); return scaled(a, s, 0.5 / s); }
inline Dual sin(const Dual& a) { return scaled(a, std::sin(a.v), std::cos(a.v)); }
inline Dual fabs(const Dual& a) { return scaled(a, std::fabs(a.v), a.v < 0.0 ? -1.0 : 1.0); }
inline Dual pow(const Dual& a, double p) {
  const double value = std::pow(a.v, p);
  return scaled(a, value, p * std::pow(a.v, p - 1.0));
}
inline Dual pow(const Dual& a, const Dual& p) { return exp(p * log(a)); }
inline Dual pow(double a, const Dual& p) { return exp(p * std::log(a)); }
inline bool isfinite(const Dual& a) { return std::isfinite(a.v); }
inline double value_of(const Dual& a) { return a.v; }
inline double value_of(double a) { return a; }

//! Modified Bessel function I0, even: I0'(x) = I1(x).
inline double bessel_i0(double x) { return boost::math::cyl_bessel_i(0, std::fabs(x)); }
inline Dual bessel_i0(const Dual& a) {
  // I1 is odd, so the sign of x carries into the slope.
  const double i1 = boost::math::cyl_bessel_i(1, std::fabs(a.v));
  return scaled(a, bessel_i0(a.v), a.v < 0.0 ? -i1 : i1);
}

//! The gamma function: Gamma'(x) = Gamma(x) psi(x).
inline double gamma_function(double x) { return std::tgamma(x); }
inline Dual gamma_function(const Dual& a) {
  const double g = std::tgamma(a.v);
  return scaled(a, g, g * boost::math::digamma(a.v));
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_FORWARD_DUAL_H

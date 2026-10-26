/**
 *  \file IMP/bff/internal/PortableRandom.h
 *  \brief Random distributions that draw the same numbers on every platform.
 *
 *  The C++ standard fixes the engines (`std::mt19937`, `std::mt19937_64` give
 *  the same stream everywhere) but not the distributions: the algorithm behind
 *  `std::uniform_int_distribution`, `std::normal_distribution`,
 *  `std::poisson_distribution`, ... and `std::shuffle` is the library's choice.
 *  libstdc++ (Linux) and libc++ (macOS) choose differently, so a seeded run
 *  gave different numbers on the two -- a 50 000-pair AV distance histogram
 *  moved by Monte Carlo noise between machines with the same seed. These
 *  replacements are specified here, bit for bit, on top of the raw engine
 *  output; they keep the standard distributions' constructor and call shape,
 *  so a call site changes only its type name.
 *
 *  The transcendental functions they use (log, exp, sqrt, lgamma) come from
 *  the platform's libm; a last-ulp difference there can flip an
 *  accept/reject decision only at a measure-zero boundary.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_PORTABLERANDOM_H
#define IMPBFF_INTERNAL_PORTABLERANDOM_H

#include <IMP/bff/bff_config.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

namespace portable {

//! 64 random bits from an engine with a 32- or 64-bit range (mt19937, mt19937_64).
template <class G>
inline std::uint64_t bits64(G& g) {
  constexpr std::uint64_t range =
      static_cast<std::uint64_t>(G::max()) - static_cast<std::uint64_t>(G::min());
  static_assert(range == 0xFFFFFFFFull || range == 0xFFFFFFFFFFFFFFFFull,
                "portable random draws need an engine with a full 32- or 64-bit range");
  if (range == 0xFFFFFFFFFFFFFFFFull) {
    return static_cast<std::uint64_t>(g() - G::min());
  }
  const std::uint64_t hi = static_cast<std::uint64_t>(g() - G::min());
  const std::uint64_t lo = static_cast<std::uint64_t>(g() - G::min());
  return (hi << 32) | lo;
}

//! Uniform double in [0, 1): the top 53 of 64 bits.
template <class G>
inline double canonical(G& g) {
  return static_cast<double>(bits64(g) >> 11) * (1.0 / 9007199254740992.0);
}

//! Uniform integer in [0, n) without bias (Lemire's multiply-shift with rejection).
template <class G>
inline std::uint64_t below(G& g, std::uint64_t n) {
  if (n == 0) return 0;
  std::uint64_t x = bits64(g);
#if defined(__SIZEOF_INT128__)
  unsigned __int128 m = static_cast<unsigned __int128>(x) * n;
  std::uint64_t low = static_cast<std::uint64_t>(m);
  if (low < n) {
    const std::uint64_t threshold = (0 - n) % n;
    while (low < threshold) {
      x = bits64(g);
      m = static_cast<unsigned __int128>(x) * n;
      low = static_cast<std::uint64_t>(m);
    }
  }
  return static_cast<std::uint64_t>(m >> 64);
#else
  // Rejection on the remainder: the same values wherever 128-bit products are missing
  // would need the same algorithm, so this branch is for completeness only (MSVC).
  const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max() -
                              std::numeric_limits<std::uint64_t>::max() % n;
  while (x >= limit) x = bits64(g);
  return x % n;
#endif
}

//! `std::uniform_int_distribution` with a fixed algorithm: integers in [a, b].
template <class T = int>
class uniform_int_distribution {
  static_assert(std::is_integral<T>::value, "integral type required");
  T a_, b_;

 public:
  using result_type = T;
  explicit uniform_int_distribution(T a = 0, T b = std::numeric_limits<T>::max())
      : a_(a), b_(b) {}
  T a() const { return a_; }
  T b() const { return b_; }
  template <class G>
  T operator()(G& g) const {
    const std::uint64_t span =
        static_cast<std::uint64_t>(b_) - static_cast<std::uint64_t>(a_);
    if (span == std::numeric_limits<std::uint64_t>::max()) {
      return static_cast<T>(static_cast<std::uint64_t>(a_) + bits64(g));
    }
    return static_cast<T>(static_cast<std::uint64_t>(a_) + below(g, span + 1));
  }
};

//! `std::uniform_real_distribution` with a fixed algorithm: doubles in [a, b).
template <class T = double>
class uniform_real_distribution {
  T a_, b_;

 public:
  using result_type = T;
  explicit uniform_real_distribution(T a = 0, T b = 1) : a_(a), b_(b) {}
  T a() const { return a_; }
  T b() const { return b_; }
  template <class G>
  T operator()(G& g) const {
    return a_ + (b_ - a_) * static_cast<T>(canonical(g));
  }
};

//! `std::normal_distribution` with a fixed algorithm: Marsaglia's polar method,
//! keeping the second value of each pair for the next call.
template <class T = double>
class normal_distribution {
  T mean_, sd_;
  bool have_spare_ = false;
  double spare_ = 0.0;

 public:
  using result_type = T;
  explicit normal_distribution(T mean = 0, T sd = 1) : mean_(mean), sd_(sd) {}
  T mean() const { return mean_; }
  T stddev() const { return sd_; }
  void reset() { have_spare_ = false; }
  template <class G>
  T operator()(G& g) {
    if (have_spare_) {
      have_spare_ = false;
      return mean_ + sd_ * static_cast<T>(spare_);
    }
    double u, v, s;
    do {
      u = 2.0 * canonical(g) - 1.0;
      v = 2.0 * canonical(g) - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double f = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * f;
    have_spare_ = true;
    return mean_ + sd_ * static_cast<T>(u * f);
  }
};

//! `std::gamma_distribution` with a fixed algorithm: Marsaglia & Tsang (2000);
//! shape < 1 through Gamma(shape + 1) * U^(1/shape).
template <class T = double>
class gamma_distribution {
  T alpha_, beta_;
  normal_distribution<double> normal_;

 public:
  using result_type = T;
  explicit gamma_distribution(T alpha = 1, T beta = 1) : alpha_(alpha), beta_(beta) {}
  T alpha() const { return alpha_; }
  T beta() const { return beta_; }
  template <class G>
  T operator()(G& g) {
    double shape = static_cast<double>(alpha_);
    double boost = 1.0;
    if (shape < 1.0) {
      double u;
      do u = canonical(g); while (u == 0.0);
      boost = std::pow(u, 1.0 / shape);
      shape += 1.0;
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      double x, v;
      do {
        x = normal_(g);
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      double u = canonical(g);
      if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return static_cast<T>(d * v * boost * beta_);
      if (u > 0.0 && std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
        return static_cast<T>(d * v * boost * beta_);
    }
  }
};

//! `std::poisson_distribution` with a fixed algorithm: multiplication of
//! uniforms below mean 10, Hormann's PTRS transformed rejection above (the
//! method numpy uses).
template <class T = int>
class poisson_distribution {
  double mean_;

 public:
  using result_type = T;
  explicit poisson_distribution(double mean = 1.0) : mean_(mean) {}
  double mean() const { return mean_; }
  template <class G>
  T operator()(G& g) const {
    if (!(mean_ > 0.0)) return 0;
    if (mean_ < 10.0) {
      const double limit = std::exp(-mean_);
      T k = 0;
      double p = canonical(g);
      while (p > limit) {
        ++k;
        p *= canonical(g);
      }
      return k;
    }
    const double slam = std::sqrt(mean_);
    const double loglam = std::log(mean_);
    const double b = 0.931 + 2.53 * slam;
    const double a = -0.059 + 0.02483 * b;
    const double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    const double vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
      const double u = canonical(g) - 0.5;
      const double v = canonical(g);
      const double us = 0.5 - std::fabs(u);
      const double k = std::floor((2.0 * a / us + b) * u + mean_ + 0.43);
      if (us >= 0.07 && v <= vr) return static_cast<T>(k);
      if (k < 0.0 || (us < 0.013 && v > us)) continue;
      if (std::log(v) + std::log(invalpha) - std::log(a / (us * us) + b) <=
          -mean_ + k * loglam - std::lgamma(k + 1.0))
        return static_cast<T>(k);
    }
  }
};

//! `std::binomial_distribution` with a fixed algorithm: the waiting-time
//! (geometric) method while n*min(p, 1-p) is small, Devroye's exact
//! order-statistic split through a Beta draw above.
template <class T = int>
class binomial_distribution {
  T n_;
  double p_;

  template <class G>
  static T waiting_time(G& g, T n, double p) {
    // Count successes as the number of geometric gaps that fit into n trials.
    if (n <= 0 || p <= 0.0) return 0;
    if (p >= 1.0) return n;
    const double q = std::log1p(-p);
    T x = 0;
    double sum = 0.0;
    for (;;) {
      double u;
      do u = canonical(g); while (u == 0.0);
      sum += std::floor(std::log(u) / q) + 1.0;
      if (sum > static_cast<double>(n)) return x;
      ++x;
    }
  }

  template <class G>
  static T draw(G& g, T n, double p) {
    if (n <= 0 || p <= 0.0) return 0;
    if (p >= 1.0) return n;
    if (p > 0.5) return n - draw(g, n, 1.0 - p);
    if (static_cast<double>(n) * p < 30.0) return waiting_time(g, n, p);
    // The i-th smallest of n uniforms is Beta(i, n + 1 - i); split at it.
    const T i = (n + 1) / 2;
    gamma_distribution<double> ga(static_cast<double>(i), 1.0);
    gamma_distribution<double> gb(static_cast<double>(n + 1 - i), 1.0);
    const double x1 = ga(g), x2 = gb(g);
    const double y = x1 / (x1 + x2);
    if (y < p) return i + draw(g, n - i, (p - y) / (1.0 - y));
    return draw(g, i - 1, p / y);
  }

 public:
  using result_type = T;
  explicit binomial_distribution(T n = 1, double p = 0.5) : n_(n), p_(p) {}
  T t() const { return n_; }
  double p() const { return p_; }
  template <class G>
  T operator()(G& g) const {
    return draw(g, n_, p_);
  }
};

//! `std::discrete_distribution` with a fixed algorithm: inversion of the
//! cumulative weights.
template <class T = int>
class discrete_distribution {
  std::vector<double> cdf_;

 public:
  using result_type = T;
  discrete_distribution() : cdf_(1, 1.0) {}
  template <class It>
  discrete_distribution(It first, It last) {
    double total = 0.0;
    for (; first != last; ++first) {
      total += static_cast<double>(*first);
      cdf_.push_back(total);
    }
    if (cdf_.empty() || !(total > 0.0)) cdf_.assign(1, 1.0);
  }
  template <class G>
  T operator()(G& g) const {
    const double u = canonical(g) * cdf_.back();
    const auto it = std::upper_bound(cdf_.begin(), cdf_.end(), u);
    const std::size_t k = static_cast<std::size_t>(std::distance(cdf_.begin(), it));
    return static_cast<T>(std::min(k, cdf_.size() - 1));
  }
};

//! `std::shuffle` with a fixed algorithm: Fisher-Yates from the back.
template <class It, class G>
inline void shuffle(It first, It last, G& g) {
  const auto n = static_cast<std::uint64_t>(std::distance(first, last));
  for (std::uint64_t i = n; i > 1; --i) {
    const std::uint64_t j = below(g, i);
    using std::swap;
    swap(*(first + static_cast<std::ptrdiff_t>(i - 1)), *(first + static_cast<std::ptrdiff_t>(j)));
  }
}

}  // namespace portable

IMPBFF_END_INTERNAL_NAMESPACE

#endif /* IMPBFF_INTERNAL_PORTABLERANDOM_H */

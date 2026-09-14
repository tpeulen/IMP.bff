/**
 *  \file IMP/bff/internal/PythonRandom.h
 *  \brief MT19937 seeded and drawn the way CPython's `random` module does.
 *
 *  A sampler ported from Python repeats its seeded runs only if it draws the
 *  same numbers. `std::mt19937` seeded with an integer is `init_genrand`;
 *  `random.seed(int)` is `init_by_array` over the integer's 32-bit words, and
 *  `random.random()` builds a 53-bit double from two draws. Neither
 *  `std::normal_distribution` nor any other standard distribution is
 *  specified bit for bit, so `gauss` is CPython's own Box-Muller with its
 *  cached second value.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_PYTHONRANDOM_H
#define IMPBFF_INTERNAL_PYTHONRANDOM_H

#include <IMP/bff/bff_config.h>

#include <cmath>
#include <cstdint>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! CPython's `random.Random`, for the draws a ported sampler makes.
class PythonRandom {
  static constexpr int N = 624;
  static constexpr int M = 397;
  std::uint32_t mt_[N];
  int mti_;
  bool have_gauss_next_;
  double gauss_next_;

  void init_genrand(std::uint32_t s) {
    mt_[0] = s;
    for (mti_ = 1; mti_ < N; ++mti_) {
      mt_[mti_] = 1812433253U * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
                  static_cast<std::uint32_t>(mti_);
    }
  }

  void init_by_array(const std::vector<std::uint32_t>& key) {
    init_genrand(19650218U);
    int i = 1, j = 0;
    const int key_length = static_cast<int>(key.size());
    for (int k = (N > key_length ? N : key_length); k; --k) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] +
               static_cast<std::uint32_t>(j);
      ++i;
      ++j;
      if (i >= N) {
        mt_[0] = mt_[N - 1];
        i = 1;
      }
      if (j >= key_length) j = 0;
    }
    for (int k = N - 1; k; --k) {
      mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) -
               static_cast<std::uint32_t>(i);
      ++i;
      if (i >= N) {
        mt_[0] = mt_[N - 1];
        i = 1;
      }
    }
    mt_[0] = 0x80000000U;
  }

  std::uint32_t genrand_uint32() {
    static const std::uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    std::uint32_t y;
    if (mti_ >= N) {
      int kk;
      for (kk = 0; kk < N - M; ++kk) {
        y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
        mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
      }
      for (; kk < N - 1; ++kk) {
        y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
        mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
      }
      y = (mt_[N - 1] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
      mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
      mti_ = 0;
    }
    y = mt_[mti_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
  }

 public:
  //! `random.seed(seed)` for an integer seed.
  explicit PythonRandom(long long seed) : mti_(N + 1), have_gauss_next_(false), gauss_next_(0) {
    // CPython seeds from abs(n), split into 32-bit words, least significant
    // first; zero is one zero word.
    unsigned long long n = seed < 0 ? static_cast<unsigned long long>(-seed)
                                    : static_cast<unsigned long long>(seed);
    std::vector<std::uint32_t> key;
    do {
      key.push_back(static_cast<std::uint32_t>(n & 0xffffffffULL));
      n >>= 32;
    } while (n);
    init_by_array(key);
  }

  //! `random.random()`: a double in [0, 1) from 53 random bits.
  double random() {
    const std::uint32_t a = genrand_uint32() >> 5, b = genrand_uint32() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

  //! `random.gauss(mu, sigma)`, CPython's Box-Muller with the cached value.
  double gauss(double mu, double sigma) {
    double z;
    if (have_gauss_next_) {
      z = gauss_next_;
      have_gauss_next_ = false;
    } else {
      const double x2pi = random() * (2.0 * M_PI);
      const double g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
      z = std::cos(x2pi) * g2rad;
      gauss_next_ = std::sin(x2pi) * g2rad;
      have_gauss_next_ = true;
    }
    return mu + z * sigma;
  }
};

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_PYTHONRANDOM_H

/**
 *  \file IMP/bff/internal/MlpMath.h
 *  \brief Vectorised elementwise functions for the network: tanh, exp,
 *         the logistic sigmoid and SiLU, bit-identical on every SIMD variant.
 *
 * libm's `tanh` was the largest single cost of a small-network training step
 * (~190 us of a 64-wide step, 164 k calls at 20 ns each in a batch-256
 * forward pass of the surrogate net). These run 2 (NEON), 4 (AVX2) or 8
 * (AVX-512) doubles at once and are used by every path alike -- MlpCore's
 * float64 forward pass and its scalar `act_value<double>`, the int8 / FP4
 * inference and FP4 training -- so a network's activations are the same
 * bits whichever path evaluates them.
 *
 * **tanh** (after Cephes `tanh.c`): for `|x| < 0.625` the Cody & Waite form
 * `x + x (z P(z) / Q(z))`, `z = x^2`, with Cephes' coefficients; above it
 * `1 - 2 / (exp(2|x|) + 1)`, the argument of `exp` clamped to 40 (tanh(20)
 * rounds to 1). Both branches share one division: `num / den` with
 * `(num, den) = (z P, Q)` or `(2, e + 1)`. Computed on `|x|`, the sign put
 * back from `x`, so the function is exactly odd, `tanh(-0) = -0`,
 * `tanh(+-inf) = +-1`, NaN passes through.
 * **exp**: Cody-Waite reduction `y = n ln2 + r` (`n` rounded with the
 * 1.5 x 2^52 shifter, fdlibm's `ln2_hi` / `ln2_lo`), `e^r` as its degree-13
 * Taylor polynomial (truncation < 2^-58 on |r| <= ln2/2) in Horner form,
 * `1 + (r + r^2 p(r))`, and `2^n` applied as two exact powers of two
 * `2^n1 2^n2` built from exponent bits (so it over- and underflows like
 * libm on [-746, 710], the clamp).
 * **sigmoid** `1 / (1 + exp(-z))`, **SiLU** `z / (1 + exp(-z))` -- the
 * expressions MlpCore used with libm's exp.
 *
 * Accuracy (checked in `test/cpp_snippets/test_mlp_math.cpp`, measured
 * against libm): see okf/neural-net.md; tanh within 2 ulp of libm on the
 * whole line, monotone on a dense grid.
 *
 * **Bit-identical variants.** Every variant performs the same IEEE
 * operations in the same order (no FMA: each product is kept a separately
 * rounded value by an empty asm, `IMPBFF_MATH_KEEP`, the FP4 kernels'
 * convention, so neither GCC's cross-statement contraction nor `-mfma`
 * fuses it). Selection is at compile time from the target macros
 * (`__aarch64__` NEON, `__AVX512F__`, `__AVX2__`, else x86-64's baseline
 * SSE2), never by cpuid;
 * `IMPBFF_MATH_NO_SIMD` (or `IMPBFF_FP4_NO_SIMD`) forces the scalar code.
 * The scalar code is also the tail of every vector loop.
 *
 * Sources and notices:
 * - Cephes Math Library 2.8, `tanh.c` (rational coefficients and the
 *   0.625 split), Copyright 1984, 1995, 2000 by Stephen L. Moshier; used
 *   freely per the Cephes README, and under the BSD 3-clause licence SciPy
 *   distributes it with (Moshier's permission).
 * - fdlibm `e_exp.c` (`ln2_hi`, `ln2_lo`, the Cody-Waite reduction):
 *   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *   Developed at SunSoft, a Sun Microsystems, Inc. business. Permission to
 *   use, copy, modify, and distribute this software is freely granted,
 *   provided that this notice is preserved.
 *
 * std-only apart from the target intrinsics headers.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPMATH_H
#define IMPBFF_INTERNAL_MLPMATH_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#if !defined(IMPBFF_MATH_NO_SIMD) && !defined(IMPBFF_FP4_NO_SIMD)
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define IMPBFF_MATH_NEON 1
#elif defined(__AVX512F__)
#include <immintrin.h>
#define IMPBFF_MATH_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define IMPBFF_MATH_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define IMPBFF_MATH_SSE2 1
#endif
#endif

// Keep `v` a separately rounded value (see the file comment).
#if defined(__GNUC__) || defined(__clang__)
#if defined(__aarch64__)
#define IMPBFF_MATH_KEEP(v) __asm__("" : "+w"(v))
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__SSE2__)
#if defined(__AVX512F__)
#define IMPBFF_MATH_KEEP(v) __asm__("" : "+v"(v))
#else
#define IMPBFF_MATH_KEEP(v) __asm__("" : "+x"(v))
#endif
#else
#define IMPBFF_MATH_KEEP(v) __asm__("" : "+m"(v))
#endif
#define IMPBFF_MATH_INLINE inline __attribute__((always_inline))
#else
#define IMPBFF_MATH_KEEP(v) ((void)0)
#define IMPBFF_MATH_INLINE inline
#endif

namespace IMP {
namespace bff {
namespace internal {
namespace mlpmath {

//! Which variant this translation unit compiled.
inline const char* math_variant() {
#if defined(IMPBFF_MATH_NEON)
    return "neon";
#elif defined(IMPBFF_MATH_AVX512)
    return "avx512";
#elif defined(IMPBFF_MATH_AVX2)
    return "avx2";
#elif defined(IMPBFF_MATH_SSE2)
    return "sse2";
#else
    return "generic";
#endif
}

namespace detail {

// Cephes tanh.c, |x| < 0.625: P (degree 2) and Q (monic, degree 3).
constexpr double kP0 = -0x1.edc5baafd6f4bp-1;   // -9.64399179425052238628E-1
constexpr double kP1 = -0x1.8d26a0e26682dp+6;   // -9.92877231001918586564E1
constexpr double kP2 = -0x1.93ac030580563p+10;  // -1.61468768441708447952E3
constexpr double kQ0 = 0x1.c33f28a581b86p+6;    //  1.12811678491632931402E2
constexpr double kQ1 = 0x1.176fa0e5535fap+11;   //  2.23548839060100448583E3
constexpr double kQ2 = 0x1.2ec102442040cp+12;   //  4.84406305325125486048E3
// exp: 1/ln2, fdlibm's ln2_hi (low 21 bits zero, n ln2_hi exact) and ln2_lo,
// the round-to-integer shifter 1.5 x 2^52, 1/k! for k = 2..13.
constexpr double kLog2e = 0x1.71547652b82fep+0;
constexpr double kLn2Hi = 0x1.62e42fee00000p-1;
constexpr double kLn2Lo = 0x1.a39ef35793c76p-33;
constexpr double kShift = 0x1.8p52;
constexpr double kC[12] = {0x1.0000000000000p-1,  0x1.5555555555555p-3,  0x1.5555555555555p-5,
                           0x1.1111111111111p-7,  0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-13,
                           0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19, 0x1.27e4fb7789f5cp-22,
                           0x1.ae64567f544e4p-26, 0x1.1eed8eff8d898p-29, 0x1.6124613a86d09p-33};
constexpr std::uint64_t kSign = 0x8000000000000000ULL;
constexpr std::uint64_t kBias = 1023ULL << 52;

inline std::uint64_t bits(double x) { std::uint64_t u; std::memcpy(&u, &x, 8); return u; }
inline double from_bits(std::uint64_t u) { double x; std::memcpy(&x, &u, 8); return x; }

//! Scalar lane operations (the generic code and every vector loop's tail).
struct SOps {
    using V = double;
    using M = bool;
    static IMPBFF_MATH_INLINE V set(double c) { return c; }
    static IMPBFF_MATH_INLINE V add(V a, V b) { return a + b; }
    static IMPBFF_MATH_INLINE V sub(V a, V b) { return a - b; }
    static IMPBFF_MATH_INLINE V mul(V a, V b) { V r = a * b; IMPBFF_MATH_KEEP(r); return r; }
    static IMPBFF_MATH_INLINE V div(V a, V b) { return a / b; }
    static IMPBFF_MATH_INLINE V min(V a, V b) { return b < a ? b : a; }
    static IMPBFF_MATH_INLINE V max(V a, V b) { return b > a ? b : a; }
    static IMPBFF_MATH_INLINE V abs(V a) { return from_bits(bits(a) & ~kSign); }
    static IMPBFF_MATH_INLINE M lt(V a, V b) { return a < b; }
    static IMPBFF_MATH_INLINE V sel(M m, V a, V b) { return m ? a : b; }
    //! |mag| with the sign bit of `s` (mag >= 0).
    static IMPBFF_MATH_INLINE V with_sign(V mag, V s) { return from_bits(bits(mag) | (bits(s) & kSign)); }
    //! `x` where it is NaN, else `y`.
    static IMPBFF_MATH_INLINE V nan_pass(V y, V x) { return x != x ? x : y; }
    //! 2^k for an integer-valued k in [-1022, 1023]: (k + shifter)'s low
    //! bits are k; shifted into the exponent field and biased.
    static IMPBFF_MATH_INLINE V pow2(V k) { return from_bits((bits(k + kShift) << 52) + kBias); }
};

#if defined(IMPBFF_MATH_NEON)
struct VOps {
    using V = float64x2_t;
    using M = uint64x2_t;
    static constexpr int kW = 2;
    static IMPBFF_MATH_INLINE V load(const double* p) { return vld1q_f64(p); }
    static IMPBFF_MATH_INLINE void store(double* p, V v) { vst1q_f64(p, v); }
    static IMPBFF_MATH_INLINE V set(double c) { return vdupq_n_f64(c); }
    static IMPBFF_MATH_INLINE V add(V a, V b) { return vaddq_f64(a, b); }
    static IMPBFF_MATH_INLINE V sub(V a, V b) { return vsubq_f64(a, b); }
    static IMPBFF_MATH_INLINE V mul(V a, V b) { V r = vmulq_f64(a, b); IMPBFF_MATH_KEEP(r); return r; }
    static IMPBFF_MATH_INLINE V div(V a, V b) { return vdivq_f64(a, b); }
    static IMPBFF_MATH_INLINE V min(V a, V b) { return vminq_f64(a, b); }
    static IMPBFF_MATH_INLINE V max(V a, V b) { return vmaxq_f64(a, b); }
    static IMPBFF_MATH_INLINE V abs(V a) { return vabsq_f64(a); }
    static IMPBFF_MATH_INLINE M lt(V a, V b) { return vcltq_f64(a, b); }
    static IMPBFF_MATH_INLINE V sel(M m, V a, V b) { return vbslq_f64(m, a, b); }
    static IMPBFF_MATH_INLINE V with_sign(V mag, V s) { return vbslq_f64(vdupq_n_u64(kSign), s, mag); }
    static IMPBFF_MATH_INLINE V nan_pass(V y, V x) { return vbslq_f64(vceqq_f64(x, x), y, x); }
    static IMPBFF_MATH_INLINE V pow2(V k) {
        const uint64x2_t u = vshlq_n_u64(vreinterpretq_u64_f64(vaddq_f64(k, vdupq_n_f64(kShift))), 52);
        return vreinterpretq_f64_u64(vaddq_u64(u, vdupq_n_u64(kBias)));
    }
};
#elif defined(IMPBFF_MATH_AVX512)
struct VOps {
    using V = __m512d;
    using M = __mmask8;
    static constexpr int kW = 8;
    static IMPBFF_MATH_INLINE V load(const double* p) { return _mm512_loadu_pd(p); }
    static IMPBFF_MATH_INLINE void store(double* p, V v) { _mm512_storeu_pd(p, v); }
    static IMPBFF_MATH_INLINE V set(double c) { return _mm512_set1_pd(c); }
    static IMPBFF_MATH_INLINE V add(V a, V b) { return _mm512_add_pd(a, b); }
    static IMPBFF_MATH_INLINE V sub(V a, V b) { return _mm512_sub_pd(a, b); }
    static IMPBFF_MATH_INLINE V mul(V a, V b) { V r = _mm512_mul_pd(a, b); IMPBFF_MATH_KEEP(r); return r; }
    static IMPBFF_MATH_INLINE V div(V a, V b) { return _mm512_div_pd(a, b); }
    static IMPBFF_MATH_INLINE V min(V a, V b) { return _mm512_min_pd(a, b); }
    static IMPBFF_MATH_INLINE V max(V a, V b) { return _mm512_max_pd(a, b); }
    static IMPBFF_MATH_INLINE V abs(V a) {
        return _mm512_castsi512_pd(_mm512_andnot_epi64(_mm512_set1_epi64(static_cast<long long>(kSign)),
                                                       _mm512_castpd_si512(a)));
    }
    static IMPBFF_MATH_INLINE M lt(V a, V b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
    static IMPBFF_MATH_INLINE V sel(M m, V a, V b) { return _mm512_mask_blend_pd(m, b, a); }
    static IMPBFF_MATH_INLINE V with_sign(V mag, V s) {
        const __m512i sg = _mm512_and_epi64(_mm512_castpd_si512(s), _mm512_set1_epi64(static_cast<long long>(kSign)));
        return _mm512_castsi512_pd(_mm512_or_epi64(_mm512_castpd_si512(mag), sg));
    }
    static IMPBFF_MATH_INLINE V nan_pass(V y, V x) { return _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q), y, x); }
    static IMPBFF_MATH_INLINE V pow2(V k) {
        const __m512i u = _mm512_slli_epi64(_mm512_castpd_si512(_mm512_add_pd(k, _mm512_set1_pd(kShift))), 52);
        return _mm512_castsi512_pd(_mm512_add_epi64(u, _mm512_set1_epi64(static_cast<long long>(kBias))));
    }
};
#elif defined(IMPBFF_MATH_AVX2)
struct VOps {
    using V = __m256d;
    using M = __m256d;
    static constexpr int kW = 4;
    static IMPBFF_MATH_INLINE V load(const double* p) { return _mm256_loadu_pd(p); }
    static IMPBFF_MATH_INLINE void store(double* p, V v) { _mm256_storeu_pd(p, v); }
    static IMPBFF_MATH_INLINE V set(double c) { return _mm256_set1_pd(c); }
    static IMPBFF_MATH_INLINE V add(V a, V b) { return _mm256_add_pd(a, b); }
    static IMPBFF_MATH_INLINE V sub(V a, V b) { return _mm256_sub_pd(a, b); }
    static IMPBFF_MATH_INLINE V mul(V a, V b) { V r = _mm256_mul_pd(a, b); IMPBFF_MATH_KEEP(r); return r; }
    static IMPBFF_MATH_INLINE V div(V a, V b) { return _mm256_div_pd(a, b); }
    static IMPBFF_MATH_INLINE V min(V a, V b) { return _mm256_min_pd(a, b); }
    static IMPBFF_MATH_INLINE V max(V a, V b) { return _mm256_max_pd(a, b); }
    static IMPBFF_MATH_INLINE V sgn() { return _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(kSign))); }
    static IMPBFF_MATH_INLINE V abs(V a) { return _mm256_andnot_pd(sgn(), a); }
    static IMPBFF_MATH_INLINE M lt(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
    static IMPBFF_MATH_INLINE V sel(M m, V a, V b) { return _mm256_blendv_pd(b, a, m); }
    static IMPBFF_MATH_INLINE V with_sign(V mag, V s) { return _mm256_or_pd(mag, _mm256_and_pd(s, sgn())); }
    static IMPBFF_MATH_INLINE V nan_pass(V y, V x) { return _mm256_blendv_pd(y, x, _mm256_cmp_pd(x, x, _CMP_UNORD_Q)); }
    static IMPBFF_MATH_INLINE V pow2(V k) {
        const __m256i u = _mm256_slli_epi64(_mm256_castpd_si256(_mm256_add_pd(k, _mm256_set1_pd(kShift))), 52);
        return _mm256_castsi256_pd(_mm256_add_epi64(u, _mm256_set1_epi64x(static_cast<long long>(kBias))));
    }
};
#elif defined(IMPBFF_MATH_SSE2)
// Every x86-64 CPU has SSE2: the baseline build (a wheel without
// IMPBFF_WITH_AVX2) still runs two lanes; blends are and/andnot/or.
struct VOps {
    using V = __m128d;
    using M = __m128d;
    static constexpr int kW = 2;
    static IMPBFF_MATH_INLINE V load(const double* p) { return _mm_loadu_pd(p); }
    static IMPBFF_MATH_INLINE void store(double* p, V v) { _mm_storeu_pd(p, v); }
    static IMPBFF_MATH_INLINE V set(double c) { return _mm_set1_pd(c); }
    static IMPBFF_MATH_INLINE V add(V a, V b) { return _mm_add_pd(a, b); }
    static IMPBFF_MATH_INLINE V sub(V a, V b) { return _mm_sub_pd(a, b); }
    static IMPBFF_MATH_INLINE V mul(V a, V b) { V r = _mm_mul_pd(a, b); IMPBFF_MATH_KEEP(r); return r; }
    static IMPBFF_MATH_INLINE V div(V a, V b) { return _mm_div_pd(a, b); }
    static IMPBFF_MATH_INLINE V min(V a, V b) { return _mm_min_pd(a, b); }
    static IMPBFF_MATH_INLINE V max(V a, V b) { return _mm_max_pd(a, b); }
    static IMPBFF_MATH_INLINE V sgn() { return _mm_castsi128_pd(_mm_set1_epi64x(static_cast<long long>(kSign))); }
    static IMPBFF_MATH_INLINE V abs(V a) { return _mm_andnot_pd(sgn(), a); }
    static IMPBFF_MATH_INLINE M lt(V a, V b) { return _mm_cmplt_pd(a, b); }
    static IMPBFF_MATH_INLINE V sel(M m, V a, V b) { return _mm_or_pd(_mm_and_pd(m, a), _mm_andnot_pd(m, b)); }
    static IMPBFF_MATH_INLINE V with_sign(V mag, V s) { return _mm_or_pd(mag, _mm_and_pd(s, sgn())); }
    static IMPBFF_MATH_INLINE V nan_pass(V y, V x) { return sel(_mm_cmpunord_pd(x, x), x, y); }
    static IMPBFF_MATH_INLINE V pow2(V k) {
        const __m128i u = _mm_slli_epi64(_mm_castpd_si128(_mm_add_pd(k, _mm_set1_pd(kShift))), 52);
        return _mm_castsi128_pd(_mm_add_epi64(u, _mm_set1_epi64x(static_cast<long long>(kBias))));
    }
};
#endif

//! e^y for y in [-746, 710] (callers clamp; NaN lanes are replaced later).
//! `Wide = false`: y in [-708, 709] only, 2^n applied as one power of two.
template <class O, bool Wide = true>
IMPBFF_MATH_INLINE typename O::V exp_clamped(typename O::V y) {
    using V = typename O::V;
    const V n = O::sub(O::add(O::mul(y, O::set(kLog2e)), O::set(kShift)), O::set(kShift));
    const V r = O::sub(O::sub(y, O::mul(n, O::set(kLn2Hi))), O::mul(n, O::set(kLn2Lo)));
    // p(r) = sum_k kC[k] r^k by Estrin's scheme (short dependency chains).
    const V r2 = O::mul(r, r);
    const V r4 = O::mul(r2, r2);
    const V r8 = O::mul(r4, r4);
    V q[6];
    for (int k = 0; k < 6; ++k) q[k] = O::add(O::set(kC[2 * k]), O::mul(O::set(kC[2 * k + 1]), r));
    const V s0 = O::add(q[0], O::mul(q[1], r2));
    const V s1 = O::add(q[2], O::mul(q[3], r2));
    const V s2 = O::add(q[4], O::mul(q[5], r2));
    const V p = O::add(O::add(s0, O::mul(s1, r4)), O::mul(s2, r8));
    const V e = O::add(O::set(1.0), O::add(r, O::mul(r2, p)));
    if (!Wide) return O::mul(e, O::pow2(n));
    // n = n1 + n2, n1 = round-half-even(n / 2): both in [-539, 513].
    const V n1 = O::sub(O::add(O::mul(n, O::set(0.5)), O::set(kShift)), O::set(kShift));
    const V n2 = O::sub(n, n1);
    return O::mul(O::mul(e, O::pow2(n1)), O::pow2(n2));
}

//! tanh on one register: both branches, one division, selected per lane.
template <class O>
IMPBFF_MATH_INLINE typename O::V tanh_v(typename O::V x) {
    using V = typename O::V;
    const V ax = O::abs(x);
    const V z = O::mul(ax, ax);
    V p = O::add(O::mul(O::set(kP0), z), O::set(kP1));
    p = O::add(O::mul(p, z), O::set(kP2));
    V q = O::add(z, O::set(kQ0));
    q = O::add(O::mul(q, z), O::set(kQ1));
    q = O::add(O::mul(q, z), O::set(kQ2));
    const V e = exp_clamped<O, false>(O::min(O::add(ax, ax), O::set(40.0)));
    const typename O::M small = O::lt(ax, O::set(0.625));
    const V r = O::div(O::sel(small, O::mul(z, p), O::set(2.0)), O::sel(small, q, O::add(e, O::set(1.0))));
    const V y = O::add(O::sel(small, ax, O::set(1.0)), O::mul(O::sel(small, ax, O::set(-1.0)), r));
    return O::nan_pass(O::with_sign(y, x), x);
}

//! exp(-z) clamped, the shared part of sigmoid and SiLU.
template <class O>
IMPBFF_MATH_INLINE typename O::V exp_neg(typename O::V z) {
    return exp_clamped<O>(O::max(O::min(O::sub(O::set(0.0), z), O::set(710.0)), O::set(-746.0)));
}

}  // namespace detail

//! tanh(x), the scalar code: the same operations as the vector lanes, only
//! the branch that is selected is evaluated.
inline double tanh(double x) {
    using O = detail::SOps;
    namespace d = detail;
    if (x != x) return x;
    const double ax = O::abs(x);
    double y;
    if (ax < 0.625) {
        const double z = O::mul(ax, ax);
        double p = O::add(O::mul(d::kP0, z), d::kP1);
        p = O::add(O::mul(p, z), d::kP2);
        double q = O::add(z, d::kQ0);
        q = O::add(O::mul(q, z), d::kQ1);
        q = O::add(O::mul(q, z), d::kQ2);
        y = O::add(ax, O::mul(ax, O::div(O::mul(z, p), q)));
    } else {
        const double e = d::exp_clamped<O, false>(O::min(O::add(ax, ax), 40.0));
        y = O::add(1.0, O::mul(-1.0, O::div(2.0, O::add(e, 1.0))));
    }
    return O::with_sign(y, x);
}

//! 1 / (1 + exp(-z)).
inline double sigmoid(double z) {
    using O = detail::SOps;
    if (z != z) return z;
    return 1.0 / (1.0 + detail::exp_neg<O>(z));
}

//! z / (1 + exp(-z)).
inline double silu(double z) {
    using O = detail::SOps;
    if (z != z) return z;
    return z / (1.0 + detail::exp_neg<O>(z));
}

namespace detail {
#if defined(IMPBFF_MATH_NEON) || defined(IMPBFF_MATH_AVX2) || defined(IMPBFF_MATH_AVX512) || defined(IMPBFF_MATH_SSE2)
template <class O>
IMPBFF_MATH_INLINE typename O::V sigmoid_v(typename O::V z) {
    return O::nan_pass(O::div(O::set(1.0), O::add(O::set(1.0), exp_neg<O>(z))), z);
}
template <class O>
IMPBFF_MATH_INLINE typename O::V silu_v(typename O::V z) {
    return O::nan_pass(O::div(z, O::add(O::set(1.0), exp_neg<O>(z))), z);
}
//! Map over whole registers, then the scalar code for the tail.
template <template <class> class F, class SF>
IMPBFF_MATH_INLINE void map_n(const double* x, double* y, std::size_t n, SF sf) {
    constexpr std::size_t W = VOps::kW;
    std::size_t i = 0;
    for (; i + 2 * W <= n; i += 2 * W) {
        const VOps::V a = F<VOps>::apply(VOps::load(x + i));
        const VOps::V b = F<VOps>::apply(VOps::load(x + i + W));
        VOps::store(y + i, a);
        VOps::store(y + i + W, b);
    }
    for (; i + W <= n; i += W) VOps::store(y + i, F<VOps>::apply(VOps::load(x + i)));
    for (; i < n; ++i) y[i] = sf(x[i]);
}
template <class O> struct TanhF { static IMPBFF_MATH_INLINE typename O::V apply(typename O::V v) { return tanh_v<O>(v); } };
template <class O> struct SigmoidF { static IMPBFF_MATH_INLINE typename O::V apply(typename O::V v) { return sigmoid_v<O>(v); } };
template <class O> struct OneMinusSqF {
    static IMPBFF_MATH_INLINE typename O::V apply(typename O::V v) { return O::sub(O::set(1.0), O::mul(v, v)); }
};
template <class O> struct SiluF { static IMPBFF_MATH_INLINE typename O::V apply(typename O::V v) { return silu_v<O>(v); } };
#define IMPBFF_MATH_LOOP(VFN, SFN) \
    detail::map_n<detail::VFN>(x, y, n, [](double v) { return SFN(v); });
#else
#define IMPBFF_MATH_LOOP(VFN, SFN) \
    for (std::size_t i = 0; i < n; ++i) y[i] = SFN(x[i]);
#endif
}  // namespace detail

//! y[i] = tanh(x[i]); y may alias x.
inline void tanh_n(const double* x, double* y, std::size_t n) { IMPBFF_MATH_LOOP(TanhF, mlpmath::tanh) }
//! y[i] = sigmoid(x[i]); y may alias x.
inline void sigmoid_n(const double* x, double* y, std::size_t n) { IMPBFF_MATH_LOOP(SigmoidF, mlpmath::sigmoid) }
//! y[i] = 1 - x[i]^2 (tanh' from tanh), the square rounded before the
//! subtraction on every variant; y may alias x.
inline double one_minus_sq(double a) { return 1.0 - detail::SOps::mul(a, a); }
inline void one_minus_sq_n(const double* x, double* y, std::size_t n) { IMPBFF_MATH_LOOP(OneMinusSqF, mlpmath::one_minus_sq) }
//! y[i] = silu(x[i]); y may alias x.
inline void silu_n(const double* x, double* y, std::size_t n) { IMPBFF_MATH_LOOP(SiluF, mlpmath::silu) }

#undef IMPBFF_MATH_LOOP

}  // namespace mlpmath
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPMATH_H

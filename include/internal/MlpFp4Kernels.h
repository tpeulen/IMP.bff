/**
 *  \file IMP/bff/internal/MlpFp4Kernels.h
 *  \brief Integer-SIMD kernels on packed FP4 (E2M1) codes: FP4 x int8 and
 *         FP4 x FP4 dot products and GEMMs, and the fast forward pass.
 *
 * **CPUs have no FP4 arithmetic.** "Native" here means every product runs on
 * the packed 4-bit codes through integer SIMD dot products; nothing is
 * dequantised to float first. The approach is llama.cpp/ggml's
 * (`ggml_vec_dot_mxfp4_q8_0`, `ggml_vec_dot_nvfp4_q8_0`), re-expressed for
 * bff's layout:
 *
 * - The E2M1 values times two are the integers
 *   `{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}`
 *   (ggml's `kvalues_mxfp4` / `kvalues_fp4`), so a 16-byte table lookup
 *   turns 16 nibbles into 16 int8 values at once: `vqtbl1q_s8` on ARM NEON,
 *   `_mm_shuffle_epi8` (pshufb) on x86. The factor 1/2 goes into the scales.
 * - The other operand is either int8 (activations quantised per block to
 *   int8 with one float scale a block: 32 elements, or 16 for nvfp4 --
 *   ggml's Q8_0 idea) or FP4 codes looked up through the same table.
 * - The integer dot product runs on `vdotq_s32` (ARMv8.2 dotprod), on
 *   `vmull_s8` + `vmlal_s8` + `vaddlvq_s16` (plain AArch64 NEON), or on
 *   `_mm256_maddubs_epi16` + `_mm256_madd_epi16` with the `_mm256_sign_epi8`
 *   trick for signed x signed (AVX2). One 16-byte group (32 elements) gives
 *   two exact int32 sums, one per 16-element sub-block.
 * - The scales are applied after, in float: per 16-element sub-block,
 *   `sum_int * (scale_a * scale_b)`, with scale = 1/2 x (E8M0 for mxfp4,
 *   E4M3 x float32 tensor scale for nvfp4, the float32 row scale for fp4,
 *   amax/127 for int8). This combination is one shared scalar function,
 *   with four independent accumulators, so every SIMD variant gives
 *   *bit-identical* results to the generic one (the integer part is exact
 *   and the float part is the same code).
 *
 * Layout. FP4 rows are bff's standard packing (element 2i in the low
 * nibble, MlpFp4.h), padded to a multiple of 32 elements. A 16-byte group
 * holds elements 0..31: the low nibbles are the even elements, the high
 * nibbles the odd ones. ggml instead packs element j with j + 16; rather
 * than repack the weights, the *int8 activations* are stored permuted
 * within each group of 32 -- `a'[j] = a[2j]`, `a'[16 + j] = a[2j + 1]` for
 * j < 16 -- so that the low-nibble lookup pairs with `a'[0..15]` and the
 * high-nibble lookup with `a'[16..31]`, exactly ggml's loop. For FP4 x FP4
 * both operands use the same packing, so low pairs with low and high with
 * high and no permutation is needed.
 *
 * Variants are chosen at **compile time** from the target macros --
 * `__ARM_FEATURE_DOTPROD` (NEON + dotprod), `__ARM_NEON` (NEON), `__AVX2__`
 * -- never by run-time cpuid (under Rosetta 2 cpuid reports no AVX2 while
 * AVX2 instructions execute). `IMPBFF_FP4_NO_SIMD` forces the generic
 * scalar code. `kernel_name()` says which was compiled. There is no
 * AVX-512 VNNI variant (not needed for correctness; this development
 * machine cannot run one).
 *
 * Derived from llama.cpp / ggml (https://github.com/ggml-org/llama.cpp):
 * ggml/src/ggml-common.h (`kvalues_mxfp4`, `block_mxfp4`, `block_nvfp4`),
 * ggml/src/ggml-cpu/quants.c (`ggml_vec_dot_mxfp4_q8_0_generic`,
 * `ggml_vec_dot_nvfp4_q8_0_generic`),
 * ggml/src/ggml-cpu/arch/arm/quants.c (`ggml_vec_dot_mxfp4_q8_0`,
 * `ggml_vec_dot_nvfp4_q8_0`, NEON) and
 * ggml/src/ggml-cpu/arch/x86/quants.c (`ggml_vec_dot_mxfp4_q8_0`,
 * `ggml_vec_dot_nvfp4_q8_0`, AVX2), under this licence:
 *
 *   MIT License
 *
 *   Copyright (c) 2023-2026 The ggml authors
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a
 *   copy of this software and associated documentation files (the
 *   "Software"), to deal in the Software without restriction, including
 *   without limitation the rights to use, copy, modify, merge, publish,
 *   distribute, sublicense, and/or sell copies of the Software, and to
 *   permit persons to whom the Software is furnished to do so, subject to
 *   the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included
 *   in all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 *   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 *   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 *   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * std-only apart from the target intrinsics headers.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPFP4KERNELS_H
#define IMPBFF_INTERNAL_MLPFP4KERNELS_H

#include <IMP/bff/internal/MlpFp4.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#if !defined(IMPBFF_FP4_NO_SIMD)
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_DOTPROD)
#define IMPBFF_FP4_NEON_DOTPROD 1
#else
#define IMPBFF_FP4_NEON 1
#endif
#elif defined(__AVX2__)
#include <immintrin.h>
#define IMPBFF_FP4_AVX2 1
#endif
#endif

namespace IMP {
namespace bff {
namespace internal {
namespace mlpfp4 {
namespace kern {

//! E2M1 values times two, by code (ggml's kvalues_mxfp4).
alignas(16) constexpr std::int8_t kValues2[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                                  0, -1, -2, -3, -4, -6, -8, -12};

//! Which variant this translation unit compiled.
inline const char* kernel_name() {
#if defined(IMPBFF_FP4_NEON_DOTPROD)
    return "neon-dotprod";
#elif defined(IMPBFF_FP4_NEON)
    return "neon";
#elif defined(IMPBFF_FP4_AVX2)
    return "avx2";
#else
    return "generic";
#endif
}

// ------------------------------------------------------------------ generic

//! FP4 x int8: for each of `n_groups` 16-byte groups of `w` (32 codes) and
//! the matching 32 permuted int8 values of `a`, the two sub-block sums
//! `out[2g]` (elements 0..15) and `out[2g + 1]` (16..31), of 2 x E2M1 x int8.
inline void isums_q8_generic(const std::uint8_t* w, const std::int8_t* a, int n_groups,
                             std::int32_t* out) {
    for (int g = 0; g < n_groups; ++g) {
        const std::uint8_t* wg = w + 16 * g;
        const std::int8_t* ag = a + 32 * g;
        std::int32_t s[2] = {0, 0};
        for (int j = 0; j < 16; ++j) {
            s[j / 8] += kValues2[wg[j] & 0x0F] * ag[j] + kValues2[wg[j] >> 4] * ag[16 + j];
        }
        out[2 * g] = s[0];
        out[2 * g + 1] = s[1];
    }
}

//! FP4 x FP4: the same sub-block sums for two packed operands, of
//! (2 x E2M1) x (2 x E2M1).
inline void isums_fp4_generic(const std::uint8_t* x, const std::uint8_t* y, int n_groups,
                              std::int32_t* out) {
    for (int g = 0; g < n_groups; ++g) {
        const std::uint8_t* xg = x + 16 * g;
        const std::uint8_t* yg = y + 16 * g;
        std::int32_t s[2] = {0, 0};
        for (int j = 0; j < 16; ++j) {
            s[j / 8] += kValues2[xg[j] & 0x0F] * kValues2[yg[j] & 0x0F] +
                        kValues2[xg[j] >> 4] * kValues2[yg[j] >> 4];
        }
        out[2 * g] = s[0];
        out[2 * g + 1] = s[1];
    }
}

//! The shared float combination: `sum_s is[s] * (sa[s] * sb[s])` over
//! `n_sub` sub-blocks in four lanes (sub-block s goes to lane s % 4, a
//! trailing pair to lanes 0 and 1), then `(l0 + l1) + (l2 + l3)`. Separate
//! statements, so no multiply-add is contracted: the SIMD variants run the
//! same lanes with vector multiplies and adds and give the same bits.
inline float combine(const std::int32_t* is, const float* sa, const float* sb, int n_sub) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int s = 0;
    for (; s + 4 <= n_sub; s += 4)
        for (int i = 0; i < 4; ++i) {
            const float sc = sa[s + i] * sb[s + i];
            const float t = static_cast<float>(is[s + i]) * sc;
            acc[i] += t;
        }
    for (int i = 0; s < n_sub; ++s, ++i) {
        const float sc = sa[s] * sb[s];
        const float t = static_cast<float>(is[s]) * sc;
        acc[i] += t;
    }
    const float l01 = acc[0] + acc[1];
    const float l23 = acc[2] + acc[3];
    return l01 + l23;
}

//! Generic FP4 x int8 dot of `n_groups` groups with sub-block scales.
inline float dot_q8_generic(const std::uint8_t* w, const std::int8_t* a, const float* sa,
                            const float* sw, int n_groups, std::int32_t* scratch) {
    isums_q8_generic(w, a, n_groups, scratch);
    return combine(scratch, sa, sw, 2 * n_groups);
}
//! Generic FP4 x FP4 dot.
inline float dot_fp4_generic(const std::uint8_t* x, const std::uint8_t* y, const float* sx,
                             const float* sy, int n_groups, std::int32_t* scratch) {
    isums_fp4_generic(x, y, n_groups, scratch);
    return combine(scratch, sx, sy, 2 * n_groups);
}

// ------------------------------------------------------------------ SIMD

// Each SIMD variant supplies, for its native vector types `ivec` (4 x int32)
// and `fvec` (4 x float): `quad_q8(w, a)` / `quad_fp4(x, y)` -- the four
// sub-block sums of two groups -- and `pair_q8` / `pair_fp4` -- the two of one
// group, in lanes 0 and 1 -- plus `accumulate` and `finish`, which do in
// vectors exactly what combine() does in scalars.

#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
#define IMPBFF_FP4_SIMD 1
namespace simd {
using ivec = int32x4_t;
using fvec = float32x4_t;
inline fvec fzero() { return vdupq_n_f32(0.0f); }
inline int8x16_t lut_lo(uint8x16_t v) { return vqtbl1q_s8(vld1q_s8(kValues2), vandq_u8(v, vdupq_n_u8(0x0F))); }
inline int8x16_t lut_hi(uint8x16_t v) { return vqtbl1q_s8(vld1q_s8(kValues2), vshrq_n_u8(v, 4)); }
#if defined(IMPBFF_FP4_NEON_DOTPROD)
//! Lanes 0 + 1 and 2 + 3 of lo . a0 + hi . a1 (vdotq: 4 bytes a lane).
inline int32x4_t group_dot(int8x16_t lo, int8x16_t a0, int8x16_t hi, int8x16_t a1) {
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), lo, a0), hi, a1);
}
inline ivec quad(int32x4_t p0, int32x4_t p1) { return vpaddq_s32(p0, p1); }
inline ivec pair(int32x4_t p0) { return vpaddq_s32(p0, p0); }
#else
// Plain AArch64 NEON: widening multiplies; |2 x E2M1 x int8| <= 1524, two of
// them summed per int16 lane (vmlal) stay below 2^15. Returns the two
// sub-block sums in lanes 0 and 1.
inline int32x4_t group_dot(int8x16_t lo, int8x16_t a0, int8x16_t hi, int8x16_t a1) {
    const int16x8_t p = vmlal_s8(vmull_s8(vget_low_s8(lo), vget_low_s8(a0)), vget_low_s8(hi), vget_low_s8(a1));
    const int16x8_t q = vmlal_s8(vmull_s8(vget_high_s8(lo), vget_high_s8(a0)), vget_high_s8(hi), vget_high_s8(a1));
    return vcombine_s32(vpadd_s32(vget_low_s32(vpaddlq_s16(p)), vget_high_s32(vpaddlq_s16(p))),
                        vpadd_s32(vget_low_s32(vpaddlq_s16(q)), vget_high_s32(vpaddlq_s16(q))));
}
inline ivec quad(int32x4_t p0, int32x4_t p1) { return vpaddq_s32(p0, p1); }
inline ivec pair(int32x4_t p0) { return vpaddq_s32(p0, p0); }
#endif
// (plain NEON: group_dot's lanes are [s0a, s0b, s1a, s1b] as well, so
// quad/pair are the same pairwise adds.)
inline ivec quad_q8(const std::uint8_t* w, const std::int8_t* a) {
    const uint8x16_t w0 = vld1q_u8(w), w1 = vld1q_u8(w + 16);
    return quad(group_dot(lut_lo(w0), vld1q_s8(a), lut_hi(w0), vld1q_s8(a + 16)),
                group_dot(lut_lo(w1), vld1q_s8(a + 32), lut_hi(w1), vld1q_s8(a + 48)));
}
inline ivec pair_q8(const std::uint8_t* w, const std::int8_t* a) {
    const uint8x16_t w0 = vld1q_u8(w);
    return pair(group_dot(lut_lo(w0), vld1q_s8(a), lut_hi(w0), vld1q_s8(a + 16)));
}
inline ivec quad_fp4(const std::uint8_t* x, const std::uint8_t* y) {
    const uint8x16_t x0 = vld1q_u8(x), x1 = vld1q_u8(x + 16);
    const uint8x16_t y0 = vld1q_u8(y), y1 = vld1q_u8(y + 16);
    return quad(group_dot(lut_lo(x0), lut_lo(y0), lut_hi(x0), lut_hi(y0)),
                group_dot(lut_lo(x1), lut_lo(y1), lut_hi(x1), lut_hi(y1)));
}
inline ivec pair_fp4(const std::uint8_t* x, const std::uint8_t* y) {
    const uint8x16_t x0 = vld1q_u8(x), y0 = vld1q_u8(y);
    return pair(group_dot(lut_lo(x0), lut_lo(y0), lut_hi(x0), lut_hi(y0)));
}
inline fvec accumulate(fvec acc, ivec s, const float* sa, const float* sb) {
    const float32x4_t sc = vmulq_f32(vld1q_f32(sa), vld1q_f32(sb));
    return vaddq_f32(acc, vmulq_f32(vcvtq_f32_s32(s), sc));
}
inline void store(fvec v, float* l) { vst1q_f32(l, v); }
inline void store(ivec v, std::int32_t* l) { vst1q_s32(l, v); }
}  // namespace simd
#elif defined(IMPBFF_FP4_AVX2)
#define IMPBFF_FP4_SIMD 1
namespace simd {
using ivec = __m128i;
using fvec = __m128;
inline fvec fzero() { return _mm_setzero_ps(); }
//! The 32 int8 values of one group, low-nibble lookups in the low lane.
inline __m256i lookup(const std::uint8_t* p) {
    const __m128i tab = _mm_load_si128(reinterpret_cast<const __m128i*>(kValues2));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i lo = _mm_shuffle_epi8(tab, _mm_and_si128(w, m4));
    const __m128i hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(w, 4), m4));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
//! Signed x signed: |x| (unsigned) times y with x's sign, maddubs then madd
//! (int16 pairs stay below 2^15). Returns [s0, s1, s0, s1].
inline __m128i pair(__m256i x, __m256i y) {
    const __m256i d16 = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    const __m256i d32 = _mm256_madd_epi16(d16, _mm256_set1_epi16(1));
    // lanes 0,1: sub-block 0 of the low-nibble half, 2,3: sub-block 1; 4..7 the high half
    const __m128i t = _mm_add_epi32(_mm256_castsi256_si128(d32), _mm256_extracti128_si256(d32, 1));
    return _mm_hadd_epi32(t, t);
}
inline __m256i load32(const std::int8_t* a) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)); }
inline ivec pair_q8(const std::uint8_t* w, const std::int8_t* a) { return pair(lookup(w), load32(a)); }
inline ivec quad_q8(const std::uint8_t* w, const std::int8_t* a) {
    return _mm_unpacklo_epi64(pair_q8(w, a), pair_q8(w + 16, a + 32));
}
inline ivec pair_fp4(const std::uint8_t* x, const std::uint8_t* y) { return pair(lookup(x), lookup(y)); }
inline ivec quad_fp4(const std::uint8_t* x, const std::uint8_t* y) {
    return _mm_unpacklo_epi64(pair_fp4(x, y), pair_fp4(x + 16, y + 16));
}
inline fvec accumulate(fvec acc, ivec s, const float* sa, const float* sb) {
    const __m128 sc = _mm_mul_ps(_mm_loadu_ps(sa), _mm_loadu_ps(sb));
    return _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), sc));
}
inline void store(fvec v, float* l) { _mm_storeu_ps(l, v); }
inline void store(ivec v, std::int32_t* l) { _mm_storeu_si128(reinterpret_cast<__m128i*>(l), v); }
}  // namespace simd
#endif

#if defined(IMPBFF_FP4_SIMD)
namespace simd {
//! combine()'s last step: the four lanes plus a trailing pair in lanes 0, 1.
inline float finish(fvec acc, bool tail, ivec p, const float* sa, const float* sb) {
    float l[4];
    store(acc, l);
    if (tail) {
        std::int32_t s[4];
        store(p, s);
        const float c0 = sa[0] * sb[0], c1 = sa[1] * sb[1];
        const float t0 = static_cast<float>(s[0]) * c0, t1 = static_cast<float>(s[1]) * c1;
        l[0] += t0;
        l[1] += t1;
    }
    const float l01 = l[0] + l[1];
    const float l23 = l[2] + l[3];
    return l01 + l23;
}
//! Four dots of one left row against four right rows (the left operand's
//! loads are shared). `Quad` / `Pair` are quad_q8/pair_q8 or the fp4 ones.
template <class L, class R, class Quad, class Pair>
inline void dot4(const L* a, const float* sa, const R* const* w, const float* const* sw, int ng,
                 float* out, Quad quad, Pair pair) {
    fvec acc[4] = {fzero(), fzero(), fzero(), fzero()};
    const int la = std::is_same<L, std::int8_t>::value ? 32 : 16;  // left elements a group
    int g = 0;
    for (; g + 1 < ng; g += 2)
        for (int i = 0; i < 4; ++i)
            acc[i] = accumulate(acc[i], quad(w[i] + 16 * g, a + la * g), sa + 2 * g, sw[i] + 2 * g);
    const bool tail = g < ng;
    for (int i = 0; i < 4; ++i) {
        const ivec p = tail ? pair(w[i] + 16 * g, a + la * g) : ivec();
        out[i] = finish(acc[i], tail, p, sa + 2 * g, sw[i] + 2 * g);
    }
}
template <class L, class R, class Quad, class Pair>
inline float dot1(const L* a, const float* sa, const R* w, const float* sw, int ng, Quad quad, Pair pair) {
    fvec acc = fzero();
    const int la = std::is_same<L, std::int8_t>::value ? 32 : 16;
    int g = 0;
    for (; g + 1 < ng; g += 2) acc = accumulate(acc, quad(w + 16 * g, a + la * g), sa + 2 * g, sw + 2 * g);
    const bool tail = g < ng;
    return finish(acc, tail, tail ? pair(w + 16 * g, a + la * g) : ivec(), sa + 2 * g, sw + 2 * g);
}
}  // namespace simd
#endif

//! The compiled variant's integer sums (the tests compare them with the
//! generic ones).
inline void isums_q8_simd(const std::uint8_t* w, const std::int8_t* a, int n_groups,
                          std::int32_t* out) {
#if defined(IMPBFF_FP4_SIMD)
    for (int g = 0; g < n_groups; ++g) {
        std::int32_t s[4];
        simd::store(simd::pair_q8(w + 16 * g, a + 32 * g), s);
        out[2 * g] = s[0];
        out[2 * g + 1] = s[1];
    }
    for (int g = 0; g + 1 < n_groups; g += 2) {  // the two-group path as well
        std::int32_t s[4];
        simd::store(simd::quad_q8(w + 16 * g, a + 32 * g), s);
        if (s[0] != out[2 * g] || s[1] != out[2 * g + 1] || s[2] != out[2 * g + 2] || s[3] != out[2 * g + 3])
            out[2 * g] = std::numeric_limits<std::int32_t>::min();  // make the test fail
    }
#else
    isums_q8_generic(w, a, n_groups, out);
#endif
}
inline void isums_fp4_simd(const std::uint8_t* x, const std::uint8_t* y, int n_groups,
                           std::int32_t* out) {
#if defined(IMPBFF_FP4_SIMD)
    for (int g = 0; g < n_groups; ++g) {
        std::int32_t s[4];
        simd::store(simd::pair_fp4(x + 16 * g, y + 16 * g), s);
        out[2 * g] = s[0];
        out[2 * g + 1] = s[1];
    }
    for (int g = 0; g + 1 < n_groups; g += 2) {
        std::int32_t s[4];
        simd::store(simd::quad_fp4(x + 16 * g, y + 16 * g), s);
        if (s[0] != out[2 * g] || s[1] != out[2 * g + 1] || s[2] != out[2 * g + 2] || s[3] != out[2 * g + 3])
            out[2 * g] = std::numeric_limits<std::int32_t>::min();
    }
#else
    isums_fp4_generic(x, y, n_groups, out);
#endif
}

// ------------------------------------------------------------------ operands

//! Rows of int8 values (permuted within groups of 32) with one float scale
//! per 16-element sub-block; `kp` is the padded row length.
struct Q8Rows {
    int rows = 0;
    int kp = 0;
    std::vector<std::int8_t> q;
    std::vector<float> sc;
    int n_sub() const { return kp / 16; }
};

//! Rows of packed FP4 codes (`stride` bytes apart) with one float scale per
//! 16-element sub-block that includes the table's factor 1/2. `codes` points
//! into `own` or into a tensor that outlives the operand.
struct Fp4Rows {
    int rows = 0;
    int kp = 0;
    const std::uint8_t* codes = nullptr;
    std::size_t stride = 0;
    std::vector<float> sc;
    std::vector<std::uint8_t> own;
    int n_sub() const { return kp / 16; }
};

//! Position of element `k` of a row in the permuted int8 layout.
inline int permuted(int k) {
    const int g = k / 32, j = k % 32;
    return 32 * g + ((j % 2) ? 16 + j / 2 : j / 2);
}

//! Quantise `rows x K` doubles to int8, one scale a block of `block` (16 or
//! 32) elements: `d = float(amax / 127)`, `q = round(x / d)`, halves away
//! from zero, stored permuted (see the file comment).
inline void quantize_q8(const double* A, int rows, int K, int block, Q8Rows& out) {
    out.rows = rows;
    out.kp = padded_cols(K);
    out.q.assign(static_cast<std::size_t>(rows) * out.kp, 0);
    out.sc.assign(static_cast<std::size_t>(rows) * out.n_sub(), 0.0f);
    std::vector<double> x(static_cast<std::size_t>(out.kp), 0.0);
    for (int r = 0; r < rows; ++r) {
        std::copy(A + static_cast<std::size_t>(r) * K, A + static_cast<std::size_t>(r + 1) * K, x.begin());
        std::int8_t* q = out.q.data() + static_cast<std::size_t>(r) * out.kp;
        float* sc = out.sc.data() + static_cast<std::size_t>(r) * out.n_sub();
        for (int k0 = 0; k0 < out.kp; k0 += block) {
            double amax = 0.0;
            for (int k = k0; k < k0 + block; ++k) amax = std::max(amax, std::abs(x[static_cast<std::size_t>(k)]));
            const float d = static_cast<float>(amax / 127.0);
            for (int s = k0 / 16; s < (k0 + block) / 16; ++s) sc[s] = d;
            const double inv = d > 0.0f ? 1.0 / static_cast<double>(d) : 0.0;
            for (int k = k0; k < k0 + block; ++k) {
                const double v = std::max(-127.0, std::min(127.0, x[static_cast<std::size_t>(k)] * inv));
                const int j = k % 32;
                q[k - j + ((j & 1) ? 16 + j / 2 : j / 2)] = static_cast<std::int8_t>(v + std::copysign(0.5, v));
            }
        }
    }
}

namespace kd {
//! 0.5 x the E4M3 / E8M0 values, as float tables.
struct HalfScaleTables {
    float e4m3[256];
    float e8m0[256];
    HalfScaleTables() {
        for (int c = 0; c < 256; ++c) {
            e4m3[c] = static_cast<float>(0.5 * e4m3_decode(static_cast<std::uint8_t>(c)));
            e8m0[c] = static_cast<float>(0.5 * e8m0_decode(static_cast<std::uint8_t>(c)));
        }
    }
};
inline const HalfScaleTables& half_scales() {
    static const HalfScaleTables t;
    return t;
}
//! 0.5 x a block scale as float (nvfp4: times the float32 tensor scale,
//! the product rounded once).
inline float half_scale(Format f, std::uint8_t code, float g) {
    if (f == Format::MXFP4) return half_scales().e8m0[code];
    return static_cast<float>(static_cast<double>(half_scales().e4m3[code]) * static_cast<double>(g));
}
}  // namespace kd

//! Float sub-block scales (with the 1/2) of an Fp4Tensor's rows.
inline void tensor_subblock_scales(const Fp4Tensor& t, std::vector<float>& sc) {
    const int kp = static_cast<int>(t.padded()), ns = kp / 16;
    sc.assign(static_cast<std::size_t>(t.rows) * ns, 0.0f);
    for (int r = 0; r < t.rows; ++r) {
        const std::uint8_t* s = t.scales.data() + static_cast<std::size_t>(r) * t.row_scale_bytes();
        float* o = sc.data() + static_cast<std::size_t>(r) * ns;
        if (t.format == Format::FP4) {
            const float v = static_cast<float>(0.5 * static_cast<double>(detail::get_f32(s)));
            for (int i = 0; i < ns; ++i) o[i] = v;
        } else {
            for (int i = 0; i < ns; ++i)
                o[i] = kd::half_scale(t.format, s[i * 16 / t.block], t.tensor_scale);
        }
    }
}

//! An operand viewing a tensor's codes.
inline Fp4Rows rows_of(const Fp4Tensor& t) {
    Fp4Rows r;
    r.rows = t.rows;
    r.kp = static_cast<int>(t.padded());
    r.codes = t.codes.data();
    r.stride = t.row_bytes();
    tensor_subblock_scales(t, r.sc);
    return r;
}

//! Quantise `rows x K` doubles to an FP4 operand, blocks along each row;
//! nvfp4 takes one global scale per row (so rows are independent).
inline Fp4Rows quantize_rows(const double* A, int rows, int K, Format f,
                             Rounding rnd = Rounding::NearestEven, SplitMix64* rng = nullptr) {
    Fp4Rows out;
    out.rows = rows;
    out.kp = padded_cols(K);
    out.stride = static_cast<std::size_t>(out.kp / 2);
    out.own.assign(static_cast<std::size_t>(rows) * out.stride, 0);
    out.sc.assign(static_cast<std::size_t>(rows) * out.n_sub(), 0.0f);
    const int block = block_size(f, K);
    std::vector<std::uint8_t> scales(f == Format::FP4 ? 4 : static_cast<std::size_t>(out.kp / block)), tmp;
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * K;
        const float g = f == Format::NVFP4 ? nvfp4_tensor_scale(detail::absmax(x, static_cast<std::size_t>(K)))
                                           : 1.0f;
        encode_row(f, x, K, block, g, out.own.data() + static_cast<std::size_t>(r) * out.stride,
                   scales.data(), tmp, rnd, rng);
        float* o = out.sc.data() + static_cast<std::size_t>(r) * out.n_sub();
        for (int i = 0; i < out.n_sub(); ++i)
            o[i] = f == Format::FP4 ? static_cast<float>(0.5 * static_cast<double>(detail::get_f32(scales.data())))
                                    : kd::half_scale(f, scales[i * 16 / block], g);
    }
    out.codes = out.own.data();
    return out;
}

// ------------------------------------------------------------------ GEMMs

namespace kd {
//! C (M x N) = L R^T over `ng` groups: left rows `lrow(r)` with scales
//! `lsc(r)`, right rows `rrow(o)` / `rsc(o)`. Four right rows at a time
//! through the SIMD dot4 (unless `Generic`), the generic dot otherwise.
template <bool Generic, class L, class LRow, class RRow, class Quad, class Pair, class GDot>
inline void gemm(int M, int N, int ng, LRow lrow, const float* lsc, RRow rrow, const float* rsc,
                 double* C, Quad quad, Pair pair, GDot gdot) {
    const int ns = 2 * ng;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (static_cast<long>(M) * N * ng > (1L << 17))
#endif
    for (int r = 0; r < M; ++r) {
        std::int32_t scratch[4096];
        std::vector<std::int32_t> big;
        std::int32_t* is = scratch;
        if (ns > 4096) { big.resize(static_cast<std::size_t>(ns)); is = big.data(); }
        const L* a = lrow(r);
        const float* sa = lsc + static_cast<std::size_t>(r) * ns;
        double* c = C + static_cast<std::size_t>(r) * N;
        int o = 0;
#if defined(IMPBFF_FP4_SIMD)
        if (!Generic) {
            for (; o + 4 <= N; o += 4) {
                const std::uint8_t* w[4] = {rrow(o), rrow(o + 1), rrow(o + 2), rrow(o + 3)};
                const float* sw[4] = {rsc + static_cast<std::size_t>(o) * ns, rsc + static_cast<std::size_t>(o + 1) * ns,
                                      rsc + static_cast<std::size_t>(o + 2) * ns, rsc + static_cast<std::size_t>(o + 3) * ns};
                float out[4];
                simd::dot4(a, sa, w, sw, ng, out, quad, pair);
                for (int i = 0; i < 4; ++i) c[o + i] = out[i];
            }
            for (; o < N; ++o)
                c[o] = simd::dot1(a, sa, rrow(o), rsc + static_cast<std::size_t>(o) * ns, ng, quad, pair);
        }
#else
        (void)quad;
        (void)pair;
#endif
        for (; o < N; ++o) c[o] = gdot(rrow(o), a, sa, rsc + static_cast<std::size_t>(o) * ns, ng, is);
    }
}
}  // namespace kd

//! C (A.rows x W.rows, row-major) = A W^T, A int8, W FP4. `Generic` picks
//! the scalar integer kernel (tests); otherwise the compiled SIMD variant.
template <bool Generic = false>
inline void gemm_q8(const Q8Rows& A, const Fp4Rows& W, double* C) {
    const auto lrow = [&](int r) { return A.q.data() + static_cast<std::size_t>(r) * A.kp; };
    const auto rrow = [&](int o) { return W.codes + static_cast<std::size_t>(o) * W.stride; };
    const auto gdot = [](const std::uint8_t* w, const std::int8_t* a, const float* sa, const float* sw, int ng,
                         std::int32_t* is) { return dot_q8_generic(w, a, sa, sw, ng, is); };
#if defined(IMPBFF_FP4_SIMD)
    const auto quad = [](const std::uint8_t* w, const std::int8_t* a) { return simd::quad_q8(w, a); };
    const auto pair = [](const std::uint8_t* w, const std::int8_t* a) { return simd::pair_q8(w, a); };
#else
    const int quad = 0, pair = 0;
#endif
    kd::gemm<Generic, std::int8_t>(A.rows, W.rows, A.kp / 32, lrow, A.sc.data(), rrow, W.sc.data(), C,
                                       quad, pair, gdot);
}

//! C (X.rows x Y.rows, row-major) = X Y^T, both FP4.
template <bool Generic = false>
inline void gemm_fp4(const Fp4Rows& X, const Fp4Rows& Y, double* C) {
    const auto lrow = [&](int r) { return X.codes + static_cast<std::size_t>(r) * X.stride; };
    const auto rrow = [&](int o) { return Y.codes + static_cast<std::size_t>(o) * Y.stride; };
    const auto gdot = [](const std::uint8_t* y, const std::uint8_t* x, const float* sx, const float* sy, int ng,
                         std::int32_t* is) { return dot_fp4_generic(x, y, sx, sy, ng, is); };
#if defined(IMPBFF_FP4_SIMD)
    const auto quad = [](const std::uint8_t* y, const std::uint8_t* x) { return simd::quad_fp4(x, y); };
    const auto pair = [](const std::uint8_t* y, const std::uint8_t* x) { return simd::pair_fp4(x, y); };
#else
    const int quad = 0, pair = 0;
#endif
    kd::gemm<Generic, std::uint8_t>(X.rows, Y.rows, X.kp / 32, lrow, X.sc.data(), rrow, Y.sc.data(), C,
                                        quad, pair, gdot);
}

// ------------------------------------------------------------------ forward

//! int8 activation block for a weight format: 16 for nvfp4, else 32.
inline int q8_block(Format f) { return f == Format::NVFP4 ? 16 : 32; }

//! The forward pass on the packed codes, in the model's physical units.
/*! Without `quantize_activations` each layer input is quantised to int8
    per block (W4A8, ggml's scheme) and multiplied by gemm_q8; with it, to
    FP4 in the model's format per row (W4A4) and multiplied by gemm_fp4.
    Bias, activation and scalers in double; a full-precision layer runs in
    double through `Gemm`. */
template <class Gemm = mlpcore::PortableGemm>
inline void predict(const Fp4Model& m, const double* X, int n_rows, std::vector<double>& y) {
    y.clear();
    if (n_rows <= 0 || m.layers.empty()) return;
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    std::vector<double> a(X, X + rows * static_cast<std::size_t>(m.n_inputs())), z;
    mlpcore::detail::scale_in(a, n_rows, m.n_inputs(), m.x_scaler);
    Q8Rows q8;
    for (const Fp4Layer& l : m.layers) {
        z.resize(rows * static_cast<std::size_t>(l.n_out));
        if (l.full_precision) {
            Gemm::nt(n_rows, l.n_out, l.n_in, a.data(), l.weight_f64.data(), z.data());
        } else if (m.quantize_activations) {
            const Fp4Rows W = rows_of(l.weight);
            gemm_fp4(quantize_rows(a.data(), n_rows, l.n_in, m.format), W, z.data());
        } else {
            const Fp4Rows W = rows_of(l.weight);
            quantize_q8(a.data(), n_rows, l.n_in, q8_block(m.format), q8);
            gemm_q8(q8, W, z.data());
        }
        for (std::size_t r = 0; r < rows; ++r) {
            double* zr = z.data() + r * static_cast<std::size_t>(l.n_out);
            for (int o = 0; o < l.n_out; ++o) zr[o] += l.bias[static_cast<std::size_t>(o)];
        }
        a.resize(z.size());
        mlpcore::act_apply(z.data(), a.data(), z.size(), l.activation);
    }
    y.swap(a);
    mlpcore::detail::unscale_out(y, n_rows, m.n_outputs(), m.y_scaler);
}

}  // namespace kern
}  // namespace mlpfp4
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPFP4KERNELS_H

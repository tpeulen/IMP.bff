/**
 *  \file IMP/bff/internal/MlpFp4Kernels.h
 *  \brief Integer-SIMD kernels on packed FP4 (E2M1) codes: the blocked FP4
 *         GEMM, the vectorised quantisers, and the fast forward pass.
 *
 * **CPUs have no FP4 arithmetic.** "Native" here means every product runs on
 * the 4-bit codes through integer SIMD dot products; nothing is dequantised
 * to float first. The approach is llama.cpp/ggml's
 * (`ggml_vec_dot_mxfp4_q8_0`, `ggml_vec_dot_nvfp4_q8_0`), re-expressed as a
 * register-blocked GEMM:
 *
 * - The E2M1 values times two are the integers
 *   `{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}`
 *   (ggml's `kvalues_mxfp4`), so a 16-byte table lookup turns 16 nibbles
 *   into 16 int8 values at once: `vqtbl1q_s8` (NEON), `_mm256_shuffle_epi8`
 *   (AVX2), `_mm512_shuffle_epi8` (AVX-512BW). The factor 1/2 goes into the
 *   scales.
 * - The **left operand** (activations, or in training the gradient) is int8
 *   in natural element order with one float scale per 16-element sub-block:
 *   W4A8 quantises the activations to int8 per block (32; 16 for nvfp4,
 *   ggml's Q8_0 idea); W4A4 and training quantise them to FP4 and store the
 *   codes' values `2 x E2M1` (exact, in [-12, 12]).
 * - The **right operand** (weights; in wgrad the Hadamard-transformed
 *   activations) stays packed FP4, repacked once into the micro-kernel's
 *   tile layout (PackedRight): for a tile of `kNR` output rows and a 16-wide
 *   sub-block, two chunks of `4 kNR` bytes whose low nibbles are k-quad
 *   `2p` and high nibbles k-quad `2p + 1` of every row of the tile (byte
 *   `4 o + j` = row o, element j of the quad). One table lookup of a chunk
 *   is then exactly the operand of a lane-wise 4-byte dot product:
 *   `vdotq_laneq_s32` (ARMv8.2 dotprod; plain NEON: `vmull_s8` +
 *   `vpaddlq_s16`), `_mm256_maddubs_epi16` + `_mm256_madd_epi16` (AVX2),
 *   `_mm512_dpbusd_epi32` (AVX-512 VNNI) -- each lane is one output row, the
 *   left operand's 4 bytes broadcast. x86's u8 x s8 instructions take the
 *   left operand offset by 128 (`a ^ 0x80`) and subtract `128 x` the tile
 *   row's sub-block sum, precomputed at packing (`corr`); the sums are exact.
 * - The micro-kernel holds `kMR` left rows x `kNR` right rows (NEON 4 x 4,
 *   AVX2 2 x 8, AVX-512 4 x 16) and walks the contraction one 16-element
 *   sub-block at a time: the exact int32 sub-block sums, then in float
 *   `acc[s % 4] += float(sum) * (scale_left * scale_right)` per output, and
 *   finally `(acc0 + acc1) + (acc2 + acc3)`. That float sequence is the
 *   generic `combine()`, done per output lane, and the multiply is kept
 *   apart from the add (`IMPBFF_FP4_KEEP`, an empty asm that stops GCC and
 *   Clang contracting it into an FMA), so **every variant, and every
 *   compiler flag set, gives the generic kernel's bits**.
 *
 * Quantisation (the per-call and per-step cost around the GEMMs) is
 * vectorised the same way: block absmax, the E2M1 threshold network, the
 * int8 rounding and the stochastic-rounding comparison run on 2 (NEON), 4
 * (AVX2) or 8 (AVX-512) doubles; the per-block scale codes are scalar but
 * bit-level (MlpFp4.h). Each vector step performs exactly the scalar
 * reference's IEEE operations, so the codes are identical.
 *
 * Variants are chosen at **compile time** from the target macros --
 * `__ARM_FEATURE_DOTPROD` (NEON + dotprod), `__ARM_NEON` (NEON),
 * `__AVX512F__ && __AVX512BW__ && __AVX512VNNI__` (AVX-512 VNNI),
 * `__AVX2__` -- never by run-time cpuid (under Rosetta 2 cpuid reports no
 * AVX2 while AVX2 instructions execute). `IMPBFF_FP4_NO_SIMD` forces the
 * generic scalar code; `IMPBFF_FP4_NO_DOTPROD` forces plain NEON where the
 * toolchain enables dotprod regardless of `-march`. `kernel_name()` says which was compiled.
 *
 * Derived from llama.cpp / ggml (https://github.com/ggml-org/llama.cpp):
 * ggml/src/ggml-common.h (`kvalues_mxfp4`, `block_mxfp4`, `block_nvfp4`),
 * ggml/src/ggml-cpu/quants.c (`ggml_vec_dot_mxfp4_q8_0_generic`,
 * `ggml_vec_dot_nvfp4_q8_0_generic`),
 * ggml/src/ggml-cpu/arch/arm/quants.c (`ggml_vec_dot_mxfp4_q8_0`,
 * `ggml_vec_dot_nvfp4_q8_0`, NEON) and
 * ggml/src/ggml-cpu/arch/x86/quants.c (`ggml_vec_dot_mxfp4_q8_0`,
 * `ggml_vec_dot_nvfp4_q8_0`, AVX2 / AVX-512: the table lookup through
 * `shuffle_epi8`, the u8 x s8 `maddubs` / `dpbusd` products), under this
 * licence:
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
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if !defined(IMPBFF_FP4_NO_SIMD)
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_DOTPROD) && !defined(IMPBFF_FP4_NO_DOTPROD)
#define IMPBFF_FP4_NEON_DOTPROD 1
#else
#define IMPBFF_FP4_NEON 1
#endif
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define IMPBFF_FP4_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define IMPBFF_FP4_AVX2 1
#endif
#endif

// Keep `v` a separately rounded value: an empty asm the optimiser cannot see
// through, so `acc + v` is never contracted with the multiply that made `v`
// into a fused multiply-add (GCC contracts across statements by default,
// -ffp-contract=fast; an FMA rounds once, the generic code twice).
#if defined(__GNUC__) || defined(__clang__)
#if defined(__aarch64__)
#define IMPBFF_FP4_KEEP(v) __asm__("" : "+w"(v))
#elif defined(__x86_64__) || defined(__i386__)
#if defined(IMPBFF_FP4_AVX512)
#define IMPBFF_FP4_KEEP(v) __asm__("" : "+v"(v))
#else
#define IMPBFF_FP4_KEEP(v) __asm__("" : "+x"(v))
#endif
#else
#define IMPBFF_FP4_KEEP(v) __asm__("" : "+m"(v))
#endif
#else
#define IMPBFF_FP4_KEEP(v) ((void)0)
#endif

namespace IMP {
namespace bff {
namespace internal {
namespace mlpfp4 {
namespace kern {

//! E2M1 values times two, by code (ggml's kvalues_mxfp4).
alignas(16) constexpr std::int8_t kValues2[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                                  0, -1, -2, -3, -4, -6, -8, -12};

#if defined(IMPBFF_FP4_AVX512)
constexpr int kNR = 16;  //!< right rows (outputs) a micro-kernel tile
constexpr int kMR = 4;   //!< left rows a micro-kernel tile
#elif defined(IMPBFF_FP4_AVX2)
constexpr int kNR = 8;
constexpr int kMR = 2;
#else
constexpr int kNR = 4;
constexpr int kMR = 4;
#endif

//! Which variant this translation unit compiled.
inline const char* kernel_name() {
#if defined(IMPBFF_FP4_NEON_DOTPROD)
    return "neon-dotprod";
#elif defined(IMPBFF_FP4_NEON)
    return "neon";
#elif defined(IMPBFF_FP4_AVX512)
    return "avx512-vnni";
#elif defined(IMPBFF_FP4_AVX2)
    return "avx2";
#else
    return "generic";
#endif
}

// ------------------------------------------------------------------ operands

//! Left operand: `rows` rows of int8 values in natural element order,
//! padded with zeros to `kp` (a multiple of 32), and one float scale per
//! 16-element sub-block (`sc`, rows x kp/16). W4A8: activations quantised to
//! int8; W4A4 / training: `2 x E2M1` of FP4 codes, the scale with the 1/2.
struct Q8Rows {
    int rows = 0;
    int kp = 0;
    std::vector<std::int8_t> q;
    std::vector<float> sc;
    int n_sub() const { return kp / 16; }
};

//! Rows of packed FP4 codes in bff's standard packing (`stride` bytes
//! apart) with one float scale per 16-element sub-block that includes the
//! table's factor 1/2. `codes` points into `own` or into a tensor that
//! outlives the operand.
struct Fp4Rows {
    int rows = 0;
    int kp = 0;
    const std::uint8_t* codes = nullptr;
    std::size_t stride = 0;
    std::vector<float> sc;
    std::vector<std::uint8_t> own;
    int n_sub() const { return kp / 16; }
};

//! Right operand repacked for the micro-kernel (see the file comment):
//! `nt = ceil(rows / kNR)` tiles, each `ns = kp / 16` sub-blocks of
//! `8 kNR` code bytes, `kNR` float scales and `kNR` int32 corrections
//! (`128 x` the row's sub-block sum of `2 x E2M1`). Rows past `rows` are
//! zero codes with zero scale.
struct PackedRight {
    int rows = 0;
    int kp = 0;
    int ns = 0;
    int nt = 0;
    std::vector<std::uint8_t> codes;
    std::vector<float> sc;
    std::vector<std::int32_t> corr;
};

// ------------------------------------------------------------------ scales

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

// ------------------------------------------------------------------ vector quantisation primitives
//
// Each does, lane by lane, exactly the scalar reference's IEEE operations:
//   absmax      -- std::max(a, std::abs(x)) from 0 (NaN never wins),
//   encode_rne  -- e2m1_encode(x / d),
//   encode_sr   -- e2m1_encode_sr(x / d, u),
//   round_q8    -- int8(v + copysign(0.5, v)), v = max(-127, min(127, x * inv)).

namespace vq {

inline double absmax_scalar(const double* x, int n) {
    double a = 0.0;
    for (int i = 0; i < n; ++i) a = std::max(a, std::abs(x[i]));
    return a;
}
inline std::int8_t round_q8_scalar(double x, double inv) {
    const double v = std::max(-127.0, std::min(127.0, x * inv));
    return static_cast<std::int8_t>(v + std::copysign(0.5, v));
}

//! Whether `d` is a power of two whose reciprocal is normal: then x * (1/d)
//! is x / d bit for bit (the same exact value, rounded once), and the
//! encoders multiply instead of divide (every mxfp4 block).
inline bool pow2_scale(double d) {
    const std::uint64_t u = detail::bits_of(d);
    const int be = static_cast<int>((u >> 52) & 0x7FF);
    return (u & ((1ULL << 52) - 1)) == 0 && be > 1 && be < 2045 && !(u >> 63);
}

#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
template <bool P2>
inline float64x2_t quot(float64x2_t x, float64x2_t d, float64x2_t inv) { return P2 ? vmulq_f64(x, inv) : vdivq_f64(x, d); }
inline double absmax(const double* x, int n) {
    float64x2_t a0 = vdupq_n_f64(0.0), a1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 = vmaxnmq_f64(a0, vabsq_f64(vld1q_f64(x + i)));
        a1 = vmaxnmq_f64(a1, vabsq_f64(vld1q_f64(x + i + 2)));
    }
    double a = vmaxnmvq_f64(vmaxnmq_f64(a0, a1));
    for (; i < n; ++i) a = std::max(a, std::abs(x[i]));
    return a;
}
//! Codes of two quotients q (NaN -> 0).
inline uint64x2_t e2m1_codes(float64x2_t q) {
    const float64x2_t m = vabsq_f64(q);
    int64x2_t c = vreinterpretq_s64_u64(vcgtq_f64(m, vdupq_n_f64(0.25)));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgeq_f64(m, vdupq_n_f64(0.75))));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgtq_f64(m, vdupq_n_f64(1.25))));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgeq_f64(m, vdupq_n_f64(1.75))));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgtq_f64(m, vdupq_n_f64(2.5))));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgeq_f64(m, vdupq_n_f64(3.5))));
    c = vaddq_s64(c, vreinterpretq_s64_u64(vcgtq_f64(m, vdupq_n_f64(5.0))));
    const uint64x2_t sign = vshlq_n_u64(vshrq_n_u64(vreinterpretq_u64_f64(q), 63), 3);
    return vandq_u64(vorrq_u64(vreinterpretq_u64_s64(vnegq_s64(c)), sign), vceqq_f64(q, q));
}
//! Codes of four quotients narrowed to float32 with round-to-odd (FCVTXN):
//! a double and its round-to-odd float compare alike with every float of
//! even last mantissa bit -- all seven thresholds -- so the codes are exact.
inline uint32x4_t e2m1_codes_f32(float32x4_t q) {
    const float32x4_t m = vabsq_f32(q);
    int32x4_t c = vreinterpretq_s32_u32(vcgtq_f32(m, vdupq_n_f32(0.25f)));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgeq_f32(m, vdupq_n_f32(0.75f))));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgtq_f32(m, vdupq_n_f32(1.25f))));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgeq_f32(m, vdupq_n_f32(1.75f))));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgtq_f32(m, vdupq_n_f32(2.5f))));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgeq_f32(m, vdupq_n_f32(3.5f))));
    c = vaddq_s32(c, vreinterpretq_s32_u32(vcgtq_f32(m, vdupq_n_f32(5.0f))));
    const uint32x4_t sign = vshlq_n_u32(vshrq_n_u32(vreinterpretq_u32_f32(q), 31), 3);
    return vandq_u32(vorrq_u32(vreinterpretq_u32_s32(vnegq_s32(c)), sign), vceqq_f32(q, q));
}
template <bool P2>
inline void encode_rne_t(const double* x, int n, double d, std::uint8_t* out) {
    const float64x2_t dv = vdupq_n_f64(d), iv = vdupq_n_f64(1.0 / d);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t q0 = vcvtx_high_f32_f64(vcvtx_f32_f64(quot<P2>(vld1q_f64(x + i), dv, iv)),
                                                  quot<P2>(vld1q_f64(x + i + 2), dv, iv));
        const float32x4_t q1 = vcvtx_high_f32_f64(vcvtx_f32_f64(quot<P2>(vld1q_f64(x + i + 4), dv, iv)),
                                                  quot<P2>(vld1q_f64(x + i + 6), dv, iv));
        const uint16x8_t c = vcombine_u16(vmovn_u32(e2m1_codes_f32(q0)), vmovn_u32(e2m1_codes_f32(q1)));
        vst1_u8(out + i, vmovn_u16(c));
    }
    for (; i + 2 <= n; i += 2) {
        const uint64x2_t c = e2m1_codes(quot<P2>(vld1q_f64(x + i), dv, iv));
        out[i] = static_cast<std::uint8_t>(vgetq_lane_u64(c, 0));
        out[i + 1] = static_cast<std::uint8_t>(vgetq_lane_u64(c, 1));
    }
    for (; i < n; ++i) out[i] = e2m1_encode(x[i] / d);
}
//! Stochastic rounding of four quotients: the neighbour index, lo and
//! 1 / (hi - lo) from the round-to-odd float32 copy (thresholds 0.5 .. 6 have
//! even mantissas, lo and 1/step are small dyadics: exact), the comparison
//! `u < (m - lo) / step` in double on the original quotients.
inline uint32x4_t e2m1_codes_sr(float64x2_t qa, float64x2_t qb, const double* u) {
    const float32x4_t q = vcvtx_high_f32_f64(vcvtx_f32_f64(qa), qb);
    const float32x4_t m = vabsq_f32(q);
    const float32x4_t h = vdupq_n_f32(0.5f), one = vdupq_n_f32(1.0f);
    const uint32x4_t g05 = vcgeq_f32(m, h), g1 = vcgeq_f32(m, one), g15 = vcgeq_f32(m, vdupq_n_f32(1.5f)),
                     g2 = vcgeq_f32(m, vdupq_n_f32(2.0f)), g3 = vcgeq_f32(m, vdupq_n_f32(3.0f)),
                     g4 = vcgeq_f32(m, vdupq_n_f32(4.0f)), g6 = vcgeq_f32(m, vdupq_n_f32(6.0f));
    int32x4_t c = vaddq_s32(vaddq_s32(vreinterpretq_s32_u32(g05), vreinterpretq_s32_u32(g1)),
                            vaddq_s32(vreinterpretq_s32_u32(g15), vreinterpretq_s32_u32(g2)));
    c = vaddq_s32(c, vaddq_s32(vreinterpretq_s32_u32(g3), vreinterpretq_s32_u32(g4)));
    const uint32x4_t hb = vreinterpretq_u32_f32(h), ob = vreinterpretq_u32_f32(one);
    float32x4_t lo = vaddq_f32(vaddq_f32(vreinterpretq_f32_u32(vandq_u32(g05, hb)), vreinterpretq_f32_u32(vandq_u32(g1, hb))),
                               vaddq_f32(vreinterpretq_f32_u32(vandq_u32(g15, hb)), vreinterpretq_f32_u32(vandq_u32(g2, hb))));
    lo = vaddq_f32(lo, vaddq_f32(vreinterpretq_f32_u32(vandq_u32(g3, ob)), vreinterpretq_f32_u32(vandq_u32(g4, ob))));
    const float32x4_t inv = vsubq_f32(vsubq_f32(vdupq_n_f32(2.0f), vreinterpretq_f32_u32(vandq_u32(g2, ob))),
                                      vreinterpretq_f32_u32(vandq_u32(g4, hb)));
    const float64x2_t pa = vmulq_f64(vsubq_f64(vabsq_f64(qa), vcvt_f64_f32(vget_low_f32(lo))), vcvt_f64_f32(vget_low_f32(inv)));
    const float64x2_t pb = vmulq_f64(vsubq_f64(vabsq_f64(qb), vcvt_high_f64_f32(lo)), vcvt_high_f64_f32(inv));
    const uint32x4_t up = vcombine_u32(vmovn_u64(vcltq_f64(vld1q_f64(u), pa)), vmovn_u64(vcltq_f64(vld1q_f64(u + 2), pb)));
    c = vaddq_s32(c, vreinterpretq_s32_u32(up));  // c is -count
    uint32x4_t code = vbslq_u32(g6, vdupq_n_u32(7), vreinterpretq_u32_s32(vnegq_s32(c)));
    const uint32x4_t sign = vshlq_n_u32(vshrq_n_u32(vreinterpretq_u32_f32(q), 31), 3);
    return vandq_u32(vorrq_u32(code, sign), vceqq_f32(q, q));
}
template <bool P2>
inline void encode_sr_t(const double* x, int n, double d, const double* u, std::uint8_t* out) {
    const float64x2_t dv = vdupq_n_f64(d), iv = vdupq_n_f64(1.0 / d);
    const float64x2_t h = vdupq_n_f64(0.5), one = vdupq_n_f64(1.0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint32x4_t c0 = e2m1_codes_sr(quot<P2>(vld1q_f64(x + i), dv, iv), quot<P2>(vld1q_f64(x + i + 2), dv, iv), u + i);
        const uint32x4_t c1 = e2m1_codes_sr(quot<P2>(vld1q_f64(x + i + 4), dv, iv), quot<P2>(vld1q_f64(x + i + 6), dv, iv), u + i + 4);
        vst1_u8(out + i, vmovn_u16(vcombine_u16(vmovn_u32(c0), vmovn_u32(c1))));
    }
    for (; i + 2 <= n; i += 2) {
        const float64x2_t q = quot<P2>(vld1q_f64(x + i), dv, iv);
        const float64x2_t m = vabsq_f64(q);
        const uint64x2_t g05 = vcgeq_f64(m, h), g1 = vcgeq_f64(m, one), g15 = vcgeq_f64(m, vdupq_n_f64(1.5)),
                         g2 = vcgeq_f64(m, vdupq_n_f64(2.0)), g3 = vcgeq_f64(m, vdupq_n_f64(3.0)),
                         g4 = vcgeq_f64(m, vdupq_n_f64(4.0)), g6 = vcgeq_f64(m, vdupq_n_f64(6.0));
        int64x2_t c = vaddq_s64(vaddq_s64(vreinterpretq_s64_u64(g05), vreinterpretq_s64_u64(g1)),
                                vaddq_s64(vreinterpretq_s64_u64(g15), vreinterpretq_s64_u64(g2)));
        c = vaddq_s64(c, vaddq_s64(vreinterpretq_s64_u64(g3), vreinterpretq_s64_u64(g4)));
        // lo = kE2M1[i] and 1 / (hi - lo), both exact
        float64x2_t lo = vaddq_f64(vaddq_f64(vreinterpretq_f64_u64(vandq_u64(g05, vreinterpretq_u64_f64(h))),
                                             vreinterpretq_f64_u64(vandq_u64(g1, vreinterpretq_u64_f64(h)))),
                                   vaddq_f64(vreinterpretq_f64_u64(vandq_u64(g15, vreinterpretq_u64_f64(h))),
                                             vreinterpretq_f64_u64(vandq_u64(g2, vreinterpretq_u64_f64(h)))));
        lo = vaddq_f64(lo, vaddq_f64(vreinterpretq_f64_u64(vandq_u64(g3, vreinterpretq_u64_f64(one))),
                                     vreinterpretq_f64_u64(vandq_u64(g4, vreinterpretq_u64_f64(one)))));
        const float64x2_t inv = vsubq_f64(vsubq_f64(vdupq_n_f64(2.0), vreinterpretq_f64_u64(vandq_u64(g2, vreinterpretq_u64_f64(one)))),
                                          vreinterpretq_f64_u64(vandq_u64(g4, vreinterpretq_u64_f64(h))));
        const uint64x2_t up = vcltq_f64(vld1q_f64(u + i), vmulq_f64(vsubq_f64(m, lo), inv));
        c = vaddq_s64(c, vreinterpretq_s64_u64(up));  // c is -count: add -1 to count one more
        uint64x2_t code = vreinterpretq_u64_s64(vnegq_s64(c));
        code = vbslq_u64(g6, vdupq_n_u64(7), code);
        const uint64x2_t sign = vshlq_n_u64(vshrq_n_u64(vreinterpretq_u64_f64(q), 63), 3);
        code = vandq_u64(vorrq_u64(code, sign), vceqq_f64(q, q));
        out[i] = static_cast<std::uint8_t>(vgetq_lane_u64(code, 0));
        out[i + 1] = static_cast<std::uint8_t>(vgetq_lane_u64(code, 1));
    }
    for (; i < n; ++i) out[i] = e2m1_encode_sr(x[i] / d, u[i]);
}
inline void round_q8(const double* x, int n, double inv, std::int8_t* out) {
    const float64x2_t iv = vdupq_n_f64(inv), hi = vdupq_n_f64(127.0), lo = vdupq_n_f64(-127.0);
    const uint64x2_t sm = vdupq_n_u64(0x8000000000000000ULL), half = vreinterpretq_u64_f64(vdupq_n_f64(0.5));
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vmulq_f64(vld1q_f64(x + i), iv);
        v = vbslq_f64(vcltq_f64(v, hi), v, hi);  // std::min(127, v)
        v = vbslq_f64(vcltq_f64(lo, v), v, lo);  // std::max(-127, v)
        const float64x2_t r = vaddq_f64(v, vreinterpretq_f64_u64(vorrq_u64(vandq_u64(vreinterpretq_u64_f64(v), sm), half)));
        const int64x2_t t = vcvtq_s64_f64(r);
        out[i] = static_cast<std::int8_t>(vgetq_lane_s64(t, 0));
        out[i + 1] = static_cast<std::int8_t>(vgetq_lane_s64(t, 1));
    }
    for (; i < n; ++i) out[i] = round_q8_scalar(x[i], inv);
}
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
inline double absmax(const double* x, int n) {
    const __m256d sm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {  // max_pd(|x|, acc) returns acc when |x| is NaN
        a0 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x + i), sm), a0);
        a1 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x + i + 4), sm), a1);
    }
    double l[8];
    _mm256_storeu_pd(l, a0);
    _mm256_storeu_pd(l + 4, a1);
    double a = 0.0;
    for (double v : l) a = std::max(a, v);
    for (; i < n; ++i) a = std::max(a, std::abs(x[i]));
    return a;
}
#if defined(IMPBFF_FP4_AVX512)
template <bool P2>
inline __m512d quot(__m512d x, __m512d d, __m512d inv) { return P2 ? _mm512_mul_pd(x, inv) : _mm512_div_pd(x, d); }
inline __m512i e2m1_codes(__m512d q) {
    const __m512d m = _mm512_abs_pd(q);
    const __m512i one = _mm512_set1_epi64(1);
    __m512i c = _mm512_setzero_si512();
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(0.25), _CMP_GT_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(0.75), _CMP_GE_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(1.25), _CMP_GT_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(1.75), _CMP_GE_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(2.5), _CMP_GT_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(3.5), _CMP_GE_OQ), c, one);
    c = _mm512_mask_add_epi64(c, _mm512_cmp_pd_mask(m, _mm512_set1_pd(5.0), _CMP_GT_OQ), c, one);
    const __m512i sign = _mm512_slli_epi64(_mm512_srli_epi64(_mm512_castpd_si512(q), 63), 3);
    return _mm512_maskz_or_epi64(_mm512_cmp_pd_mask(q, q, _CMP_ORD_Q), c, sign);
}
template <bool P2>
inline void encode_rne_t(const double* x, int n, double d, std::uint8_t* out) {
    const __m512d dv = _mm512_set1_pd(d), iv = _mm512_set1_pd(1.0 / d);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i),
                         _mm512_cvtepi64_epi8(e2m1_codes(quot<P2>(_mm512_loadu_pd(x + i), dv, iv))));
    for (; i < n; ++i) out[i] = e2m1_encode(x[i] / d);
}
template <bool P2>
inline void encode_sr_t(const double* x, int n, double d, const double* u, std::uint8_t* out) {
    const __m512d iv = _mm512_set1_pd(1.0 / d);
    const __m512d dv = _mm512_set1_pd(d), h = _mm512_set1_pd(0.5), one = _mm512_set1_pd(1.0);
    const __m512i ione = _mm512_set1_epi64(1);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m512d q = quot<P2>(_mm512_loadu_pd(x + i), dv, iv);
        const __m512d m = _mm512_abs_pd(q);
        const __mmask8 g05 = _mm512_cmp_pd_mask(m, h, _CMP_GE_OQ), g1 = _mm512_cmp_pd_mask(m, one, _CMP_GE_OQ),
                       g15 = _mm512_cmp_pd_mask(m, _mm512_set1_pd(1.5), _CMP_GE_OQ),
                       g2 = _mm512_cmp_pd_mask(m, _mm512_set1_pd(2.0), _CMP_GE_OQ),
                       g3 = _mm512_cmp_pd_mask(m, _mm512_set1_pd(3.0), _CMP_GE_OQ),
                       g4 = _mm512_cmp_pd_mask(m, _mm512_set1_pd(4.0), _CMP_GE_OQ),
                       g6 = _mm512_cmp_pd_mask(m, _mm512_set1_pd(6.0), _CMP_GE_OQ);
        __m512i c = _mm512_setzero_si512();
        c = _mm512_mask_add_epi64(c, g05, c, ione);
        c = _mm512_mask_add_epi64(c, g1, c, ione);
        c = _mm512_mask_add_epi64(c, g15, c, ione);
        c = _mm512_mask_add_epi64(c, g2, c, ione);
        c = _mm512_mask_add_epi64(c, g3, c, ione);
        c = _mm512_mask_add_epi64(c, g4, c, ione);
        const __m512d z = _mm512_setzero_pd();
        __m512d lo = _mm512_add_pd(_mm512_add_pd(_mm512_mask_mov_pd(z, g05, h), _mm512_mask_mov_pd(z, g1, h)),
                                   _mm512_add_pd(_mm512_mask_mov_pd(z, g15, h), _mm512_mask_mov_pd(z, g2, h)));
        lo = _mm512_add_pd(lo, _mm512_add_pd(_mm512_mask_mov_pd(z, g3, one), _mm512_mask_mov_pd(z, g4, one)));
        const __m512d inv = _mm512_sub_pd(_mm512_sub_pd(_mm512_set1_pd(2.0), _mm512_mask_mov_pd(z, g2, one)),
                                          _mm512_mask_mov_pd(z, g4, h));
        const __mmask8 up = _mm512_cmp_pd_mask(_mm512_loadu_pd(u + i), _mm512_mul_pd(_mm512_sub_pd(m, lo), inv), _CMP_LT_OQ);
        c = _mm512_mask_add_epi64(c, up, c, ione);
        c = _mm512_mask_mov_epi64(c, g6, _mm512_set1_epi64(7));
        const __m512i sign = _mm512_slli_epi64(_mm512_srli_epi64(_mm512_castpd_si512(q), 63), 3);
        c = _mm512_maskz_or_epi64(_mm512_cmp_pd_mask(q, q, _CMP_ORD_Q), c, sign);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i), _mm512_cvtepi64_epi8(c));
    }
    for (; i < n; ++i) out[i] = e2m1_encode_sr(x[i] / d, u[i]);
}
inline void round_q8(const double* x, int n, double inv, std::int8_t* out) {
    const __m512d iv = _mm512_set1_pd(inv), hi = _mm512_set1_pd(127.0), lo = _mm512_set1_pd(-127.0);
    const __m512i sm = _mm512_set1_epi64(static_cast<long long>(0x8000000000000000ULL));
    const __m512i half = _mm512_castpd_si512(_mm512_set1_pd(0.5));
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_mul_pd(_mm512_loadu_pd(x + i), iv);
        v = _mm512_min_pd(v, hi);  // (v < 127) ? v : 127, NaN -> 127, as std::min(127, v)
        v = _mm512_max_pd(v, lo);  // (v > -127) ? v : -127, as std::max(-127, v)
        const __m512d r = _mm512_add_pd(v, _mm512_castsi512_pd(_mm512_or_si512(_mm512_and_si512(_mm512_castpd_si512(v), sm), half)));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i),
                         _mm512_cvtepi32_epi8(_mm512_castsi256_si512(_mm512_cvttpd_epi32(r))));
    }
    for (; i < n; ++i) out[i] = round_q8_scalar(x[i], inv);
}
#else  // AVX2
template <bool P2>
inline __m256d quot(__m256d x, __m256d d, __m256d inv) { return P2 ? _mm256_mul_pd(x, inv) : _mm256_div_pd(x, d); }
inline __m256i e2m1_codes(__m256d q) {
    const __m256d m = _mm256_and_pd(q, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    __m256i c = _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(0.25), _CMP_GT_OQ));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(0.75), _CMP_GE_OQ)));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(1.25), _CMP_GT_OQ)));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(1.75), _CMP_GE_OQ)));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(2.5), _CMP_GT_OQ)));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(3.5), _CMP_GE_OQ)));
    c = _mm256_add_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd(m, _mm256_set1_pd(5.0), _CMP_GT_OQ)));
    c = _mm256_sub_epi64(_mm256_setzero_si256(), c);
    const __m256i sign = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_castpd_si256(q), 63), 3);
    return _mm256_and_si256(_mm256_or_si256(c, sign), _mm256_castpd_si256(_mm256_cmp_pd(q, q, _CMP_ORD_Q)));
}
//! The low byte of each 64-bit lane, to 4 bytes.
inline void store4(__m256i c, std::uint8_t* out) {
    const __m256i b = _mm256_shuffle_epi8(c, _mm256_setr_epi8(0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    const int lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(b)) & 0xFFFF;
    const int hi = _mm_cvtsi128_si32(_mm256_extracti128_si256(b, 1)) & 0xFFFF;
    const std::uint32_t v = static_cast<std::uint32_t>(lo | (hi << 16));
    std::memcpy(out, &v, 4);
}
template <bool P2>
inline void encode_rne_t(const double* x, int n, double d, std::uint8_t* out) {
    const __m256d dv = _mm256_set1_pd(d), iv = _mm256_set1_pd(1.0 / d);
    int i = 0;
    for (; i + 4 <= n; i += 4) store4(e2m1_codes(quot<P2>(_mm256_loadu_pd(x + i), dv, iv)), out + i);
    for (; i < n; ++i) out[i] = e2m1_encode(x[i] / d);
}
template <bool P2>
inline void encode_sr_t(const double* x, int n, double d, const double* u, std::uint8_t* out) {
    const __m256d iv = _mm256_set1_pd(1.0 / d);
    const __m256d dv = _mm256_set1_pd(d), h = _mm256_set1_pd(0.5), one = _mm256_set1_pd(1.0);
    const __m256d am = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d q = quot<P2>(_mm256_loadu_pd(x + i), dv, iv);
        const __m256d m = _mm256_and_pd(q, am);
        const __m256d g05 = _mm256_cmp_pd(m, h, _CMP_GE_OQ), g1 = _mm256_cmp_pd(m, one, _CMP_GE_OQ),
                      g15 = _mm256_cmp_pd(m, _mm256_set1_pd(1.5), _CMP_GE_OQ),
                      g2 = _mm256_cmp_pd(m, _mm256_set1_pd(2.0), _CMP_GE_OQ),
                      g3 = _mm256_cmp_pd(m, _mm256_set1_pd(3.0), _CMP_GE_OQ),
                      g4 = _mm256_cmp_pd(m, _mm256_set1_pd(4.0), _CMP_GE_OQ),
                      g6 = _mm256_cmp_pd(m, _mm256_set1_pd(6.0), _CMP_GE_OQ);
        __m256i c = _mm256_add_epi64(_mm256_add_epi64(_mm256_castpd_si256(g05), _mm256_castpd_si256(g1)),
                                     _mm256_add_epi64(_mm256_castpd_si256(g15), _mm256_castpd_si256(g2)));
        c = _mm256_add_epi64(c, _mm256_add_epi64(_mm256_castpd_si256(g3), _mm256_castpd_si256(g4)));
        __m256d lo = _mm256_add_pd(_mm256_add_pd(_mm256_and_pd(g05, h), _mm256_and_pd(g1, h)),
                                   _mm256_add_pd(_mm256_and_pd(g15, h), _mm256_and_pd(g2, h)));
        lo = _mm256_add_pd(lo, _mm256_add_pd(_mm256_and_pd(g3, one), _mm256_and_pd(g4, one)));
        const __m256d inv = _mm256_sub_pd(_mm256_sub_pd(_mm256_set1_pd(2.0), _mm256_and_pd(g2, one)), _mm256_and_pd(g4, h));
        const __m256d up = _mm256_cmp_pd(_mm256_loadu_pd(u + i), _mm256_mul_pd(_mm256_sub_pd(m, lo), inv), _CMP_LT_OQ);
        c = _mm256_add_epi64(c, _mm256_castpd_si256(up));
        c = _mm256_sub_epi64(_mm256_setzero_si256(), c);
        c = _mm256_blendv_epi8(c, _mm256_set1_epi64x(7), _mm256_castpd_si256(g6));
        const __m256i sign = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_castpd_si256(q), 63), 3);
        c = _mm256_and_si256(_mm256_or_si256(c, sign), _mm256_castpd_si256(_mm256_cmp_pd(q, q, _CMP_ORD_Q)));
        store4(c, out + i);
    }
    for (; i < n; ++i) out[i] = e2m1_encode_sr(x[i] / d, u[i]);
}
inline void round_q8(const double* x, int n, double inv, std::int8_t* out) {
    const __m256d iv = _mm256_set1_pd(inv), hi = _mm256_set1_pd(127.0), lo = _mm256_set1_pd(-127.0);
    const __m256d sm = _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
    const __m256d half = _mm256_set1_pd(0.5);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_mul_pd(_mm256_loadu_pd(x + i), iv);
        v = _mm256_min_pd(v, hi);  // (v < 127) ? v : 127, as std::min(127, v)
        v = _mm256_max_pd(v, lo);  // (v > -127) ? v : -127, as std::max(-127, v)
        const __m256d r = _mm256_add_pd(v, _mm256_or_pd(_mm256_and_pd(v, sm), half));
        const __m128i t = _mm256_cvttpd_epi32(r);
        const __m128i b = _mm_shuffle_epi8(t, _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
        const int v4 = _mm_cvtsi128_si32(b);
        std::memcpy(out + i, &v4, 4);
    }
    for (; i < n; ++i) out[i] = round_q8_scalar(x[i], inv);
}
#endif
#endif
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON) || defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
inline void encode_rne(const double* x, int n, double d, std::uint8_t* out) {
    if (pow2_scale(d)) encode_rne_t<true>(x, n, d, out);
    else encode_rne_t<false>(x, n, d, out);
}
inline void encode_sr(const double* x, int n, double d, const double* u, std::uint8_t* out) {
    if (pow2_scale(d)) encode_sr_t<true>(x, n, d, u, out);
    else encode_sr_t<false>(x, n, d, u, out);
}
#else  // generic
inline double absmax(const double* x, int n) { return absmax_scalar(x, n); }
inline void encode_rne(const double* x, int n, double d, std::uint8_t* out) {
    for (int i = 0; i < n; ++i) out[i] = e2m1_encode(x[i] / d);
}
inline void encode_sr(const double* x, int n, double d, const double* u, std::uint8_t* out) {
    for (int i = 0; i < n; ++i) out[i] = e2m1_encode_sr(x[i] / d, u[i]);
}
inline void round_q8(const double* x, int n, double inv, std::int8_t* out) {
    for (int i = 0; i < n; ++i) out[i] = round_q8_scalar(x[i], inv);
}
#endif

//! absmax() of exactly 16 values, unrolled (the same maxima: exact, NaN
//! never wins).
inline double absmax16(const double* x) {
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
    float64x2_t a[8];
    for (int i = 0; i < 8; ++i) a[i] = vabsq_f64(vld1q_f64(x + 2 * i));
    for (int i = 0; i < 4; ++i) a[i] = vmaxnmq_f64(a[i], a[i + 4]);
    a[0] = vmaxnmq_f64(vmaxnmq_f64(a[0], a[2]), vmaxnmq_f64(a[1], a[3]));
    return std::max(0.0, vmaxnmvq_f64(a[0]));
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
    const __m256d sm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), z = _mm256_setzero_pd();
    // max_pd(|x|, acc) returns acc when |x| is NaN; acc starts at 0
    __m256d a0 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x), sm), z);
    __m256d a1 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x + 4), sm), z);
    a0 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x + 8), sm), a0);
    a1 = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(x + 12), sm), a1);
    a0 = _mm256_max_pd(a0, a1);
    const __m128d h = _mm_max_pd(_mm256_castpd256_pd128(a0), _mm256_extractf128_pd(a0, 1));
    return std::max(_mm_cvtsd_f64(h), _mm_cvtsd_f64(_mm_unpackhi_pd(h, h)));
#else
    return absmax_scalar(x, 16);
#endif
}

//! `n` uniforms of a SplitMix64 stream, in order.
inline void draws(SplitMix64& rng, int n, double* u) {
    for (int i = 0; i < n; ++i) u[i] = rng.uniform();
}

//! The 2 x E2M1 value of each of `n` codes.
inline void codes_to_values(const std::uint8_t* c, int n, std::int8_t* v) {
    int i = 0;
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
    const int8x16_t tab = vld1q_s8(kValues2);
    for (; i + 16 <= n; i += 16) vst1q_s8(v + i, vqtbl1q_s8(tab, vld1q_u8(c + i)));
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
    const __m128i tab = _mm_load_si128(reinterpret_cast<const __m128i*>(kValues2));
    for (; i + 16 <= n; i += 16)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(v + i),
                         _mm_shuffle_epi8(tab, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i))));
#endif
    for (; i < n; ++i) v[i] = kValues2[c[i] & 0x0F];
}
}  // namespace vq

// ------------------------------------------------------------------ quantisers

//! Quantise `rows x K` doubles to int8, one scale a block of `block` (16 or
//! 32) elements: `d = float(amax / 127)`, `q = round(x / d)`, halves away
//! from zero, in natural order, rows zero padded to padded_cols(K).
inline void quantize_q8(const double* A, int rows, int K, int block, Q8Rows& out) {
    out.rows = rows;
    out.kp = padded_cols(K);
    out.q.assign(static_cast<std::size_t>(rows) * out.kp, 0);
    out.sc.assign(static_cast<std::size_t>(rows) * out.n_sub(), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * K;
        std::int8_t* q = out.q.data() + static_cast<std::size_t>(r) * out.kp;
        float* sc = out.sc.data() + static_cast<std::size_t>(r) * out.n_sub();
        for (int k0 = 0; k0 < out.kp; k0 += block) {
            const int n = std::max(0, std::min(block, K - k0));  // the rest of the block is padding (zero)
            const double amax = n > 0 ? vq::absmax(x + k0, n) : 0.0;
            const float d = static_cast<float>(amax / 127.0);
            for (int s = k0 / 16; s < (k0 + block) / 16; ++s) sc[s] = d;
            const double inv = d > 0.0f ? 1.0 / static_cast<double>(d) : 0.0;
            if (n > 0) vq::round_q8(x + k0, n, inv, q + k0);
        }
    }
}

//! nvfp4's block scales of `nb` blocks from their absmaxes (`dv` in):
//! block_scale_code(NVFP4, amax, g) (`codes`, if given), the half scales
//! `hsc` = float(0.5 e4m3 g) and decoded scales `dv` = e4m3 g (out). The
//! normal E4M3 range and zero blocks go 2 / 4 lanes at a time --
//! e4m3_encode's bit-level rounding, the value rebuilt from the code bits
//! (exact products) -- the rest through the scalar code and tables.
inline void nvfp4_block_scales(int nb, float g, std::uint8_t* codes, float* hsc, double* dv) {
    const double den = kE2M1Max * static_cast<double>(g), gd = static_cast<double>(g);
    const kd::HalfScaleTables& hs = kd::half_scales();
    const detail::ScaleTables& st = detail::scale_tables();
    auto scalar = [&](int b) {
        const std::uint8_t code = block_scale_code(Format::NVFP4, dv[b], g);
        if (codes) codes[b] = code;
        hsc[b] = static_cast<float>(static_cast<double>(hs.e4m3[code]) * gd);
        dv[b] = st.e4m3[code] * gd;
    };
    int b = 0;
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
    const float64x2_t dn = vdupq_n_f64(den), gv = vdupq_n_f64(gd), hg = vdupq_n_f64(0.5 * gd);
    const uint64x2_t m52 = vdupq_n_u64((1ULL << 52) - 1), m49 = vdupq_n_u64((1ULL << 49) - 1), half = vdupq_n_u64(1ULL << 48);
    for (; b + 2 <= nb; b += 2) {
        const float64x2_t a = vld1q_f64(dv + b);
        const uint64x2_t u = vreinterpretq_u64_f64(vdivq_f64(a, dn));
        const uint64x2_t be = vshrq_n_u64(u, 52);
        const uint64x2_t zero = vceqq_f64(a, vdupq_n_f64(0.0));  // -> e4m3_encode(1.0) = 0x38
        const uint64x2_t ok = vorrq_u64(vandq_u64(vcgeq_u64(be, vdupq_n_u64(1017)), vcleq_u64(be, vdupq_n_u64(1031))), zero);
        if ((vgetq_lane_u64(ok, 0) & vgetq_lane_u64(ok, 1)) == 0) {
            scalar(b);
            scalar(b + 1);
            continue;
        }
        const uint64x2_t mant = vandq_u64(u, m52), rem = vandq_u64(mant, m49);
        const uint64x2_t q = vorrq_u64(vdupq_n_u64(8), vshrq_n_u64(mant, 49));
        const uint64x2_t up = vorrq_u64(vcgtq_u64(rem, half), vandq_u64(vceqq_u64(rem, half), vtstq_u64(q, vdupq_n_u64(1))));
        uint64x2_t c = vsubq_u64(vaddq_u64(vshlq_n_u64(vsubq_u64(be, vdupq_n_u64(1016)), 3), q), vdupq_n_u64(8));
        c = vsubq_u64(c, up);  // up is 0 / all ones
        c = vbslq_u64(vcgtq_u64(c, vdupq_n_u64(0x7E)), vdupq_n_u64(0x7E), c);
        c = vbslq_u64(zero, vdupq_n_u64(0x38), c);
        // the E4M3 value 2^(e - 7) (1 + m / 8), e >= 1 here
        const float64x2_t val = vreinterpretq_f64_u64(
                vorrq_u64(vshlq_n_u64(vaddq_u64(vshrq_n_u64(c, 3), vdupq_n_u64(1016)), 52),
                          vshlq_n_u64(vandq_u64(c, vdupq_n_u64(7)), 49)));
        vst1q_f64(dv + b, vmulq_f64(val, gv));
        vst1_f32(hsc + b, vcvt_f32_f64(vmulq_f64(val, hg)));
        if (codes) {
            codes[b] = static_cast<std::uint8_t>(vgetq_lane_u64(c, 0));
            codes[b + 1] = static_cast<std::uint8_t>(vgetq_lane_u64(c, 1));
        }
    }
#endif
    for (; b < nb; ++b) scalar(b);
}

//! Phase 1 of encode_row_fast() for mxfp4 / nvfp4: every block's absmax
//! (and from them the row's, nvfp4's per-row global scale when
//! `row_scale`), then every block's scale code, half scales `hsc` (one a
//! 16-element sub-block) and decoded scale `dv` (one a block) -- independent
//! iterations the core overlaps.
inline void row_block_scales(Format f, const double* x, int cols, float g, bool row_scale, float* hsc, double* dv,
                             std::uint8_t* scale_codes = nullptr) {
    const int kp = padded_cols(cols), block = f == Format::MXFP4 ? 32 : 16, nb = kp / block;
    double row_amax = 0.0;
    for (int b = 0; b < nb; ++b) {  // dv holds the absmaxes first
        const int k0 = std::min(cols, b * block), k1 = std::min(cols, k0 + block);
        dv[b] = k1 - k0 == 16 ? vq::absmax16(x + k0) : (k1 > k0 ? vq::absmax(x + k0, k1 - k0) : 0.0);
        row_amax = std::max(row_amax, dv[b]);
    }
    if (row_scale && f == Format::NVFP4) g = nvfp4_tensor_scale(row_amax);
    const int per = block / 16;
    const kd::HalfScaleTables& hs = kd::half_scales();
    const detail::ScaleTables& st = detail::scale_tables();
    const double gd = static_cast<double>(g);
    if (f == Format::NVFP4) {
        nvfp4_block_scales(nb, g, scale_codes, hsc, dv);
        return;
    }
    for (int b = 0; b < nb; ++b) {
        const std::uint8_t code = block_scale_code(f, dv[b], g);
        if (scale_codes) scale_codes[b] = code;
        // kd::half_scale() and block_scale_value(), the tables hoisted
        const float h = f == Format::MXFP4 ? hs.e8m0[code] : static_cast<float>(static_cast<double>(hs.e4m3[code]) * gd);
        for (int j = 0; j < per; ++j) hsc[b * per + j] = h;
        dv[b] = f == Format::MXFP4 ? st.e8m0[code] : st.e4m3[code] * gd;
    }
}

//! One row of `cols` doubles to unpacked FP4 codes (`codes`, padded with
//! zero codes to padded_cols(cols)) and float half scales per 16-element
//! sub-block (`hsc`): the decisions of encode_row() (MlpFp4.h) -- same
//! scale codes, same element codes, and for stochastic rounding the same
//! draws in the same order (one a real element of a block with a non-zero
//! scale). `g` is nvfp4's global scale -- or, with `row_scale`, computed from the
//! row's absmax (one a row); `ubuf` scratch of >= cols doubles.
inline void encode_row_fast(Format f, const double* x, int cols, float g, std::uint8_t* codes, float* hsc,
                            Rounding rnd, SplitMix64* rng, double* ubuf, std::uint8_t* scale_codes = nullptr,
                            bool row_scale = false) {
    const int kp = padded_cols(cols), ns = kp / 16;
    const bool sr = rnd == Rounding::Stochastic && rng != nullptr;
    std::fill(codes + cols, codes + kp, std::uint8_t(0));
    if (f == Format::FP4) {
        float s = static_cast<float>(vq::absmax(x, cols) / kE2M1Max);
        if (!(s > 0.0f) || !std::isfinite(s)) s = 1.0f;
        if (sr) {
            vq::draws(*rng, cols, ubuf);
            vq::encode_sr(x, cols, static_cast<double>(s), ubuf, codes);
        } else {
            vq::encode_rne(x, cols, static_cast<double>(s), codes);
        }
        const float h = static_cast<float>(0.5 * static_cast<double>(s));
        for (int i = 0; i < ns; ++i) hsc[i] = h;
        if (scale_codes) detail::put_f32(s, scale_codes);
        return;
    }
    const int block = f == Format::MXFP4 ? 32 : 16, nb = kp / block;
    // phase 1: the block scales (row_block_scales)
    constexpr int kStack = 64;
    double dstack[kStack];
    std::vector<double> dheap;
    double* dv = dstack;
    if (nb > kStack) {
        dheap.resize(static_cast<std::size_t>(nb));
        dv = dheap.data();
    }
    row_block_scales(f, x, cols, g, row_scale, hsc, dv, scale_codes);
    // phase 2, the elements
    for (int b = 0; b < nb; ++b) {
        const int k0 = std::min(cols, b * block), k1 = std::min(cols, k0 + block);
        if (k1 == k0) break;
        const double d = dv[b];
        if (!(d > 0.0)) {
            std::fill(codes + k0, codes + k1, std::uint8_t(0));
        } else if (sr) {
            vq::draws(*rng, k1 - k0, ubuf);
            vq::encode_sr(x + k0, k1 - k0, d, ubuf, codes + k0);
        } else {
            vq::encode_rne(x + k0, k1 - k0, d, codes + k0);
        }
    }
}

//! Scratch for the quantisers (reused across calls).
struct QuantScratch {
    std::vector<std::uint8_t> codes;
    std::vector<double> u;
    std::vector<std::uint8_t> scodes;
    std::vector<float> hsc;
};

//! Quantise `rows x K` doubles to FP4 per row (nvfp4 with one global scale
//! a row) and store the codes' values as a left operand (W4A4, training).
inline void quantize_rows_fast(const double* A, int rows, int K, std::size_t ld, Format f, bool exact, const SrKey* key,
                               Q8Rows* left, PackedRight* right, QuantScratch& s);
inline void quantize_left(const double* A, int rows, int K, Format f, Rounding rnd, SplitMix64* rng, Q8Rows& out,
                          QuantScratch& s) {
    if (f != Format::FP4 && (rnd == Rounding::NearestEven || rng == nullptr)) {
        quantize_rows_fast(A, rows, K, static_cast<std::size_t>(K), f, true, nullptr, &out, nullptr, s);
        return;
    }
    out.rows = rows;
    out.kp = padded_cols(K);
    out.q.resize(static_cast<std::size_t>(rows) * out.kp);
    out.sc.resize(static_cast<std::size_t>(rows) * out.n_sub());
    s.codes.resize(static_cast<std::size_t>(out.kp));
    s.u.resize(static_cast<std::size_t>(std::max(1, K)));
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * K;
        encode_row_fast(f, x, K, 1.0f, s.codes.data(), out.sc.data() + static_cast<std::size_t>(r) * out.n_sub(), rnd, rng,
                        s.u.data(), nullptr, true);
        vq::codes_to_values(s.codes.data(), out.kp, out.q.data() + static_cast<std::size_t>(r) * out.kp);
    }
}

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
            for (int i = 0; i < ns; ++i) o[i] = kd::half_scale(t.format, s[i * 16 / t.block], t.tensor_scale);
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
inline Fp4Rows quantize_rows(const double* A, int rows, int K, Format f, Rounding rnd = Rounding::NearestEven,
                             SplitMix64* rng = nullptr) {
    Fp4Rows out;
    out.rows = rows;
    out.kp = padded_cols(K);
    out.stride = static_cast<std::size_t>(out.kp / 2);
    out.own.assign(static_cast<std::size_t>(rows) * out.stride, 0);
    out.sc.assign(static_cast<std::size_t>(rows) * out.n_sub(), 0.0f);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(out.kp));
    std::vector<double> u(static_cast<std::size_t>(std::max(1, K)));
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * K;
        encode_row_fast(f, x, K, 1.0f, codes.data(), out.sc.data() + static_cast<std::size_t>(r) * out.n_sub(), rnd, rng,
                        u.data(), nullptr, true);
        pack_nibbles(codes.data(), codes.size(), out.own.data() + static_cast<std::size_t>(r) * out.stride);
    }
    out.codes = out.own.data();
    return out;
}

//! The left operand of an FP4 operand (its codes' values).
inline void left_of(const Fp4Rows& X, Q8Rows& out) {
    out.rows = X.rows;
    out.kp = X.kp;
    out.q.resize(static_cast<std::size_t>(X.rows) * X.kp);
    out.sc = X.sc;
    for (int r = 0; r < X.rows; ++r)
        for (int k = 0; k < X.kp; ++k)
            out.q[static_cast<std::size_t>(r) * X.kp + k] = kValues2[nibble(X.codes + r * X.stride, static_cast<std::size_t>(k))];
}

// ------------------------------------------------------------------ packing the right operand

//! Size `P` for `rows x kp` (codes zeroed).
inline void pack_right_init(int rows, int kp, PackedRight& P) {
    P.rows = rows;
    P.kp = kp;
    P.ns = kp / 16;
    P.nt = (rows + kNR - 1) / kNR;
    const std::size_t nts = static_cast<std::size_t>(P.nt) * P.ns;
    P.codes.assign(nts * 8 * kNR, 0);
    P.sc.assign(nts * kNR, 0.0f);
    P.corr.assign(nts * kNR, 0);
}

//! Put row `o`'s unpacked codes (`kp` of them) and sub-block half scales.
inline void pack_right_row(PackedRight& P, int o, const std::uint8_t* codes, const float* hsc) {
    const int t = o / kNR, ol = o % kNR;
    for (int s = 0; s < P.ns; ++s) {
        const std::size_t ts = static_cast<std::size_t>(t) * P.ns + s;
        std::uint8_t* dst = P.codes.data() + ts * 8 * kNR;
        const std::uint8_t* c = codes + 16 * s;
        // byte j of quad p: code 8p + j low, 8p + 4 + j high (four at a time)
        std::uint32_t q[4];
        std::memcpy(q, c, 16);
        const std::uint32_t m = 0x0F0F0F0Fu;
        const std::uint32_t b0 = (q[0] & m) | ((q[1] & m) << 4), b1 = (q[2] & m) | ((q[3] & m) << 4);
        std::memcpy(dst + 4 * ol, &b0, 4);
        std::memcpy(dst + 4 * kNR + 4 * ol, &b1, 4);
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
        const std::int32_t sum = vaddlvq_s8(vqtbl1q_s8(vld1q_s8(kValues2), vandq_u8(vld1q_u8(c), vdupq_n_u8(0x0F))));
#else
        std::int32_t sum = 0;
        for (int j = 0; j < 16; ++j) sum += kValues2[c[j] & 0x0F];
#endif
        P.sc[ts * kNR + ol] = hsc[s];
        P.corr[ts * kNR + ol] = 128 * sum;
    }
}

//! Pack an FP4 operand (standard rows) for the micro-kernel.
inline void pack_right(const Fp4Rows& W, PackedRight& P) {
    pack_right_init(W.rows, W.kp, P);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(W.kp));
    for (int o = 0; o < W.rows; ++o) {
        const std::uint8_t* row = W.codes + static_cast<std::size_t>(o) * W.stride;
        for (int k = 0; k < W.kp; ++k) codes[static_cast<std::size_t>(k)] = nibble(row, static_cast<std::size_t>(k));
        pack_right_row(P, o, codes.data(), W.sc.data() + static_cast<std::size_t>(o) * (W.kp / 16));
    }
}

//! Quantise `rows x K` doubles to FP4 per row (as quantize_left) straight
//! into a packed right operand (wgrad's activations).
inline void quantize_right(const double* A, int rows, int K, Format f, Rounding rnd, SplitMix64* rng, PackedRight& P,
                           QuantScratch& s) {
    pack_right_init(rows, padded_cols(K), P);
    s.codes.resize(static_cast<std::size_t>(P.kp));
    s.u.resize(static_cast<std::size_t>(std::max(1, K)));
    std::vector<float> hsc(static_cast<std::size_t>(P.ns));
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * K;
        encode_row_fast(f, x, K, 1.0f, s.codes.data(), hsc.data(), rnd, rng, s.u.data(), nullptr, true);
        pack_right_row(P, r, s.codes.data(), hsc.data());
    }
}

// ------------------------------------------------------------------ training's operand quantiser
//
// The operands only training sees (dgrad's and wgrad's gradients, wgrad's
// transformed activations) take a cheaper element rule than inference's
// exact `e2m1(x / d)`: q = float(x * (1 / d)) -- one division a block, the
// element narrowed to float -- then e2m1_encode_f(q) (round to nearest
// even) or e2m1_encode_sr16(q, u) with the counter-based draw of SrKey
// (MlpFp4.h). Block scales are exactly quantize_left()'s. fprop's
// activations and the weights keep the exact rule: inference must
// reproduce training's forward pass. Every variant below computes these
// scalar definitions lane by lane (the float ops are single IEEE
// operations; nothing to contract).

namespace tq {
//! The 16 codes of one group (16 elements, `x` readable for all 16) at
//! reciprocal scale `r`; `key` null rounds to nearest even, else the
//! group's draws are words `n0 .. n0 + 7`.
inline void group_generic(const double* x, double r, const SrKey* key, std::uint32_t n0, std::uint8_t* c) {
    if (key == nullptr) {
        for (int j = 0; j < 16; ++j) c[j] = e2m1_encode_f(static_cast<float>(x[j] * r));
        return;
    }
    std::uint32_t w[8];
    for (int j = 0; j < 8; ++j) w[j] = key->word(n0 + static_cast<std::uint32_t>(j));
    for (int j = 0; j < 16; ++j)
        c[j] = e2m1_encode_sr16(static_cast<float>(x[j] * r), (j & 8) ? (w[j & 7] >> 16) : (w[j & 7] & 0xFFFFu));
}
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
inline uint32x4_t mix32v(uint32x4_t x) {
    x = veorq_u32(x, vshrq_n_u32(x, 16));
    x = vmulq_u32(x, vdupq_n_u32(0x21F0AAADu));
    x = veorq_u32(x, vshrq_n_u32(x, 15));
    x = vmulq_u32(x, vdupq_n_u32(0x735A2D97u));
    return veorq_u32(x, vshrq_n_u32(x, 15));
}
//! The float bits `y + 16` of e2m1_encode_sr16 for four lanes.
inline uint32x4_t sr_bits(float32x4_t q) {
    const float32x4_t y = vaddq_f32(vmaxnmq_f32(vabsq_f32(q), vdupq_n_f32(0.0f)), vdupq_n_f32(2.0f));
    const float32x4_t y2 = vfmaq_n_f32(vdupq_n_f32(3.0f), y, 0.5f);  // exact product: fused or not alike
    return vaddq_u32(vreinterpretq_u32_f32(vminq_f32(y, y2)), vdupq_n_u32(16));
}
//! e2m1_encode_sr16 of eight lanes (qa, qb) with draws `u`, in 16-bit lanes:
//! the carry of `(bits & 0x1FFFFF) + (u << 5)` into bit 21 is the carry of
//! `mid + u`, mid = bits 5..20.
inline uint8x8_t sr8(float32x4_t qa, float32x4_t qb, uint16x8_t u) {
    const uint32x4_t ya = sr_bits(qa), yb = sr_bits(qb);
    const uint16x8_t top = vshrn_high_n_u32(vshrn_n_u32(ya, 16), yb, 16);
    const uint16x8_t mid = vshrn_high_n_u32(vshrn_n_u32(ya, 5), yb, 5);
    const uint16x8_t carry = vcltq_u16(vaddq_u16(mid, u), mid);
    uint16x8_t t = vsubq_u16(vshrq_n_u16(top, 5), carry);
    t = vminq_u16(vsubq_u16(t, vdupq_n_u16(512)), vdupq_n_u16(7));
    const uint16x8_t qs = vshrn_high_n_u32(vshrn_n_u32(vreinterpretq_u32_f32(qa), 16), vreinterpretq_u32_f32(qb), 16);
    return vmovn_u16(vsliq_n_u16(t, vshrq_n_u16(qs, 15), 3));
}
//! e2m1_encode_f of eight lanes in 16-bit lanes: for positive floats and
//! thresholds t with zero low 16 bits, m >= t iff (bits(m) >> 16) >= (bits(t)
//! >> 16) and m > t iff ((bits(m) - 1) >> 16) >= (bits(t) >> 16) (m = 0
//! saturates at 0).
inline uint8x8_t rne8(float32x4_t qa, float32x4_t qb) {
    const uint32x4_t ma = vreinterpretq_u32_f32(vmaxnmq_f32(vabsq_f32(qa), vdupq_n_f32(0.0f)));
    const uint32x4_t mb = vreinterpretq_u32_f32(vmaxnmq_f32(vabsq_f32(qb), vdupq_n_f32(0.0f)));
    const uint32x4_t one = vdupq_n_u32(1);
    const uint16x8_t ge = vshrn_high_n_u32(vshrn_n_u32(ma, 16), mb, 16);
    const uint16x8_t gt = vshrn_high_n_u32(vshrn_n_u32(vqsubq_u32(ma, one), 16), vqsubq_u32(mb, one), 16);
    // 0.25 3E80, 0.75 3F40, 1.25 3FA0, 1.75 3FE0, 2.5 4020, 3.5 4060, 5 40A0
    uint16x8_t t = vcgeq_u16(gt, vdupq_n_u16(0x3E80));
    t = vaddq_u16(t, vcgeq_u16(ge, vdupq_n_u16(0x3F40)));
    t = vaddq_u16(t, vcgeq_u16(gt, vdupq_n_u16(0x3FA0)));
    t = vaddq_u16(t, vcgeq_u16(ge, vdupq_n_u16(0x3FE0)));
    t = vaddq_u16(t, vcgeq_u16(gt, vdupq_n_u16(0x4020)));
    t = vaddq_u16(t, vcgeq_u16(ge, vdupq_n_u16(0x4060)));
    t = vaddq_u16(t, vcgeq_u16(gt, vdupq_n_u16(0x40A0)));
    t = vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(t)));
    const uint16x8_t qs = vshrn_high_n_u32(vshrn_n_u32(vreinterpretq_u32_f32(qa), 16), vreinterpretq_u32_f32(qb), 16);
    return vmovn_u16(vsliq_n_u16(t, vshrq_n_u16(qs, 15), 3));
}
inline float32x4_t q4(const double* x, float64x2_t r) {
    return vcvt_high_f32_f64(vcvt_f32_f64(vmulq_f64(vld1q_f64(x), r)), vmulq_f64(vld1q_f64(x + 2), r));
}
inline void group(const double* x, double r, const SrKey* key, std::uint32_t n0, std::uint8_t* c) {
    const float64x2_t rv = vdupq_n_f64(r);
    const float32x4_t q0 = q4(x, rv), q1 = q4(x + 4, rv), q2 = q4(x + 8, rv), q3 = q4(x + 12, rv);
    if (key == nullptr) {
        vst1q_u8(c, vcombine_u8(rne8(q0, q1), rne8(q2, q3)));
        return;
    }
    static const std::uint32_t kLane[4] = {0, 1, 2, 3};
    const uint32x4_t n = vaddq_u32(vdupq_n_u32(n0), vld1q_u32(kLane));
    const uint32x4_t a = vdupq_n_u32(key->a), b = vdupq_n_u32(key->b);
    const uint32x4_t w0 = mix32v(veorq_u32(mix32v(veorq_u32(n, a)), b));
    const uint32x4_t w1 = mix32v(veorq_u32(mix32v(veorq_u32(vaddq_u32(n, vdupq_n_u32(4)), a)), b));
    // elements 0-7: low halves of words 0-7; 8-15: high halves
    const uint16x8_t ulo = vuzp1q_u16(vreinterpretq_u16_u32(w0), vreinterpretq_u16_u32(w1));
    const uint16x8_t uhi = vuzp2q_u16(vreinterpretq_u16_u32(w0), vreinterpretq_u16_u32(w1));
    vst1q_u8(c, vcombine_u8(sr8(q0, q1, ulo), sr8(q2, q3, uhi)));
}
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
//! The group's eight words (lanes j = word n0 + j).
inline __m256i words8(const SrKey& key, std::uint32_t n0) {
    auto mix = [](__m256i x) {
        x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 16));
        x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x21F0AAAD));
        x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
        x = _mm256_mullo_epi32(x, _mm256_set1_epi32(0x735A2D97));
        return _mm256_xor_si256(x, _mm256_srli_epi32(x, 15));
    };
    const __m256i n = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(n0)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const __m256i a = _mm256_set1_epi32(static_cast<int>(key.a)), b = _mm256_set1_epi32(static_cast<int>(key.b));
    return mix(_mm256_xor_si256(mix(_mm256_xor_si256(n, a)), b));
}
#if defined(IMPBFF_FP4_AVX512)
inline void group(const double* x, double r, const SrKey* key, std::uint32_t n0, std::uint8_t* c) {
    const __m512d rv = _mm512_set1_pd(r);
    const __m256 lo = _mm512_cvtpd_ps(_mm512_mul_pd(_mm512_loadu_pd(x), rv));
    const __m256 hi = _mm512_cvtpd_ps(_mm512_mul_pd(_mm512_loadu_pd(x + 8), rv));
    const __m512 q = _mm512_castpd_ps(
            _mm512_insertf64x4(_mm512_castps_pd(_mm512_castps256_ps512(lo)), _mm256_castps_pd(hi), 1));
    const __m512i qb = _mm512_castps_si512(q);
    const __m512 m = _mm512_max_ps(_mm512_abs_ps(q), _mm512_setzero_ps());  // NaN -> 0
    const __m512i sign = _mm512_and_si512(_mm512_srli_epi32(qb, 28), _mm512_set1_epi32(8));
    __m512i t;
    if (key == nullptr) {
        const __m512i one = _mm512_set1_epi32(1);
        t = _mm512_setzero_si512();
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(0.25f), _CMP_GT_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(0.75f), _CMP_GE_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(1.25f), _CMP_GT_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(1.75f), _CMP_GE_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(2.5f), _CMP_GT_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(3.5f), _CMP_GE_OQ), t, one);
        t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(m, _mm512_set1_ps(5.0f), _CMP_GT_OQ), t, one);
    } else {
        const __m256i w = words8(*key, n0);
        const __m512i u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(w, _mm256_set1_epi32(0xFFFF))),
                                             _mm256_srli_epi32(w, 16), 1);  // elements 0-7 low halves, 8-15 high
        const __m512i r1 = _mm512_or_si512(_mm512_slli_epi32(u, 5), _mm512_set1_epi32(16));
        const __m512 y = _mm512_add_ps(m, _mm512_set1_ps(2.0f));
        const __m512 y2 = _mm512_add_ps(_mm512_mul_ps(y, _mm512_set1_ps(0.5f)), _mm512_set1_ps(3.0f));
        const __m512i yb = _mm512_castps_si512(_mm512_min_ps(y, y2));
        t = _mm512_srli_epi32(_mm512_add_epi32(yb, r1), 21);
        t = _mm512_min_epu32(_mm512_sub_epi32(t, _mm512_set1_epi32(512)), _mm512_set1_epi32(7));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c), _mm512_cvtepi32_epi8(_mm512_or_si512(t, sign)));
        return;
    }
    t = _mm512_or_si512(t, sign);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(c), _mm512_cvtepi32_epi8(t));
}
#else  // AVX2
inline __m256 q8(const double* x, __m256d r) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_mul_pd(_mm256_loadu_pd(x), r))),
                                _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_loadu_pd(x + 4), r)), 1);
}
//! Codes of eight lanes: RNE (`r1` unused) or SR with `r1` = 16 + (u << 5).
inline __m256i codes8(__m256 q, bool sr, __m256i r1) {
    const __m256i qb = _mm256_castps_si256(q);
    // |q|, NaN -> 0 (max_ps returns its second operand for a NaN)
    const __m256 m = _mm256_max_ps(_mm256_and_ps(q, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))), _mm256_setzero_ps());
    __m256i t;
    if (!sr) {
        t = _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(0.25f), _CMP_GT_OQ));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(0.75f), _CMP_GE_OQ)));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(1.25f), _CMP_GT_OQ)));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(1.75f), _CMP_GE_OQ)));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(2.5f), _CMP_GT_OQ)));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(3.5f), _CMP_GE_OQ)));
        t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(m, _mm256_set1_ps(5.0f), _CMP_GT_OQ)));
        t = _mm256_sub_epi32(_mm256_setzero_si256(), t);
    } else {
        const __m256 y = _mm256_add_ps(m, _mm256_set1_ps(2.0f));
        const __m256 y2 = _mm256_add_ps(_mm256_mul_ps(y, _mm256_set1_ps(0.5f)), _mm256_set1_ps(3.0f));
        const __m256i yb = _mm256_castps_si256(_mm256_min_ps(y, y2));
        t = _mm256_srli_epi32(_mm256_add_epi32(yb, r1), 21);
        t = _mm256_min_epu32(_mm256_sub_epi32(t, _mm256_set1_epi32(512)), _mm256_set1_epi32(7));
        return _mm256_or_si256(t, _mm256_and_si256(_mm256_srli_epi32(qb, 28), _mm256_set1_epi32(8)));
    }
    return _mm256_or_si256(t, _mm256_and_si256(_mm256_srli_epi32(qb, 28), _mm256_set1_epi32(8)));
}
inline void group(const double* x, double r, const SrKey* key, std::uint32_t n0, std::uint8_t* c) {
    const __m256d rv = _mm256_set1_pd(r);
    __m256i r0 = _mm256_setzero_si256(), r8 = r0;
    if (key != nullptr) {
        const __m256i w = words8(*key, n0), lo = _mm256_set1_epi32(0x1FFFE0), h = _mm256_set1_epi32(16);
        r0 = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(w, 5), lo), h);   // elements 0-7: low halves
        r8 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(w, 11), lo), h);  // 8-15: high halves
    }
    const __m256i c0 = codes8(q8(x, rv), key != nullptr, r0), c1 = codes8(q8(x + 8, rv), key != nullptr, r8);
    const __m256i p = _mm256_packus_epi32(c0, c1);  // 16-bit: c0 0-3, c1 0-3 | c0 4-7, c1 4-7
    const __m128i b = _mm_packus_epi16(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    // bytes: c0 0-3, c1 0-3, c0 4-7, c1 4-7 -> c0 0-7, c1 0-7
    _mm_storeu_si128(reinterpret_cast<__m128i*>(c), _mm_shuffle_epi8(b, _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15)));
}
#endif
#else
inline void group(const double* x, double r, const SrKey* key, std::uint32_t n0, std::uint8_t* c) {
    group_generic(x, r, key, n0, c);
}
#endif
//! Inference's exact rule on one group: e2m1_encode(x / d) (a NaN quotient
//! may come out -0 instead of +0: the same value).
inline void group_exact(const double* x, double d, bool p2, std::uint8_t* c) {
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
    const float64x2_t dv = vdupq_n_f64(d), iv = vdupq_n_f64(1.0 / d);
    float32x4_t q[4];
    if (p2)
        for (int i = 0; i < 4; ++i)
            q[i] = vcvtx_high_f32_f64(vcvtx_f32_f64(vq::quot<true>(vld1q_f64(x + 4 * i), dv, iv)),
                                      vq::quot<true>(vld1q_f64(x + 4 * i + 2), dv, iv));
    else
        for (int i = 0; i < 4; ++i)
            q[i] = vcvtx_high_f32_f64(vcvtx_f32_f64(vq::quot<false>(vld1q_f64(x + 4 * i), dv, iv)),
                                      vq::quot<false>(vld1q_f64(x + 4 * i + 2), dv, iv));
    // round-to-odd floats compare with the thresholds as the doubles do
    vst1q_u8(c, vcombine_u8(rne8(q[0], q[1]), rne8(q[2], q[3])));
#else
    (void)p2;
    vq::encode_rne(x, 16, d, c);
#endif
}
}  // namespace tq

//! Quantise `rows x K` doubles (row stride `ld`) to FP4 per row with
//! training's element rule (above): nvfp4 one global scale a row, codes
//! into a left operand (their values) or a packed right operand. With
//! `key`, element (r, k) is stochastically rounded with key->u16(r * kp + k).
inline void quantize_rows_fast(const double* A, int rows, int K, std::size_t ld, Format f, bool exact, const SrKey* key,
                               Q8Rows* left, PackedRight* right, QuantScratch& s) {
    const int kp = padded_cols(K), ns = kp / 16, k16 = (K + 15) / 16 * 16;
    const int block = f == Format::FP4 ? kp : (f == Format::MXFP4 ? 32 : 16), nb = kp / block;
    if (left) {
        left->rows = rows;
        left->kp = kp;
        left->q.resize(static_cast<std::size_t>(rows) * kp);
        left->sc.resize(static_cast<std::size_t>(rows) * ns);
    }
    if (right) pack_right_init(rows, kp, *right);
    s.codes.resize(static_cast<std::size_t>(kp));
    s.u.resize(static_cast<std::size_t>(nb + (K % 16 ? k16 : 0)));
    s.hsc.resize(static_cast<std::size_t>(ns));
    double* dv = s.u.data();
    double* pad = dv + nb;
    std::uint8_t* codes = s.codes.data();
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * ld;
        float* hsc = left ? left->sc.data() + static_cast<std::size_t>(r) * ns : s.hsc.data();
        if (f == Format::FP4) {
            float sc = static_cast<float>(vq::absmax(x, K) / kE2M1Max);
            if (!(sc > 0.0f) || !std::isfinite(sc)) sc = 1.0f;
            dv[0] = static_cast<double>(sc);
            for (int i = 0; i < ns; ++i) hsc[i] = static_cast<float>(0.5 * static_cast<double>(sc));
        } else {
            row_block_scales(f, x, K, 1.0f, true, hsc, dv);
        }
        if (K % 16) {  // a zero-padded copy: the last group reads 16 elements
            std::copy(x, x + K, pad);
            std::fill(pad + K, pad + k16, 0.0);
            x = pad;
        }
        std::fill(codes + k16, codes + kp, std::uint8_t(0));
        const std::uint32_t nrow = static_cast<std::uint32_t>(static_cast<std::size_t>(r) * ns * 8);
        for (int b = 0; b < nb; ++b) {
            const int k0 = b * block, k1 = std::min(k16, k0 + block);
            if (k1 <= k0) break;
            const double d = dv[b];
            if (!(d > 0.0)) {
                std::fill(codes + k0, codes + k1, std::uint8_t(0));
                continue;
            }
            if (exact) {
                const bool p2 = vq::pow2_scale(d);
                for (int k = k0; k < k1; k += 16) tq::group_exact(x + k, d, p2, codes + k);
                continue;
            }
            const double rcp = 1.0 / d;
            for (int k = k0; k < k1; k += 16)
                tq::group(x + k, rcp, key, nrow + static_cast<std::uint32_t>(k / 2), codes + k);
        }
        if (left) vq::codes_to_values(codes, kp, left->q.data() + static_cast<std::size_t>(r) * kp);
        if (right) pack_right_row(*right, r, codes, hsc);
    }
}

//! quantize_rows_fast() with training's element rule.
inline void quantize_train(const double* A, int rows, int K, std::size_t ld, Format f, const SrKey* key, Q8Rows* left,
                           PackedRight* right, QuantScratch& s) {
    quantize_rows_fast(A, rows, K, ld, f, false, key, left, right, s);
}

// ------------------------------------------------------------------ the reference kernel

//! The shared float combination: `sum_s is[s] * (sa[s] * sb[s])` over
//! `n_sub` sub-blocks in four lanes (sub-block s goes to lane s % 4), then
//! `(l0 + l1) + (l2 + l3)`. Every micro-kernel does this per output.
inline float combine(const std::int32_t* is, const float* sa, const float* sb, int n_sub) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < n_sub; ++s) {
        const float sc = sa[s] * sb[s];
        float t = static_cast<float>(is[s]) * sc;
        IMPBFF_FP4_KEEP(t);
        acc[s & 3] += t;
    }
    const float l01 = acc[0] + acc[1];
    const float l23 = acc[2] + acc[3];
    return l01 + l23;
}

//! Reference: C (A.rows x W.rows) = A W^T element by element (tests).
inline void gemm_reference(const Q8Rows& A, const Fp4Rows& W, double* C) {
    const int ns = A.n_sub();
    std::vector<std::int32_t> is(static_cast<std::size_t>(ns));
    for (int r = 0; r < A.rows; ++r)
        for (int o = 0; o < W.rows; ++o) {
            const std::int8_t* a = A.q.data() + static_cast<std::size_t>(r) * A.kp;
            const std::uint8_t* w = W.codes + static_cast<std::size_t>(o) * W.stride;
            for (int s = 0; s < ns; ++s) {
                std::int32_t sum = 0;
                for (int k = 16 * s; k < 16 * s + 16; ++k)
                    sum += kValues2[nibble(w, static_cast<std::size_t>(k))] * a[k];
                is[static_cast<std::size_t>(s)] = sum;
            }
            C[static_cast<std::size_t>(r) * W.rows + o] =
                    combine(is.data(), A.sc.data() + static_cast<std::size_t>(r) * ns,
                            W.sc.data() + static_cast<std::size_t>(o) * ns, ns);
        }
}

// ------------------------------------------------------------------ micro-kernels
//
// tile<R>(a, lda, sa, lsa, wc, ws, wcorr, ns, out): R left rows starting at
// `a` (stride lda bytes; scales sa, stride lsa) against one packed tile
// (codes wc, scales ws, corrections wcorr); out[r * kNR + o] the float result.

namespace mk {
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
#if defined(IMPBFF_FP4_NEON_DOTPROD)
template <int L>
inline int32x4_t dot_lane(int32x4_t acc, int8x16_t w, int8x16_t a) { return vdotq_laneq_s32(acc, w, a, L); }
#else
template <int L>
inline int32x4_t dot_lane(int32x4_t acc, int8x16_t w, int8x16_t a) {
    const int8x16_t ad = vreinterpretq_s8_s32(vdupq_laneq_s32(vreinterpretq_s32_s8(a), L));
    const int32x4_t p0 = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(ad)));
    const int32x4_t p1 = vpaddlq_s16(vmull_high_s8(w, ad));
    return vaddq_s32(acc, vpaddq_s32(p0, p1));
}
#endif
template <int R>
inline void tile(const std::int8_t* a, std::size_t lda, const float* sa, std::size_t lsa, const std::uint8_t* wc,
                 const float* ws, const std::int32_t*, int ns, float* out) {
    const int8x16_t tab = vld1q_s8(kValues2);
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float32x4_t f[R][4];
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < 4; ++j) f[r][j] = vdupq_n_f32(0.0f);
    auto step = [&](int s, auto J) {
        constexpr int j = decltype(J)::value;
        const uint8x16_t c0 = vld1q_u8(wc + 32 * s), c1 = vld1q_u8(wc + 32 * s + 16);
        const int8x16_t w0 = vqtbl1q_s8(tab, vandq_u8(c0, m4)), w1 = vqtbl1q_s8(tab, vshrq_n_u8(c0, 4));
        const int8x16_t w2 = vqtbl1q_s8(tab, vandq_u8(c1, m4)), w3 = vqtbl1q_s8(tab, vshrq_n_u8(c1, 4));
        const float32x4_t sw = vld1q_f32(ws + 4 * s);
        for (int r = 0; r < R; ++r) {
            const int8x16_t av = vld1q_s8(a + r * lda + 16 * s);
            int32x4_t acc = dot_lane<0>(vdupq_n_s32(0), w0, av);
            acc = dot_lane<1>(acc, w1, av);
            acc = dot_lane<2>(acc, w2, av);
            acc = dot_lane<3>(acc, w3, av);
            const float32x4_t sc = vmulq_f32(vdupq_n_f32(sa[r * lsa + s]), sw);
            float32x4_t t = vmulq_f32(vcvtq_f32_s32(acc), sc);
            IMPBFF_FP4_KEEP(t);
            f[r][j] = vaddq_f32(f[r][j], t);
        }
    };
    int s = 0;
    for (; s + 4 <= ns; s += 4) {
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
        step(s + 2, std::integral_constant<int, 2>());
        step(s + 3, std::integral_constant<int, 3>());
    }
    if (s < ns) {  // ns is even: a trailing pair into lanes 0, 1
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
    }
    for (int r = 0; r < R; ++r) {
        const float32x4_t l01 = vaddq_f32(f[r][0], f[r][1]);
        const float32x4_t l23 = vaddq_f32(f[r][2], f[r][3]);
        vst1q_f32(out + r * kNR, vaddq_f32(l01, l23));
    }
}
#elif defined(IMPBFF_FP4_AVX512)
template <int R>
inline void tile(const std::int8_t* a, std::size_t lda, const float* sa, std::size_t lsa, const std::uint8_t* wc,
                 const float* ws, const std::int32_t* wcorr, int ns, float* out) {
    const __m512i tab = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(kValues2)));
    const __m512i m4 = _mm512_set1_epi8(0x0F), x80 = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512 f[R][4];
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < 4; ++j) f[r][j] = _mm512_setzero_ps();
    auto step = [&](int s, auto J) {
        constexpr int j = decltype(J)::value;
        const __m512i c0 = _mm512_loadu_si512(wc + 128 * s), c1 = _mm512_loadu_si512(wc + 128 * s + 64);
        const __m512i w0 = _mm512_shuffle_epi8(tab, _mm512_and_si512(c0, m4));
        const __m512i w1 = _mm512_shuffle_epi8(tab, _mm512_and_si512(_mm512_srli_epi16(c0, 4), m4));
        const __m512i w2 = _mm512_shuffle_epi8(tab, _mm512_and_si512(c1, m4));
        const __m512i w3 = _mm512_shuffle_epi8(tab, _mm512_and_si512(_mm512_srli_epi16(c1, 4), m4));
        const __m512i corr = _mm512_loadu_si512(wcorr + 16 * s);
        const __m512 sw = _mm512_loadu_ps(ws + 16 * s);
        for (int r = 0; r < R; ++r) {
            const std::int8_t* ar = a + r * lda + 16 * s;
            std::int32_t q[4];
            std::memcpy(q, ar, 16);
            __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_xor_si512(_mm512_set1_epi32(q[0]), x80), w0);
            acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(_mm512_set1_epi32(q[1]), x80), w1);
            acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(_mm512_set1_epi32(q[2]), x80), w2);
            acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(_mm512_set1_epi32(q[3]), x80), w3);
            acc = _mm512_sub_epi32(acc, corr);
            const __m512 sc = _mm512_mul_ps(_mm512_set1_ps(sa[r * lsa + s]), sw);
            __m512 t = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), sc);
            IMPBFF_FP4_KEEP(t);
            f[r][j] = _mm512_add_ps(f[r][j], t);
        }
    };
    int s = 0;
    for (; s + 4 <= ns; s += 4) {
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
        step(s + 2, std::integral_constant<int, 2>());
        step(s + 3, std::integral_constant<int, 3>());
    }
    if (s < ns) {
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
    }
    for (int r = 0; r < R; ++r)
        _mm512_storeu_ps(out + r * kNR, _mm512_add_ps(_mm512_add_ps(f[r][0], f[r][1]), _mm512_add_ps(f[r][2], f[r][3])));
}
#elif defined(IMPBFF_FP4_AVX2)
template <int R>
inline void tile(const std::int8_t* a, std::size_t lda, const float* sa, std::size_t lsa, const std::uint8_t* wc,
                 const float* ws, const std::int32_t* wcorr, int ns, float* out) {
    const __m256i tab = _mm256_broadcastsi128_si256(_mm_load_si128(reinterpret_cast<const __m128i*>(kValues2)));
    const __m256i m4 = _mm256_set1_epi8(0x0F), x80 = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 f[R][4];
    for (int r = 0; r < R; ++r)
        for (int j = 0; j < 4; ++j) f[r][j] = _mm256_setzero_ps();
    auto step = [&](int s, auto J) {
        constexpr int j = decltype(J)::value;
        const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc + 64 * s));
        const __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc + 64 * s + 32));
        const __m256i w0 = _mm256_shuffle_epi8(tab, _mm256_and_si256(c0, m4));
        const __m256i w1 = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(c0, 4), m4));
        const __m256i w2 = _mm256_shuffle_epi8(tab, _mm256_and_si256(c1, m4));
        const __m256i w3 = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(c1, 4), m4));
        const __m256i corr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wcorr + 8 * s));
        const __m256 sw = _mm256_loadu_ps(ws + 8 * s);
        for (int r = 0; r < R; ++r) {
            const std::int8_t* ar = a + r * lda + 16 * s;
            std::int32_t q[4];
            std::memcpy(q, ar, 16);
            // (a + 128) x w in int16 pairs (|.| <= 2 * 255 * 12), four quads summed (<= 24480)
            const __m256i p = _mm256_add_epi16(
                    _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_set1_epi32(q[0]), x80), w0),
                                     _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_set1_epi32(q[1]), x80), w1)),
                    _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_xor_si256(_mm256_set1_epi32(q[2]), x80), w2),
                                     _mm256_maddubs_epi16(_mm256_xor_si256(_mm256_set1_epi32(q[3]), x80), w3)));
            const __m256i acc = _mm256_sub_epi32(_mm256_madd_epi16(p, ones), corr);
            const __m256 sc = _mm256_mul_ps(_mm256_set1_ps(sa[r * lsa + s]), sw);
            __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(acc), sc);
            IMPBFF_FP4_KEEP(t);
            f[r][j] = _mm256_add_ps(f[r][j], t);
        }
    };
    int s = 0;
    for (; s + 4 <= ns; s += 4) {
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
        step(s + 2, std::integral_constant<int, 2>());
        step(s + 3, std::integral_constant<int, 3>());
    }
    if (s < ns) {
        step(s, std::integral_constant<int, 0>());
        step(s + 1, std::integral_constant<int, 1>());
    }
    for (int r = 0; r < R; ++r)
        _mm256_storeu_ps(out + r * kNR, _mm256_add_ps(_mm256_add_ps(f[r][0], f[r][1]), _mm256_add_ps(f[r][2], f[r][3])));
}
#else
template <int R>
inline void tile(const std::int8_t* a, std::size_t lda, const float* sa, std::size_t lsa, const std::uint8_t* wc,
                 const float* ws, const std::int32_t*, int ns, float* out) {
    float f[R][4][kNR] = {};
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* c = wc + static_cast<std::size_t>(s) * 8 * kNR;
        for (int r = 0; r < R; ++r) {
            const std::int8_t* ar = a + r * lda + 16 * s;
            for (int o = 0; o < kNR; ++o) {
                std::int32_t sum = 0;
                for (int p = 0; p < 2; ++p)
                    for (int j = 0; j < 4; ++j) {
                        const std::uint8_t b = c[p * 4 * kNR + 4 * o + j];
                        sum += kValues2[b & 0x0F] * ar[8 * p + j] + kValues2[b >> 4] * ar[8 * p + 4 + j];
                    }
                const float sc = sa[r * lsa + s] * ws[static_cast<std::size_t>(s) * kNR + o];
                float t = static_cast<float>(sum) * sc;
                IMPBFF_FP4_KEEP(t);
                f[r][s & 3][o] += t;
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int o = 0; o < kNR; ++o) {
            const float l01 = f[r][0][o] + f[r][1][o];
            const float l23 = f[r][2][o] + f[r][3][o];
            out[r * kNR + o] = l01 + l23;
        }
}
#endif
}  // namespace mk

// ------------------------------------------------------------------ GEMMs

//! C (A.rows x P.rows, row-major, `ldc` = P.rows) = A P^T (+ bias per
//! column when `bias` is given), on the compiled micro-kernel.
inline void gemm_packed(const Q8Rows& A, const PackedRight& P, double* C, const double* bias = nullptr) {
    const int M = A.rows, N = P.rows, ns = P.ns, nt = P.nt;
    if (M <= 0 || N <= 0) return;
    const std::size_t lda = static_cast<std::size_t>(A.kp), lsa = static_cast<std::size_t>(A.n_sub());
    const int nrb = (M + kMR - 1) / kMR;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (static_cast<long>(M) * N * ns > (1L << 16))
#endif
    for (int rb = 0; rb < nrb; ++rb) {
        float out[kMR * kNR];
        const int r0 = rb * kMR, mr = std::min(kMR, M - r0);
        const std::int8_t* a = A.q.data() + static_cast<std::size_t>(r0) * lda;
        const float* sa = A.sc.data() + static_cast<std::size_t>(r0) * lsa;
        for (int t = 0; t < nt; ++t) {
            const std::size_t ts = static_cast<std::size_t>(t) * ns;
            const std::uint8_t* wc = P.codes.data() + ts * 8 * kNR;
            const float* ws = P.sc.data() + ts * kNR;
            const std::int32_t* wk = P.corr.data() + ts * kNR;
            if (mr == kMR) {
                mk::tile<kMR>(a, lda, sa, lsa, wc, ws, wk, ns, out);
            } else {
                for (int r = 0; r < mr; ++r)
                    mk::tile<1>(a + r * lda, lda, sa + r * lsa, lsa, wc, ws, wk, ns, out + r * kNR);
            }
            const int o0 = t * kNR, no = std::min(kNR, N - o0);
            if (no == kNR) {  // constant trip counts: vectorised stores
                for (int r = 0; r < mr; ++r) {
                    double* c = C + static_cast<std::size_t>(r0 + r) * N + o0;
                    const float* v = out + r * kNR;
                    if (bias)
                        for (int o = 0; o < kNR; ++o) c[o] = static_cast<double>(v[o]) + bias[o0 + o];
                    else
                        for (int o = 0; o < kNR; ++o) c[o] = static_cast<double>(v[o]);
                }
                continue;
            }
            for (int r = 0; r < mr; ++r) {
                double* c = C + static_cast<std::size_t>(r0 + r) * N + o0;
                const float* v = out + r * kNR;
                if (bias)
                    for (int o = 0; o < no; ++o) c[o] = static_cast<double>(v[o]) + bias[o0 + o];
                else
                    for (int o = 0; o < no; ++o) c[o] = static_cast<double>(v[o]);
            }
        }
    }
}

//! C (A.rows x W.rows, row-major) = A W^T, A int8, W FP4. `Generic` picks
//! the element-by-element reference (tests); otherwise the micro-kernel.
template <bool Generic = false>
inline void gemm_q8(const Q8Rows& A, const Fp4Rows& W, double* C) {
    if (Generic) {
        gemm_reference(A, W, C);
        return;
    }
    PackedRight P;
    pack_right(W, P);
    gemm_packed(A, P, C);
}

//! C (X.rows x Y.rows, row-major) = X Y^T, both FP4.
template <bool Generic = false>
inline void gemm_fp4(const Fp4Rows& X, const Fp4Rows& Y, double* C) {
    Q8Rows L;
    left_of(X, L);
    gemm_q8<Generic>(L, Y, C);
}

// ------------------------------------------------------------------ forward

//! int8 activation block for a weight format: 16 for nvfp4, else 32.
inline int q8_block(Format f) { return f == Format::NVFP4 ? 16 : 32; }

//! A model with its FP4 layers packed for the micro-kernel (once).
struct Prepared {
    const Fp4Model* model = nullptr;
    std::vector<PackedRight> w;  //!< one a layer (empty for full-precision layers)
};

inline Prepared prepare(const Fp4Model& m) {
    Prepared p;
    p.model = &m;
    p.w.resize(m.layers.size());
    for (std::size_t l = 0; l < m.layers.size(); ++l)
        if (!m.layers[l].full_precision) pack_right(rows_of(m.layers[l].weight), p.w[l]);
    return p;
}

namespace kd {
struct PredictScratch {
    std::vector<double> a, z;
    Q8Rows q8;
    QuantScratch qs;
};
inline PredictScratch& predict_scratch() {
    static thread_local PredictScratch s;
    return s;
}
}  // namespace kd

//! The forward pass on the packed codes, in the model's physical units.
/*! Without `quantize_activations` each layer input is quantised to int8
    per block (W4A8, ggml's scheme); with it, to FP4 in the model's format
    per row (W4A4). Bias, activation and scalers in double; a
    full-precision layer runs in double through `Gemm`. */
template <class Gemm = mlpcore::PortableGemm>
inline void predict(const Prepared& p, const double* X, int n_rows, std::vector<double>& y) {
    const Fp4Model& m = *p.model;
    y.clear();
    if (n_rows <= 0 || m.layers.empty()) return;
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    kd::PredictScratch& s = kd::predict_scratch();
    s.a.assign(X, X + rows * static_cast<std::size_t>(m.n_inputs()));
    mlpcore::detail::scale_in(s.a, n_rows, m.n_inputs(), m.x_scaler);
    for (std::size_t li = 0; li < m.layers.size(); ++li) {
        const Fp4Layer& l = m.layers[li];
        s.z.resize(rows * static_cast<std::size_t>(l.n_out));
        if (l.full_precision) {
            Gemm::nt(n_rows, l.n_out, l.n_in, s.a.data(), l.weight_f64.data(), s.z.data());
            for (std::size_t r = 0; r < rows; ++r) {
                double* zr = s.z.data() + r * static_cast<std::size_t>(l.n_out);
                for (int o = 0; o < l.n_out; ++o) zr[o] += l.bias[static_cast<std::size_t>(o)];
            }
        } else {
            if (m.quantize_activations)
                quantize_left(s.a.data(), n_rows, l.n_in, m.format, Rounding::NearestEven, nullptr, s.q8, s.qs);
            else
                quantize_q8(s.a.data(), n_rows, l.n_in, q8_block(m.format), s.q8);
            gemm_packed(s.q8, p.w[li], s.z.data(), l.bias.data());
        }
        s.a.resize(s.z.size());
        mlpcore::act_apply(s.z.data(), s.a.data(), s.z.size(), l.activation);
    }
    y.assign(s.a.begin(), s.a.end());
    mlpcore::detail::unscale_out(y, n_rows, m.n_outputs(), m.y_scaler);
}

//! predict() for a model not prepared beforehand (packs its weights first).
template <class Gemm = mlpcore::PortableGemm>
inline void predict(const Fp4Model& m, const double* X, int n_rows, std::vector<double>& y) {
    predict<Gemm>(prepare(m), X, n_rows, y);
}

}  // namespace kern
}  // namespace mlpfp4
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPFP4KERNELS_H

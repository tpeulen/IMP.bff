/**
 *  \file IMP/bff/internal/MlpTernary.h
 *  \brief Ternary (1.58-bit) weights: BitNet b1.58's quantisers, the 2-bit
 *         and base-3 packings, and an integer-SIMD ternary x int8 GEMM.
 *
 * **Format (BitNet b1.58, W1.58A8).** Weights are trits `q in {-1, 0, +1}`
 * times one scale a tensor (Ma et al. 2024, arXiv 2402.17764, and the
 * reference code in Microsoft's "The Era of 1-bit LLMs: Training Tips, Code
 * and FAQ"): `gamma = max(mean |W|, 1e-5)` ("absmean", the mean over the
 * whole matrix), `s = 1 / gamma`, `q = clamp(round(w s), -1, 1)`, and the
 * weight's value is `q / s`. Activations are int8 per row ("per token"):
 * `s_a = 127 / max(max |x|, 1e-5)`, `q_a = clamp(round(x s_a), -128, 127)`.
 * `round` is round-half-to-even (torch.round): `w s = +-0.5 -> 0`,
 * `+-1.5 -> +-2 -> +-1`. Option (`Scale::Row`): one absmean a weight row
 * (output) instead of a tensor, for PTQ of a float-trained net; BitNet
 * itself is per tensor. BitNet's BitLinear also normalises its input
 * (RMSNorm) before the activation quantiser; a plain MLP layer here does
 * not (the layer computes `W x + b` like the float network).
 *
 * **Arithmetic.** With per-row activation and per-tensor (or per-row)
 * weight scales a whole dot product is one integer sum:
 * `z[r][o] = (sum_k q[o][k] q_a[r][k]) * (1 / s_a[r]) * (1 / s[o]) + b[o]`,
 * the sum exact in int32, the two products in double, each rounded before
 * the next operation (IMPBFF_FP4_KEEP), so every SIMD variant gives the
 * generic code's bits. The weights enter the kernel as codes `u = q + 1 in
 * {0, 1, 2}` (unsigned, so x86's u8 x s8 `maddubs` / `dpbusd` take them
 * directly), and `sum_k q a = sum_k u a - sum_k a`, the activation row sum
 * computed at quantisation (ggml's TQ2_0 trick).
 *
 * **Layouts.**
 * - Storage `"ternary"` / `"ternary_row"` (TQ2_0-style, 2 bits a weight):
 *   each row's codes `u`, four a byte, element `4 i + j` in bits `2 j`, rows
 *   padded with `u = 1` (trit 0) to a multiple of 4.
 * - Storage `"ternary_tq1"` (TQ1_0-style, 1.6 bits a weight): five trits a
 *   byte in ggml's fixed-point base-3 form -- `v = sum_n u_n 3^(4 - n)`
 *   (`u_0` most significant), stored as `ceil(v 256 / 243)`, so trit n is
 *   recovered as `((byte * 3^n) mod 256) * 3 >> 8`; element `5 i + n` of a
 *   row in byte i, trit n, rows padded with trit 0 to a multiple of 5.
 *   Only the storage differs: loading unpacks to the kernel layout, and
 *   inference is identical to `"ternary"`.
 * - Kernel (in memory, `PackedT`): tiles of `kNR` weight rows; per 16-wide
 *   k sub-block one chunk of `4 kNR` bytes, byte `4 o + j` holding
 *   `u(row o, 16 s + 4 l + j)` in bits `2 l` (l = 0..3). `(chunk >> 2 l) & 3`
 *   is then exactly the operand of a lane-wise 4-byte dot product with the
 *   activations' quad `4 l .. 4 l + 3` broadcast: `vdotq_laneq_s32` (NEON
 *   dotprod; plain NEON emulates it), `_mm256_maddubs_epi16` (AVX2),
 *   `_mm512_dpbusd_epi32` (AVX-512 VNNI). This is bitnet.cpp's I2_S idea
 *   (2-bit codes unpacked with shifts and masks in registers, int8
 *   activations, integer dot products) arranged as the FP4 kernels'
 *   register-blocked micro-kernel (MlpFp4Kernels.h, same variants and
 *   compile-time selection -- `IMPBFF_FP4_NO_SIMD`, `IMPBFF_FP4_NO_DOTPROD`).
 *
 * Sources and notices:
 * - BitNet b1.58: S. Ma et al., "The Era of 1-bit LLMs: All Large Language
 *   Models are in 1.58 Bits", arXiv 2402.17764 (2024); S. Ma, H. Wang,
 *   F. Wei, "The Era of 1-bit LLMs: Training Tips, Code and FAQ"
 *   (github.com/microsoft/unilm, bitnet/; MIT License, Copyright (c)
 *   Microsoft Corporation): `weight_quant`, `activation_quant` and
 *   BitLinear's straight-through estimator, re-expressed here in C++.
 * - bitnet.cpp (github.com/microsoft/BitNet; MIT License, Copyright (c)
 *   Microsoft Corporation): the I2_S idea -- 2-bit codes unpacked with
 *   shifts and masks in registers against int8 activations. No code copied.
 * - TQ1_0 / TQ2_0 from llama.cpp / ggml (github.com/ggml-org/llama.cpp):
 *   ggml/src/ggml-common.h (`block_tq1_0`, `block_tq2_0`),
 *   ggml/src/ggml-quants.c (`quantize_row_tq1_0_ref`,
 *   `dequantize_row_tq1_0`), ggml/src/ggml-cpu/quants.c and arch/{arm,x86}/quants.c
 *   (`ggml_vec_dot_tq{1,2}_0_q8_K`): the base-3 fixed-point byte (encode
 *   `ceil(v 256 / 243)`, decode `(byte 3^n mod 256) 3 >> 8`), the codes
 *   `q + 1` with `sum q a = sum (q + 1) a - sum a`, and the u8 x s8
 *   `maddubs` product. The element order within a row is this file's, not
 *   ggml's 256-element blocks.
 *
 * The ggml and Microsoft parts are used under the MIT License:
 *
 *   MIT License
 *
 *   Copyright (c) 2023-2026 The ggml authors
 *   Copyright (c) Microsoft Corporation.
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
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPTERNARY_H
#define IMPBFF_INTERNAL_MLPTERNARY_H

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpFp4Kernels.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpternary {

namespace kern = mlpfp4::kern;
using kern::kMR;
using kern::kNR;

//! One absmean a tensor (BitNet) or a weight row.
enum class Scale { Tensor, Row };
//! How the trits are stored in a document: 2 bits (TQ2_0-style) or
//! five trits a byte (TQ1_0-style). Inference is the same for both.
enum class Storage { TQ2, TQ1 };

constexpr double kEps = 1e-5;  //!< BitNet's clamp of both scales' denominators

//! A ternary weight matrix `rows x cols` (n_out x n_in): trits, unpacked,
//! row-major, and its absmean scale(s) `gamma` (one, or one a row).
struct TernaryTensor {
    int rows = 0;
    int cols = 0;
    Scale scale = Scale::Tensor;
    std::vector<std::int8_t> q;
    std::vector<double> gamma;
    //! `1 / s` of row `o` (s = 1 / gamma): the value of a +1 trit.
    double step(int o) const {
        const double g = gamma[scale == Scale::Tensor ? 0 : static_cast<std::size_t>(o)];
        return 1.0 / (1.0 / g);
    }
};

//! mean |w| over `n` values in a fixed order on every build: eight
//! interleaved partial sums, combined `((0+1)+(2+3))+((4+5)+(6+7))`.
inline double absmean(const double* w, std::size_t n) {
    double acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += std::abs(w[i + static_cast<std::size_t>(j)]);
    for (int j = 0; i < n; ++i, ++j) acc[j] += std::abs(w[i]);
    const double s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    return n ? s / static_cast<double>(n) : 0.0;
}

//! `clamp(round_half_even(w * s), -1, 1)`; NaN -> 0.
inline std::int8_t trit(double w, double s) {
    const double v = std::nearbyint(w * s);
    return static_cast<std::int8_t>(v >= 1.0 ? 1 : (v <= -1.0 ? -1 : 0));
}

//! BitNet b1.58's weight quantiser (weight_quant), per tensor or per row,
//! into `t` (its buffers reused: training quantises every step).
inline void quantize_into(const double* W, int rows, int cols, Scale sc, TernaryTensor& t) {
    t.rows = rows;
    t.cols = cols;
    t.scale = sc;
    t.q.resize(static_cast<std::size_t>(rows) * cols);
    const std::size_t nc = static_cast<std::size_t>(cols);
    t.gamma.clear();
    if (sc == Scale::Tensor)
        t.gamma.push_back(std::max(absmean(W, static_cast<std::size_t>(rows) * nc), kEps));
    else
        for (int o = 0; o < rows; ++o) t.gamma.push_back(std::max(absmean(W + o * nc, nc), kEps));
    for (int o = 0; o < rows; ++o) {
        const double s = 1.0 / t.gamma[sc == Scale::Tensor ? 0 : static_cast<std::size_t>(o)];
        for (std::size_t k = 0; k < nc; ++k) t.q[o * nc + k] = trit(W[o * nc + k], s);
    }
}

inline TernaryTensor quantize(const double* W, int rows, int cols, Scale sc = Scale::Tensor) {
    TernaryTensor t;
    quantize_into(W, rows, cols, sc, t);
    return t;
}

//! The dequantised weights `q / s` (float64, rows x cols): what the
//! ternary layer multiplies by, exactly (q is -1, 0 or 1).
inline void dequantize(const TernaryTensor& t, double* W) {
    for (int o = 0; o < t.rows; ++o) {
        const double d = t.step(o);
        for (int k = 0; k < t.cols; ++k) {
            const std::size_t i = static_cast<std::size_t>(o) * t.cols + k;
            W[i] = t.q[i] == 0 ? 0.0 : (t.q[i] > 0 ? d : -d);
        }
    }
}

// ------------------------------------------------------------------ storage

//! Bytes a row takes in `st`.
inline std::size_t row_bytes(int cols, Storage st) {
    const std::size_t c = static_cast<std::size_t>(cols);
    return st == Storage::TQ2 ? (c + 3) / 4 : (c + 4) / 5;
}

//! Pack the trits of `t` row by row (see the file comment for both layouts).
inline std::vector<std::uint8_t> pack(const TernaryTensor& t, Storage st) {
    const std::size_t rb = row_bytes(t.cols, st), nc = static_cast<std::size_t>(t.cols);
    std::vector<std::uint8_t> out(rb * static_cast<std::size_t>(t.rows), 0);
    for (int o = 0; o < t.rows; ++o) {
        const std::int8_t* q = t.q.data() + static_cast<std::size_t>(o) * nc;
        std::uint8_t* b = out.data() + static_cast<std::size_t>(o) * rb;
        auto u = [&](std::size_t k) { return k < nc ? static_cast<unsigned>(q[k] + 1) : 1u; };
        for (std::size_t i = 0; i < rb; ++i) {
            if (st == Storage::TQ2) {
                b[i] = static_cast<std::uint8_t>(u(4 * i) | (u(4 * i + 1) << 2) | (u(4 * i + 2) << 4) | (u(4 * i + 3) << 6));
            } else {
                unsigned v = 0;
                for (std::size_t n = 0; n < 5; ++n) v = 3 * v + u(5 * i + n);
                b[i] = static_cast<std::uint8_t>((v * 256 + 242) / 243);  // ceil(v 256 / 243)
            }
        }
    }
    return out;
}

//! Inverse of pack(); throws on a 2-bit code 3 (TQ2). Every TQ1 byte
//! decodes (ceil(242 * 256 / 243) = 255).
inline void unpack(const std::uint8_t* bytes, std::size_t n_bytes, Storage st, TernaryTensor& t) {
    const std::size_t rb = row_bytes(t.cols, st), nc = static_cast<std::size_t>(t.cols);
    if (n_bytes != rb * static_cast<std::size_t>(t.rows)) throw std::runtime_error("ternary codes length");
    t.q.assign(nc * static_cast<std::size_t>(t.rows), 0);
    static const unsigned pow3[5] = {1, 3, 9, 27, 81};
    for (int o = 0; o < t.rows; ++o) {
        const std::uint8_t* b = bytes + static_cast<std::size_t>(o) * rb;
        std::int8_t* q = t.q.data() + static_cast<std::size_t>(o) * nc;
        for (std::size_t i = 0; i < rb; ++i) {
            if (st == Storage::TQ2) {
                for (std::size_t j = 0; j < 4; ++j) {
                    const unsigned u = (b[i] >> (2 * j)) & 3u;
                    if (u == 3) throw std::runtime_error("ternary code 3");
                    if (4 * i + j < nc) q[4 * i + j] = static_cast<std::int8_t>(static_cast<int>(u) - 1);
                }
            } else {
                for (std::size_t n = 0; n < 5; ++n) {
                    const unsigned m = (static_cast<unsigned>(b[i]) * pow3[n]) & 0xFFu;
                    const unsigned u = (m * 3) >> 8;
                    if (5 * i + n < nc) q[5 * i + n] = static_cast<std::int8_t>(static_cast<int>(u) - 1);
                }
            }
        }
    }
}

// ------------------------------------------------------------------ operands

//! The kernel's right operand: codes `u = q + 1` in tiles (file comment),
//! and each row's value of a +1 trit (`step`), `nt * kNR` of them.
struct PackedT {
    int rows = 0;
    int kp = 0;  //!< cols padded to 16
    int ns = 0;  //!< kp / 16
    int nt = 0;  //!< ceil(rows / kNR)
    std::vector<std::uint8_t> codes;
    std::vector<double> step;
};

//! `t` in the kernel layout, into `p` (its buffers reused).
inline void pack_kernel_into(const TernaryTensor& t, PackedT& p) {
    p.rows = t.rows;
    p.kp = (t.cols + 15) / 16 * 16;
    p.ns = p.kp / 16;
    p.nt = (t.rows + kNR - 1) / kNR;
    p.codes.assign(static_cast<std::size_t>(p.nt) * p.ns * 4 * kNR, 0);
    p.step.assign(static_cast<std::size_t>(p.nt) * kNR, 0.0);
    for (int o = 0; o < t.rows; ++o) {
        p.step[static_cast<std::size_t>(o)] = t.step(o);
        const int tile = o / kNR, ot = o % kNR;
        const std::int8_t* q = t.q.data() + static_cast<std::size_t>(o) * t.cols;
        std::uint8_t* base = p.codes.data() + static_cast<std::size_t>(tile) * p.ns * 4 * kNR + 4 * ot;
        for (int k = 0; k < t.cols; ++k) {
            const unsigned u = static_cast<unsigned>(q[k] + 1);
            std::uint8_t& b = base[static_cast<std::size_t>(k / 16) * 4 * kNR + (k % 4)];
            b = static_cast<std::uint8_t>(b | (u << (2 * ((k % 16) / 4))));
        }
    }
}

inline PackedT pack_kernel(const TernaryTensor& t) {
    PackedT p;
    pack_kernel_into(t, p);
    return p;
}

//! The left operand: int8 rows (padded with zeros to `kp`), each row's sum
//! of its values and `1 / s_a` (`step`).
struct A8Rows {
    int rows = 0;
    int kp = 0;
    std::vector<std::int8_t> q;
    std::vector<std::int32_t> sum;
    std::vector<double> step;
};

// ------------------------------------------------------------------ activations

namespace va {
//! One row: `q = clamp(round_half_even(x * s), -128, 127)` (NaN -> 0), and
//! the sum of the codes. The same IEEE product and rounding on every
//! variant (the clamp of an integral value, the conversion exact).
inline std::int32_t round_row_scalar(const double* x, int n, double s, std::int8_t* q) {
    std::int32_t sum = 0;
    for (int k = 0; k < n; ++k) {
        double v = std::nearbyint(x[k] * s);
        v = v == v ? std::min(127.0, std::max(-128.0, v)) : 0.0;
        q[k] = static_cast<std::int8_t>(v);
        sum += q[k];
    }
    return sum;
}
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
inline std::int32_t round_row(const double* x, int n, double s, std::int8_t* q) {
    const float64x2_t sv = vdupq_n_f64(s), lo = vdupq_n_f64(-128.0), hi = vdupq_n_f64(127.0);
    int32x4_t acc = vdupq_n_s32(0);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        int64x2_t c[4];
        for (int j = 0; j < 4; ++j) {
            const float64x2_t xv = vld1q_f64(x + k + 2 * j);
            float64x2_t v = vrndnq_f64(vmulq_f64(xv, sv));
            v = vminq_f64(hi, vmaxq_f64(lo, v));
            v = vbslq_f64(vceqq_f64(v, v), v, vdupq_n_f64(0.0));
            c[j] = vcvtq_s64_f64(v);
        }
        const int32x4_t w0 = vcombine_s32(vmovn_s64(c[0]), vmovn_s64(c[1]));
        const int32x4_t w1 = vcombine_s32(vmovn_s64(c[2]), vmovn_s64(c[3]));
        const int16x8_t h = vcombine_s16(vmovn_s32(w0), vmovn_s32(w1));
        vst1_s8(q + k, vmovn_s16(h));
        acc = vaddq_s32(acc, vaddq_s32(w0, w1));
    }
    return vaddvq_s32(acc) + round_row_scalar(x + k, n - k, s, q + k);
}
#elif defined(IMPBFF_FP4_AVX512) || defined(IMPBFF_FP4_AVX2)
inline std::int32_t round_row(const double* x, int n, double s, std::int8_t* q) {
    const __m256d sv = _mm256_set1_pd(s), lo = _mm256_set1_pd(-128.0), hi = _mm256_set1_pd(127.0);
    __m128i acc = _mm_setzero_si128();
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        __m256d v = _mm256_round_pd(_mm256_mul_pd(_mm256_loadu_pd(x + k), sv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v = _mm256_min_pd(hi, _mm256_max_pd(lo, v));  // NaN: max_pd(lo, NaN) returns NaN, min_pd(hi, NaN) -> NaN
        v = _mm256_and_pd(v, _mm256_cmp_pd(v, v, _CMP_ORD_Q));
        const __m128i c = _mm256_cvttpd_epi32(v);
        acc = _mm_add_epi32(acc, c);
        const __m128i b = _mm_packs_epi16(_mm_packs_epi32(c, c), _mm_setzero_si128());
        const int w = _mm_cvtsi128_si32(b);
        std::memcpy(q + k, &w, 4);
    }
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0x4E));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0xB1));
    return _mm_cvtsi128_si32(acc) + round_row_scalar(x + k, n - k, s, q + k);
}
#else
inline std::int32_t round_row(const double* x, int n, double s, std::int8_t* q) { return round_row_scalar(x, n, s, q); }
#endif
}  // namespace va

//! BitNet's activation quantiser (activation_quant) on `rows x K` doubles
//! (row stride `ld`): per row `s_a = 127 / max(max |x|, 1e-5)`.
inline void quantize_rows(const double* A, int rows, int K, std::size_t ld, A8Rows& out) {
    out.rows = rows;
    out.kp = (K + 15) / 16 * 16;
    out.q.assign(static_cast<std::size_t>(rows) * out.kp, 0);
    out.sum.resize(static_cast<std::size_t>(rows));
    out.step.resize(static_cast<std::size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        const double* x = A + static_cast<std::size_t>(r) * ld;
        const double s = 127.0 / std::max(kern::vq::absmax(x, K), kEps);
        out.sum[static_cast<std::size_t>(r)] = va::round_row(x, K, s, out.q.data() + static_cast<std::size_t>(r) * out.kp);
        out.step[static_cast<std::size_t>(r)] = 1.0 / s;
    }
}

// ------------------------------------------------------------------ micro-kernel
//
// tile<R>: R left rows x kNR right rows, the exact int32 sums
// `sum_k u[o][k] a[r][k]` over all `ns` sub-blocks into out[r * kNR + o].

namespace mk {
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const std::uint8_t* wc, int ns, std::int32_t* out) {
    using kern::mk::dot_lane;
    const uint8x16_t m3 = vdupq_n_u8(3);
    int32x4_t acc[R];
    for (int r = 0; r < R; ++r) acc[r] = vdupq_n_s32(0);
    for (int s = 0; s < ns; ++s) {
        const uint8x16_t c = vld1q_u8(wc + 16 * s);
        const int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8(c, m3));
        const int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(c, 2), m3));
        const int8x16_t w2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(c, 4), m3));
        const int8x16_t w3 = vreinterpretq_s8_u8(vshrq_n_u8(c, 6));
        for (int r = 0; r < R; ++r) {
            const int8x16_t av = vld1q_s8(a + r * lda + 16 * s);
            acc[r] = dot_lane<0>(acc[r], w0, av);
            acc[r] = dot_lane<1>(acc[r], w1, av);
            acc[r] = dot_lane<2>(acc[r], w2, av);
            acc[r] = dot_lane<3>(acc[r], w3, av);
        }
    }
    for (int r = 0; r < R; ++r) vst1q_s32(out + r * kNR, acc[r]);
}
#elif defined(IMPBFF_FP4_AVX512)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const std::uint8_t* wc, int ns, std::int32_t* out) {
    const __m512i m3 = _mm512_set1_epi8(3);
    __m512i acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_si512();
    for (int s = 0; s < ns; ++s) {
        const __m512i c = _mm512_loadu_si512(wc + 64 * s);
        const __m512i w0 = _mm512_and_si512(c, m3);
        const __m512i w1 = _mm512_and_si512(_mm512_srli_epi16(c, 2), m3);
        const __m512i w2 = _mm512_and_si512(_mm512_srli_epi16(c, 4), m3);
        const __m512i w3 = _mm512_and_si512(_mm512_srli_epi16(c, 6), m3);
        for (int r = 0; r < R; ++r) {
            std::int32_t q[4];
            std::memcpy(q, a + r * lda + 16 * s, 16);
            acc[r] = _mm512_dpbusd_epi32(acc[r], w0, _mm512_set1_epi32(q[0]));
            acc[r] = _mm512_dpbusd_epi32(acc[r], w1, _mm512_set1_epi32(q[1]));
            acc[r] = _mm512_dpbusd_epi32(acc[r], w2, _mm512_set1_epi32(q[2]));
            acc[r] = _mm512_dpbusd_epi32(acc[r], w3, _mm512_set1_epi32(q[3]));
        }
    }
    for (int r = 0; r < R; ++r) _mm512_storeu_si512(out + r * kNR, acc[r]);
}
#elif defined(IMPBFF_FP4_AVX2)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const std::uint8_t* wc, int ns, std::int32_t* out) {
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    __m256i acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_si256();
    for (int s = 0; s < ns; ++s) {
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wc + 32 * s));
        const __m256i w0 = _mm256_and_si256(c, m3);
        const __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(c, 2), m3);
        const __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(c, 4), m3);
        const __m256i w3 = _mm256_and_si256(_mm256_srli_epi16(c, 6), m3);
        for (int r = 0; r < R; ++r) {
            std::int32_t q[4];
            std::memcpy(q, a + r * lda + 16 * s, 16);
            // u (0..2) x a (int8) in int16 pairs (|.| <= 512), four quads summed (<= 2048)
            const __m256i p = _mm256_add_epi16(
                    _mm256_add_epi16(_mm256_maddubs_epi16(w0, _mm256_set1_epi32(q[0])),
                                     _mm256_maddubs_epi16(w1, _mm256_set1_epi32(q[1]))),
                    _mm256_add_epi16(_mm256_maddubs_epi16(w2, _mm256_set1_epi32(q[2])),
                                     _mm256_maddubs_epi16(w3, _mm256_set1_epi32(q[3]))));
            acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(p, ones));
        }
    }
    for (int r = 0; r < R; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + r * kNR), acc[r]);
}
#else
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const std::uint8_t* wc, int ns, std::int32_t* out) {
    std::int32_t acc[R][kNR] = {};
    for (int s = 0; s < ns; ++s) {
        // unpack the sub-block's codes once (row o, element 4 l + j), then
        // plain int8 dot products of 16 the compiler can vectorise
        const std::uint8_t* c = wc + static_cast<std::size_t>(s) * 4 * kNR;
        std::int8_t w[kNR][16];
        for (int o = 0; o < kNR; ++o)
            for (int l = 0; l < 4; ++l)
                for (int j = 0; j < 4; ++j) w[o][4 * l + j] = static_cast<std::int8_t>((c[4 * o + j] >> (2 * l)) & 3);
        for (int r = 0; r < R; ++r) {
            const std::int8_t* ar = a + r * lda + 16 * s;
            for (int o = 0; o < kNR; ++o) {
                std::int32_t d = 0;
                for (int k = 0; k < 16; ++k) d += static_cast<std::int32_t>(w[o][k]) * ar[k];
                acc[r][o] += d;
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int o = 0; o < kNR; ++o) out[r * kNR + o] = acc[r][o];
}
#endif
}  // namespace mk

// ------------------------------------------------------------------ GEMM

//! C (A.rows x P.rows, row-major) = dequantised A times dequantised P^T
//! (+ bias per column): `((sum - rowsum) * step_a) * step_w (+ b)`, each
//! product rounded on its own.
inline void gemm(const A8Rows& A, const PackedT& P, double* C, const double* bias = nullptr) {
    const int M = A.rows, N = P.rows, ns = P.ns, nt = P.nt;
    if (M <= 0 || N <= 0) return;
    if (A.kp != P.kp) throw std::runtime_error("ternary gemm: operand widths differ");
    const std::size_t lda = static_cast<std::size_t>(A.kp);
    const int nrb = (M + kMR - 1) / kMR;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (static_cast<long>(M) * N * ns > (1L << 16))
#endif
    for (int rb = 0; rb < nrb; ++rb) {
        std::int32_t out[kMR * kNR];
        const int r0 = rb * kMR, mr = std::min(kMR, M - r0);
        const std::int8_t* a = A.q.data() + static_cast<std::size_t>(r0) * lda;
        for (int t = 0; t < nt; ++t) {
            const std::uint8_t* wc = P.codes.data() + static_cast<std::size_t>(t) * ns * 4 * kNR;
            if (mr == kMR) {
                mk::tile<kMR>(a, lda, wc, ns, out);
            } else {
                for (int r = 0; r < mr; ++r) mk::tile<1>(a + r * lda, lda, wc, ns, out + r * kNR);
            }
            const int o0 = t * kNR, no = std::min(kNR, N - o0);
            const double* sw = P.step.data() + o0;
            for (int r = 0; r < mr; ++r) {
                double* c = C + static_cast<std::size_t>(r0 + r) * N + o0;
                const std::int32_t* v = out + r * kNR;
                const std::int32_t rs = A.sum[static_cast<std::size_t>(r0 + r)];
                const double sa = A.step[static_cast<std::size_t>(r0 + r)];
                for (int o = 0; o < no; ++o) {
                    double u = static_cast<double>(v[o] - rs) * sa;
                    IMPBFF_FP4_KEEP(u);
                    double w = u * sw[o];
                    IMPBFF_FP4_KEEP(w);
                    c[o] = bias ? w + bias[o0 + o] : w;
                }
            }
        }
    }
}

//! The same product element by element from the unpacked trits (tests).
inline void gemm_reference(const A8Rows& A, const TernaryTensor& W, double* C) {
    for (int r = 0; r < A.rows; ++r)
        for (int o = 0; o < W.rows; ++o) {
            std::int32_t s = 0;
            for (int k = 0; k < W.cols; ++k)
                s += static_cast<std::int32_t>(W.q[static_cast<std::size_t>(o) * W.cols + k]) *
                     A.q[static_cast<std::size_t>(r) * A.kp + k];
            double u = static_cast<double>(s) * A.step[static_cast<std::size_t>(r)];
            IMPBFF_FP4_KEEP(u);
            double w = u * W.step(o);
            IMPBFF_FP4_KEEP(w);
            C[static_cast<std::size_t>(r) * W.rows + o] = w;
        }
}

// ------------------------------------------------------------------ model

//! A layer: ternary weights, or float64 (a layer ternary training kept).
struct TLayer {
    int n_in = 0;
    int n_out = 0;
    Activation activation = Activation::Identity;
    std::vector<double> bias;
    bool full_precision = false;
    std::vector<double> weight_f64;
    TernaryTensor weight;
};

//! A ternary network: layers, the scalers, how its trits are stored.
struct TModel {
    Scale scale = Scale::Tensor;
    Storage storage = Storage::TQ2;
    std::vector<TLayer> layers;
    StandardScaler x_scaler, y_scaler;
    int n_inputs() const { return layers.empty() ? 0 : layers.front().n_in; }
    int n_outputs() const { return layers.empty() ? 0 : layers.back().n_out; }
    std::size_t n_weights() const {
        std::size_t n = 0;
        for (const TLayer& l : layers) n += static_cast<std::size_t>(l.n_in) * l.n_out;
        return n;
    }
    //! Stored bytes of the weights: packed trits plus 8 a scale (float64
    //! layers 8 a weight).
    std::size_t weight_bytes() const {
        std::size_t n = 0;
        for (const TLayer& l : layers)
            n += l.full_precision ? 8 * l.weight_f64.size()
                                  : row_bytes(l.n_in, storage) * l.n_out + 8 * l.weight.gamma.size();
        return n;
    }
};

//! The format name of (scale, storage): "ternary", "ternary_row", "ternary_tq1".
inline std::string format_name(Scale s, Storage st) {
    if (st == Storage::TQ1) return s == Scale::Row ? "ternary_tq1_row" : "ternary_tq1";
    return s == Scale::Row ? "ternary_row" : "ternary";
}
//! Whether `name` is a ternary format; sets scale and storage.
inline bool parse_format(const std::string& name, Scale& s, Storage& st) {
    if (name == "ternary") { s = Scale::Tensor; st = Storage::TQ2; return true; }
    if (name == "ternary_row") { s = Scale::Row; st = Storage::TQ2; return true; }
    if (name == "ternary_tq1") { s = Scale::Tensor; st = Storage::TQ1; return true; }
    if (name == "ternary_tq1_row") { s = Scale::Row; st = Storage::TQ1; return true; }
    return false;
}

//! Post-training quantisation of a float64 model: every layer ternary.
inline TModel quantize(const MlpModel& m, Scale s = Scale::Tensor, Storage st = Storage::TQ2) {
    TModel out;
    out.scale = s;
    out.storage = st;
    out.x_scaler = m.x_scaler;
    out.y_scaler = m.y_scaler;
    for (const DenseLayer& d : m.layers) {
        TLayer l;
        l.n_in = d.n_in;
        l.n_out = d.n_out;
        l.activation = d.activation;
        l.bias = d.bias;
        l.weight = quantize(d.weight.data(), d.n_out, d.n_in, s);
        out.layers.push_back(std::move(l));
    }
    return out;
}

//! A model with its ternary layers in the kernel layout (once).
struct Prepared {
    const TModel* model = nullptr;
    std::vector<PackedT> w;  //!< one a layer (empty for float64 layers)
};

inline Prepared prepare(const TModel& m) {
    Prepared p;
    p.model = &m;
    p.w.resize(m.layers.size());
    for (std::size_t l = 0; l < m.layers.size(); ++l)
        if (!m.layers[l].full_precision) p.w[l] = pack_kernel(m.layers[l].weight);
    return p;
}

namespace kd {
struct PredictScratch {
    std::vector<double> a, z;
    A8Rows q8;
};
inline PredictScratch& predict_scratch() {
    static thread_local PredictScratch s;
    return s;
}
}  // namespace kd

//! One layer's pre-activation `z` of a batch `a` (`rows x n_in`): ternary
//! layers quantise `a` per row to int8 and run the kernel, float64 layers
//! run through `Gemm`. Training's forward pass calls the same function.
template <class Gemm>
inline void layer_forward(const TLayer& l, const PackedT* p, const double* a, int rows, A8Rows& q8, double* z) {
    if (l.full_precision) {
        Gemm::nt(rows, l.n_out, l.n_in, a, l.weight_f64.data(), z);
        for (int r = 0; r < rows; ++r) {
            double* zr = z + static_cast<std::size_t>(r) * l.n_out;
            for (int o = 0; o < l.n_out; ++o) zr[o] += l.bias[static_cast<std::size_t>(o)];
        }
        return;
    }
    quantize_rows(a, rows, l.n_in, static_cast<std::size_t>(l.n_in), q8);
    gemm(q8, *p, z, l.bias.data());
}

//! The forward pass in the model's physical units (W1.58A8): bias,
//! activation and scalers in double.
template <class Gemm = mlpcore::PortableGemm>
inline void predict(const Prepared& p, const double* X, int n_rows, std::vector<double>& y) {
    const TModel& m = *p.model;
    y.clear();
    if (n_rows <= 0 || m.layers.empty()) return;
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    kd::PredictScratch& s = kd::predict_scratch();
    s.a.assign(X, X + rows * static_cast<std::size_t>(m.n_inputs()));
    mlpcore::detail::scale_in(s.a, n_rows, m.n_inputs(), m.x_scaler);
    for (std::size_t li = 0; li < m.layers.size(); ++li) {
        const TLayer& l = m.layers[li];
        s.z.resize(rows * static_cast<std::size_t>(l.n_out));
        layer_forward<Gemm>(l, &p.w[li], s.a.data(), n_rows, s.q8, s.z.data());
        s.a.resize(s.z.size());
        mlpcore::act_apply(s.z.data(), s.a.data(), s.z.size(), l.activation);
    }
    y.assign(s.a.begin(), s.a.end());
    mlpcore::detail::unscale_out(y, n_rows, m.n_outputs(), m.y_scaler);
}

template <class Gemm = mlpcore::PortableGemm>
inline void predict(const TModel& m, const double* X, int n_rows, std::vector<double>& y) {
    predict<Gemm>(prepare(m), X, n_rows, y);
}

}  // namespace mlpternary
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPTERNARY_H

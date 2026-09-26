/**
 *  \file IMP/bff/internal/MlpTernaryGrad.h
 *  \brief Low-precision backward pass of a ternary layer: int8 dgrad on the
 *         ternary x int8 kernel (SwitchBack) and an int8 x int8 wgrad with
 *         per-block stochastic rounding (Jetfire-style).
 *
 * A ternary layer computes `Z = Â Ŵ^T + b` with `Â = q_a / s_a` (int8 per
 * row, from fprop) and `Ŵ = q / s` (trits). The straight-through estimator
 * (MlpTernaryTrain.h) gives `dA = dZ Ŵ` and `dW = dZ^T Â`. Here both GEMMs
 * run on integers:
 *
 * - **dgrad** (SwitchBack; Wortsman et al. 2023, arXiv 2304.13013, and
 *   bitsandbytes' `SwitchBackLinear`): `dZ` is quantised to int8 per row
 *   (round to nearest even, `s = 127 / max |dZ_r|`; no 1e-5 floor -- a
 *   gradient row may be tiny; an all-zero row has step 0), and `dZ Ŵ` runs
 *   on the ternary x int8 kernel with the trits of `Ŵ^T` packed once a
 *   step (pack_transposed_into). With a per-row weight scale (Scale::Row)
 *   `dZ`'s column o is multiplied by row o's step first, so the transposed
 *   trits carry step 1.
 * - **wgrad** (Jetfire; Xi et al., ICML 2024, arXiv 2403.12422: per-block
 *   int8 with the block coefficients applied during accumulation). `Â`'s
 *   per-row scale lies on the contraction (batch) index, so it is folded
 *   into the gradient, `dZ'[r][o] = dZ[r][o] / s_a[r]`, and `Â`'s int8 codes
 *   from fprop are used as they are (not re-quantised). `dZ'^T` (n_out x
 *   batch) is quantised to int8 per block of kGradBlock = 32 batch rows a
 *   column o (Jetfire's block size), `s = 127 / max |block|`, with
 *   **stochastic rounding** `q = floor(float(x s) + u)`, `u = (u16 + 1/2) /
 *   2^16` from the counter-based draws of SrKey (MlpFp4.h; unbiased to
 *   2^-17), clamped to [-127, 127]. Then `dW[o][k] = sum_b step[o][b] *
 *   (sum_{r in b} q[o][r] q_a[r][k])`: the block sums exact in int32 on the
 *   int8 x int8 micro-kernel, the combine in float, one block after the
 *   other, each product rounded before the add (IMPBFF_FP4_KEEP) -- so every
 *   variant gives the generic code's bits.
 *
 * int8 x int8 micro-kernel, register-blocked like MlpTernary.h's (kMR x
 * kNR, 16-wide sub-blocks, the right operand's tile chunks `4 kNR` bytes a
 * k-quad, the left quad broadcast): `vdotq_laneq_s32` (NEON dotprod; plain
 * NEON `vmull_s8` + `vpaddlq_s16`), `_mm256_maddubs_epi16` on `|w|` and
 * `sign(a, w)` (AVX2: codes in [-127, 127], so the int16 pair sums cannot
 * saturate), `_mm512_dpbusd_epi32` on `w + 128` minus `128 x` the left
 * block sum (AVX-512 VNNI), plain int32 loops (generic).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPTERNARYGRAD_H
#define IMPBFF_INTERNAL_MLPTERNARYGRAD_H

#include <IMP/bff/internal/MlpTernary.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpternary {

using mlpfp4::SrKey;

constexpr int kGradBlock = 32;  //!< wgrad's quantisation block along the batch

// ------------------------------------------------------------------ dgrad

//! dgrad's left operand: every row of `G` (`rows x K`, stride `ld`) to int8,
//! round to nearest even at `s = 127 / max |row|` (0 for a zero row).
inline void quantize_grad_rows(const double* G, int rows, int K, std::size_t ld, A8Rows& out) {
    out.rows = rows;
    out.kp = (K + 15) / 16 * 16;
    out.q.assign(static_cast<std::size_t>(rows) * out.kp, 0);
    out.sum.resize(static_cast<std::size_t>(rows));
    out.step.resize(static_cast<std::size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        const double* x = G + static_cast<std::size_t>(r) * ld;
        const double amax = kern::vq::absmax(x, K);
        double s = amax > 0.0 ? 127.0 / amax : 0.0;
        if (!(s < 1e308)) s = 0.0;  // subnormal absmax: the row is dropped
        out.sum[static_cast<std::size_t>(r)] = va::round_row(x, K, s, out.q.data() + static_cast<std::size_t>(r) * out.kp);
        out.step[static_cast<std::size_t>(r)] = s > 0.0 ? 1.0 / s : 0.0;
    }
}

//! The trits of `t` transposed (`cols x rows`) in the kernel layout,
//! straight from `t`: dgrad's right operand. A per-tensor scale is kept, a
//! per-row scale becomes step 1 (dgrad() folds it into the gradient).
inline void pack_transposed_into(const TernaryTensor& t, PackedT& p) {
    p.rows = t.cols;
    p.kp = (t.rows + 15) / 16 * 16;
    p.ns = p.kp / 16;
    p.nt = (t.cols + kNR - 1) / kNR;
    p.codes.assign(static_cast<std::size_t>(p.nt) * p.ns * 4 * kNR, 0);
    p.step.assign(static_cast<std::size_t>(p.nt) * kNR, 0.0);
    const double st = t.scale == Scale::Tensor ? t.step(0) : 1.0;
    for (int k = 0; k < t.cols; ++k) p.step[static_cast<std::size_t>(k)] = st;
    const std::size_t tile_bytes = static_cast<std::size_t>(p.ns) * 4 * kNR;
    // byte j of sub-block s, row k: codes of W rows 16 s + 4 l + j (l = 0..3)
    // at column k in bits 2 l -- four rows of W read along k; rows past
    // t.rows give code 0 (their gradient codes are 0)
    std::vector<std::int8_t> none(static_cast<std::size_t>(t.cols), -1);
    for (int s = 0; s < p.ns; ++s)
        for (int j = 0; j < 4; ++j) {
            const std::int8_t* w[4];
            for (int l = 0; l < 4; ++l) {
                const int o = 16 * s + 4 * l + j;
                w[l] = o < t.rows ? t.q.data() + static_cast<std::size_t>(o) * t.cols : none.data();
            }
            std::uint8_t* base = p.codes.data() + static_cast<std::size_t>(s) * 4 * kNR + j;
            for (int k = 0; k < t.cols; ++k)
                base[static_cast<std::size_t>(k / kNR) * tile_bytes + 4 * static_cast<std::size_t>(k % kNR)] =
                        static_cast<std::uint8_t>((w[0][k] + 1) | ((w[1][k] + 1) << 2) | ((w[2][k] + 1) << 4) |
                                                  ((w[3][k] + 1) << 6));
        }
}

//! `dA = dZ Ŵ` (`bs x t.cols`) on the ternary x int8 kernel; `wt` is
//! pack_transposed_into(t); `buf`, `q` scratch.
inline void dgrad(const double* dZ, int bs, const TernaryTensor& t, const PackedT& wt, std::vector<double>& buf,
                  A8Rows& q, double* dA) {
    const double* g = dZ;
    if (t.scale == Scale::Row) {
        buf.resize(static_cast<std::size_t>(bs) * t.rows);
        for (int r = 0; r < bs; ++r)
            for (int o = 0; o < t.rows; ++o) {
                const std::size_t i = static_cast<std::size_t>(r) * t.rows + o;
                buf[i] = dZ[i] * t.step(o);
            }
        g = buf.data();
    }
    quantize_grad_rows(g, bs, t.rows, static_cast<std::size_t>(t.rows), q);
    gemm(q, wt, dA);
}

// ------------------------------------------------------------------ wgrad's left operand

//! `rows` (n_out) rows of int8 codes over the batch padded to `kp` (a
//! multiple of kGradBlock), one float step and one code sum a block.
struct G8Blocks {
    int rows = 0;
    int kp = 0;
    int nb = 0;  //!< kp / kGradBlock
    std::vector<std::int8_t> q;
    std::vector<float> step;
    std::vector<std::int32_t> sum;
};

namespace vg {
//! One stochastically rounded code: `p = float(x s)`, `u = u16 / 2^16 +
//! 2^-17` (exact in float), `clamp(floor(p + u), -127, 127)` in float, NaN
//! -> 0. Float lanes (4 NEON, 8 AVX2) carry these exact operations.
inline std::int8_t sr_code(double x, double s, std::uint32_t u16) {
    const float p = static_cast<float>(x * s);
    const float u = static_cast<float>(u16) * (1.0f / 65536.0f) + (1.0f / 131072.0f);
    float v = std::floor(p + u);
    v = v == v ? std::min(127.0f, std::max(-127.0f, v)) : 0.0f;
    return static_cast<std::int8_t>(v);
}
//! 16 codes of one group: element j draws the low (j < 8) or high half of
//! word `n0 + (j & 7)` (SrKey::u16's order).
inline void sr_group16_scalar(const double* x, double s, const SrKey& key, std::uint32_t n0, std::int8_t* q) {
    for (int j = 0; j < 16; ++j) {
        const std::uint32_t w = key.word(n0 + static_cast<std::uint32_t>(j & 7));
        q[j] = sr_code(x[j], s, (j & 8) ? (w >> 16) : (w & 0xFFFFu));
    }
}
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
IMPBFF_FP4_INLINE void sr_group16(const double* x, double s, const SrKey& key, std::uint32_t n0, std::int8_t* q) {
    using kern::tq::mix32v;
    static const std::uint32_t kLane[4] = {0, 1, 2, 3};
    const uint32x4_t n = vaddq_u32(vdupq_n_u32(n0), vld1q_u32(kLane));
    const uint32x4_t a = vdupq_n_u32(key.a), b = vdupq_n_u32(key.b);
    const uint32x4_t w0 = mix32v(veorq_u32(mix32v(veorq_u32(n, a)), b));
    const uint32x4_t w1 = mix32v(veorq_u32(mix32v(veorq_u32(vaddq_u32(n, vdupq_n_u32(4)), a)), b));
    const uint32x4_t m = vdupq_n_u32(0xFFFFu);
    // elements 0-3, 4-7: low halves of words 0-3, 4-7; 8-11, 12-15: high halves
    const uint32x4_t u[4] = {vandq_u32(w0, m), vandq_u32(w1, m), vshrq_n_u32(w0, 16), vshrq_n_u32(w1, 16)};
    const float64x2_t sv = vdupq_n_f64(s);
    const float32x4_t sc = vdupq_n_f32(1.0f / 65536.0f), hf = vdupq_n_f32(1.0f / 131072.0f);
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    int16x4_t h[4];
    for (int i = 0; i < 4; ++i) {
        const float32x4_t p = vcvt_high_f32_f64(vcvt_f32_f64(vmulq_f64(vld1q_f64(x + 4 * i), sv)),
                                                vmulq_f64(vld1q_f64(x + 4 * i + 2), sv));
        const float32x4_t uf = vaddq_f32(vmulq_f32(vcvtq_f32_u32(u[i]), sc), hf);  // exact
        // floor and convert in one (NaN -> 0, +-inf saturate), then the clamp
        const int32x4_t c = vminq_s32(hi, vmaxq_s32(lo, vcvtmq_s32_f32(vaddq_f32(p, uf))));
        h[i] = vmovn_s32(c);
    }
    vst1q_s8(q, vcombine_s8(vmovn_s16(vcombine_s16(h[0], h[1])), vmovn_s16(vcombine_s16(h[2], h[3]))));
}
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
IMPBFF_FP4_INLINE void sr_group16(const double* x, double s, const SrKey& key, std::uint32_t n0, std::int8_t* q) {
    const __m256i w = kern::tq::words8(key, n0);
    const __m256i u[2] = {_mm256_and_si256(w, _mm256_set1_epi32(0xFFFF)), _mm256_srli_epi32(w, 16)};  // elements 0-7, 8-15
    const __m256d sv = _mm256_set1_pd(s);
    const __m256 sc = _mm256_set1_ps(1.0f / 65536.0f), hf = _mm256_set1_ps(1.0f / 131072.0f);
    const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    __m128i c[4];
    for (int i = 0; i < 2; ++i) {
        const __m128 p0 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_loadu_pd(x + 8 * i), sv));
        const __m128 p1 = _mm256_cvtpd_ps(_mm256_mul_pd(_mm256_loadu_pd(x + 8 * i + 4), sv));
        const __m256 p = _mm256_insertf128_ps(_mm256_castps128_ps256(p0), p1, 1);
        const __m256 uf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(u[i]), sc), hf);  // exact
        __m256 v = _mm256_floor_ps(_mm256_add_ps(p, uf));
        v = _mm256_min_ps(hi, _mm256_max_ps(lo, v));  // NaN stays NaN (second operand)
        v = _mm256_and_ps(v, _mm256_cmp_ps(v, v, _CMP_ORD_Q));
        const __m256i ci = _mm256_cvttps_epi32(v);
        c[2 * i] = _mm256_castsi256_si128(ci);
        c[2 * i + 1] = _mm256_extracti128_si256(ci, 1);
    }
    const __m128i b = _mm_packs_epi16(_mm_packs_epi32(c[0], c[1]), _mm_packs_epi32(c[2], c[3]));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q), b);
}
#else
inline void sr_group16(const double* x, double s, const SrKey& key, std::uint32_t n0, std::int8_t* q) {
    sr_group16_scalar(x, s, key, n0, q);
}
#endif
}  // namespace vg

//! wgrad's left operand: `dZ'^T` with `dZ'[r][o] = dZ[r][o] * step_a[r]`
//! (`dZ` is `bs x n_out`), per block of kGradBlock batch rows a column,
//! stochastically rounded with `key` (element `o * kp + r`); `T` scratch
//! (one block of `dZ'^T` at a time).
inline void quantize_wgrad_left(const double* dZ, int bs, int n_out, const double* step_a, const SrKey& key,
                                std::vector<double>& T, G8Blocks& out) {
    constexpr int B = kGradBlock;
    const int kp = (bs + B - 1) / B * B;
    out.rows = n_out;
    out.kp = kp;
    out.nb = kp / B;
    out.q.resize(static_cast<std::size_t>(n_out) * kp);
    out.step.resize(static_cast<std::size_t>(n_out) * out.nb);
    out.sum.resize(out.step.size());
    T.resize(static_cast<std::size_t>(n_out) * B);
    for (int b = 0; b < out.nb; ++b) {
        const int r0 = b * B, nr = std::min(B, bs - r0);
        if (nr < B) std::fill(T.begin(), T.end(), 0.0);
        for (int o0 = 0; o0 < n_out; o0 += 8) {  // T[o][rr] = dZ'[r0 + rr][o], 8 columns at a time
            const int no = std::min(8, n_out - o0);
            for (int rr = 0; rr < nr; ++rr) {
                const double sa = step_a[r0 + rr];
                const double* g = dZ + static_cast<std::size_t>(r0 + rr) * n_out + o0;
                double* t = T.data() + static_cast<std::size_t>(o0) * B + rr;
                for (int j = 0; j < no; ++j) t[static_cast<std::size_t>(j) * B] = g[j] * sa;
            }
        }
        for (int o = 0; o < n_out; ++o) {
            const double* x = T.data() + static_cast<std::size_t>(o) * B;
            const std::size_t e = static_cast<std::size_t>(o) * kp + static_cast<std::size_t>(r0);
            const double amax = kern::vq::absmax(x, B);
            double s = amax > 0.0 ? 127.0 / amax : 0.0;
            if (!(s < 1e308)) s = 0.0;
            std::int8_t* q = out.q.data() + e;
            for (int g = 0; g < B; g += 16)
                vg::sr_group16(x + g, s, key, static_cast<std::uint32_t>((e + g) >> 4) * 8u, q + g);
            std::int32_t sum = 0;
            for (int k = 0; k < B; ++k) sum += q[k];
            out.step[static_cast<std::size_t>(o) * out.nb + b] = s > 0.0 ? static_cast<float>(amax / 127.0) : 0.0f;
            out.sum[static_cast<std::size_t>(o) * out.nb + b] = sum;
        }
    }
}

// ------------------------------------------------------------------ wgrad's right operand

//! `Â^T` (n_in rows over the batch, padded to `kp`) in tiles of kNR rows:
//! per 16-wide sub-block four chunks of `4 kNR` bytes, chunk l byte `4 o + j`
//! = row o's element `16 s + 4 l + j` -- the operand of the lane-wise dot.
struct PackedI8 {
    int rows = 0;
    int kp = 0;
    int nsb = 0;  //!< kp / 16
    int nt = 0;
    std::vector<std::int8_t> codes;
};

//! Pack fprop's int8 activations `A` (`bs x n_in`, codes only) as `Â^T`.
inline void pack_wgrad_right(const A8Rows& A, int n_in, int kp, PackedI8& p) {
    p.rows = n_in;
    p.kp = kp;
    p.nsb = kp / 16;
    p.nt = (n_in + kNR - 1) / kNR;
    p.codes.assign(static_cast<std::size_t>(p.nt) * p.nsb * 16 * kNR, 0);
    const std::size_t tile_bytes = static_cast<std::size_t>(p.nsb) * 16 * kNR;
    std::vector<std::int8_t> zero(static_cast<std::size_t>(n_in), 0);
    for (int r = 0; r < A.rows; r += 4) {  // a quad of batch rows: 4 bytes at a time
        const std::int8_t* a[4];
        for (int j = 0; j < 4; ++j)
            a[j] = r + j < A.rows ? A.q.data() + static_cast<std::size_t>(r + j) * A.kp : zero.data();
        std::int8_t* base = p.codes.data() + static_cast<std::size_t>(r / 16) * 16 * kNR +
                            static_cast<std::size_t>((r % 16) / 4) * 4 * kNR;
        for (int k = 0; k < n_in; ++k) {
            std::int8_t* d = base + static_cast<std::size_t>(k / kNR) * tile_bytes + 4 * static_cast<std::size_t>(k % kNR);
            d[0] = a[0][k];
            d[1] = a[1][k];
            d[2] = a[2][k];
            d[3] = a[3][k];
        }
    }
}

// ------------------------------------------------------------------ int8 x int8 micro-kernel
//
// tile<R>: R left rows (stride lda; steps / sums `nb` a row) x kNR right
// rows; per block the exact int32 sums over its two sub-blocks, then
// `f += keep(float(sum) * step)`; out[r * kNR + o] = f.

namespace mk8 {
//! Left rows a tile: kMR, but 4 on AVX2 (its tile has 8 right rows; four
//! left rows amortise the right chunks' loads and |w| over more products).
#if defined(IMPBFF_FP4_AVX2)
constexpr int kMR8 = 4;
#else
constexpr int kMR8 = kMR;
#endif
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const float* st, const std::int32_t*,
                            const std::int8_t* wc, int nb, float* out) {
    using kern::mk::dot_lane;
    float32x4_t f[R];
    for (int r = 0; r < R; ++r) f[r] = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; ++b) {
        int32x4_t acc[R];
        for (int r = 0; r < R; ++r) acc[r] = vdupq_n_s32(0);
        for (int s = 2 * b; s < 2 * b + 2; ++s) {
            const std::int8_t* c = wc + static_cast<std::size_t>(s) * 16 * kNR;
            const int8x16_t w0 = vld1q_s8(c), w1 = vld1q_s8(c + 16), w2 = vld1q_s8(c + 32), w3 = vld1q_s8(c + 48);
            for (int r = 0; r < R; ++r) {
                const int8x16_t av = vld1q_s8(a + r * lda + 16 * s);
                acc[r] = dot_lane<0>(acc[r], w0, av);
                acc[r] = dot_lane<1>(acc[r], w1, av);
                acc[r] = dot_lane<2>(acc[r], w2, av);
                acc[r] = dot_lane<3>(acc[r], w3, av);
            }
        }
        for (int r = 0; r < R; ++r) {
            float32x4_t u = vmulq_f32(vcvtq_f32_s32(acc[r]), vdupq_n_f32(st[r * nb + b]));
            IMPBFF_FP4_KEEP(u);
            f[r] = vaddq_f32(f[r], u);
        }
    }
    for (int r = 0; r < R; ++r) vst1q_f32(out + r * kNR, f[r]);
}
#elif defined(IMPBFF_FP4_AVX512)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const float* st, const std::int32_t* sum,
                            const std::int8_t* wc, int nb, float* out) {
    const __m512i x80 = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512 f[R];
    for (int r = 0; r < R; ++r) f[r] = _mm512_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        __m512i acc[R];
        for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_si512();
        for (int s = 2 * b; s < 2 * b + 2; ++s) {
            const std::int8_t* c = wc + static_cast<std::size_t>(s) * 16 * kNR;
            __m512i w[4];  // w + 128 as u8
            for (int l = 0; l < 4; ++l) w[l] = _mm512_xor_si512(_mm512_loadu_si512(c + 64 * l), x80);
            for (int r = 0; r < R; ++r) {
                std::int32_t q[4];
                std::memcpy(q, a + r * lda + 16 * s, 16);
                for (int l = 0; l < 4; ++l) acc[r] = _mm512_dpbusd_epi32(acc[r], w[l], _mm512_set1_epi32(q[l]));
            }
        }
        for (int r = 0; r < R; ++r) {
            const __m512i v = _mm512_sub_epi32(acc[r], _mm512_set1_epi32(128 * sum[r * nb + b]));
            __m512 u = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(st[r * nb + b]));
            IMPBFF_FP4_KEEP(u);
            f[r] = _mm512_add_ps(f[r], u);
        }
    }
    for (int r = 0; r < R; ++r) _mm512_storeu_ps(out + r * kNR, f[r]);
}
#elif defined(IMPBFF_FP4_AVX2)
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const float* st, const std::int32_t*,
                            const std::int8_t* wc, int nb, float* out) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 f[R];
    for (int r = 0; r < R; ++r) f[r] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        __m256i acc[R];
        for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_si256();
        for (int s = 2 * b; s < 2 * b + 2; ++s) {
            const std::int8_t* c = wc + static_cast<std::size_t>(s) * 16 * kNR;
            __m256i w[4], aw[4];
            for (int l = 0; l < 4; ++l) {
                w[l] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 32 * l));
                aw[l] = _mm256_abs_epi8(w[l]);
            }
            for (int r = 0; r < R; ++r) {
                std::int32_t q[4];
                std::memcpy(q, a + r * lda + 16 * s, 16);
                for (int l = 0; l < 4; ++l) {
                    // |w| x sign(a, w) = w a; |.| <= 127 x 127, pairs <= 32258: no saturation
                    const __m256i p = _mm256_maddubs_epi16(aw[l], _mm256_sign_epi8(_mm256_set1_epi32(q[l]), w[l]));
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(p, ones));
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            __m256 u = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_set1_ps(st[r * nb + b]));
            IMPBFF_FP4_KEEP(u);
            f[r] = _mm256_add_ps(f[r], u);
        }
    }
    for (int r = 0; r < R; ++r) _mm256_storeu_ps(out + r * kNR, f[r]);
}
#else
template <int R>
IMPBFF_FP4_INLINE void tile(const std::int8_t* a, std::size_t lda, const float* st, const std::int32_t*,
                            const std::int8_t* wc, int nb, float* out) {
    float f[R][kNR] = {};
    for (int b = 0; b < nb; ++b) {
        std::int32_t acc[R][kNR] = {};
        for (int s = 2 * b; s < 2 * b + 2; ++s) {
            const std::int8_t* c = wc + static_cast<std::size_t>(s) * 16 * kNR;
            for (int r = 0; r < R; ++r) {
                const std::int8_t* ar = a + r * lda + 16 * s;
                for (int o = 0; o < kNR; ++o) {
                    std::int32_t d = 0;
                    for (int l = 0; l < 4; ++l)
                        for (int j = 0; j < 4; ++j) d += static_cast<std::int32_t>(c[4 * kNR * l + 4 * o + j]) * ar[4 * l + j];
                    acc[r][o] += d;
                }
            }
        }
        for (int r = 0; r < R; ++r)
            for (int o = 0; o < kNR; ++o) {
                float u = static_cast<float>(acc[r][o]) * st[r * nb + b];
                IMPBFF_FP4_KEEP(u);
                f[r][o] += u;
            }
    }
    for (int r = 0; r < R; ++r)
        for (int o = 0; o < kNR; ++o) out[r * kNR + o] = f[r][o];
}
#endif
}  // namespace mk8

//! `dW` (L.rows x P.rows, row-major, float64 of the float result) =
//! dequantised `L` times `P^T` (the codes of `Â^T`, unscaled: the
//! activation steps are in `L`).
inline void wgrad_gemm(const G8Blocks& L, const PackedI8& P, double* C) {
    const int M = L.rows, N = P.rows, nb = L.nb, nt = P.nt;
    if (M <= 0 || N <= 0) return;
    if (L.kp != P.kp) throw std::runtime_error("int8 wgrad: operand widths differ");
    const std::size_t lda = static_cast<std::size_t>(L.kp);
    constexpr int kMR = mk8::kMR8;
    const int nrb = (M + kMR - 1) / kMR;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (static_cast<long>(M) * N * nb > (1L << 14))
#endif
    for (int rb = 0; rb < nrb; ++rb) {
        float out[kMR * kNR];
        const int r0 = rb * kMR, mr = std::min(kMR, M - r0);
        const std::int8_t* a = L.q.data() + static_cast<std::size_t>(r0) * lda;
        const float* st = L.step.data() + static_cast<std::size_t>(r0) * nb;
        const std::int32_t* sm = L.sum.data() + static_cast<std::size_t>(r0) * nb;
        for (int t = 0; t < nt; ++t) {
            const std::int8_t* wc = P.codes.data() + static_cast<std::size_t>(t) * P.nsb * 16 * kNR;
            if (mr == kMR) {
                mk8::tile<kMR>(a, lda, st, sm, wc, nb, out);
            } else {
                for (int r = 0; r < mr; ++r)
                    mk8::tile<1>(a + r * lda, lda, st + r * nb, sm + r * nb, wc, nb, out + r * kNR);
            }
            const int o0 = t * kNR, no = std::min(kNR, N - o0);
            for (int r = 0; r < mr; ++r) {
                double* c = C + static_cast<std::size_t>(r0 + r) * N + o0;
                for (int o = 0; o < no; ++o) c[o] = static_cast<double>(out[r * kNR + o]);
            }
        }
    }
}

//! The same product element by element from fprop's codes (tests).
inline void wgrad_reference(const G8Blocks& L, const A8Rows& A, int n_in, double* C) {
    for (int o = 0; o < L.rows; ++o)
        for (int k = 0; k < n_in; ++k) {
            float f = 0.0f;
            for (int b = 0; b < L.nb; ++b) {
                std::int32_t s = 0;
                for (int r = b * kGradBlock; r < (b + 1) * kGradBlock && r < A.rows; ++r)
                    s += static_cast<std::int32_t>(L.q[static_cast<std::size_t>(o) * L.kp + r]) *
                         A.q[static_cast<std::size_t>(r) * A.kp + k];
                float u = static_cast<float>(s) * L.step[static_cast<std::size_t>(o) * L.nb + b];
                IMPBFF_FP4_KEEP(u);
                f += u;
            }
            C[static_cast<std::size_t>(o) * n_in + k] = static_cast<double>(f);
        }
}

//! `dW = dZ^T Â` of a ternary layer on the int8 kernels: `A` is fprop's
//! int8 input (codes and steps), `key` the draws; `T`, `L`, `P` scratch.
inline void wgrad(const double* dZ, int bs, int n_out, const A8Rows& A, int n_in, const SrKey& key,
                  std::vector<double>& T, G8Blocks& L, PackedI8& P, double* dW) {
    quantize_wgrad_left(dZ, bs, n_out, A.step.data(), key, T, L);
    pack_wgrad_right(A, n_in, L.kp, P);
    wgrad_gemm(L, P, dW);
}

}  // namespace mlpternary
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPTERNARYGRAD_H

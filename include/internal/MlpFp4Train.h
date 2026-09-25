/**
 *  \file IMP/bff/internal/MlpFp4Train.h
 *  \brief FP4 training of a dense network: NVIDIA's NVFP4 pretraining recipe
 *         (and its MXFP4 analogue) on the integer FP4 kernels.
 *
 * Follows "Pretraining Large Language Models with NVFP4" (NVIDIA,
 * arXiv 2509.25149), section 4 and appendix B, for a dense regressor:
 *
 * - **All three GEMMs of an FP4 layer run in FP4** (section 4.1: "All GEMM
 *   operations quantize their inputs to NVFP4"), on the codes through
 *   MlpFp4Kernels.h's micro-kernel (the left operand as the codes' values
 *   2 x E2M1, the right packed FP4) -- never dequantised to float:
 *   fprop `Z = Q(A) Q(W)^T`, dgrad `dA = Q(dZ) Q(W)`, wgrad
 *   `dW = Q(dZ)^T Q(A)`, each operand quantised along the GEMM's
 *   contraction dimension (1 x 16 blocks, appendix B; 1 x 32 for mxfp4).
 * - **Master weights and the Adam state stay float64**; the weights are
 *   re-quantised every step.
 * - **Weights use 2-D 16 x 16 block scaling** (section 4.3: "elements are
 *   grouped and scaled in 16x16 blocks"), so W in fprop and W^T in dgrad are
 *   the same quantised values (the chain rule then holds for the function
 *   actually evaluated). One quantisation a step, packed straight into both
 *   kernel operands (quantize_2d_packed; the reference is `quantize_2d` /
 *   `transpose_2d` in MlpFp4.h).
 * - **A random Hadamard transform on the wgrad inputs only** (section 4.2:
 *   "we restrict Hadamard transforms to Wgrad inputs"), d = 16, with "a
 *   single random sign vector that is shared across all linear layers
 *   throughout training": `T = H_16 diag(s) / 4`, applied to both operands
 *   along the contraction (batch) dimension in tiles of 16, so
 *   `(T dZ)^T (T A) = dZ^T A` before quantisation. The sign vector comes from
 *   a fixed seed (kHadamardSeed), not from the training seed.
 * - **Stochastic rounding for the gradients** (section 4.4: stochastic
 *   rounding "during quantization of high precision values to FP4" of
 *   gradient tensors; round-to-nearest-even for weights and activations):
 *   dZ is stochastically rounded in dgrad and in wgrad; W and A (also A in
 *   wgrad) round to nearest even. The draws are counter-based (SrKey in
 *   MlpFp4.h: 16 bits an element, keyed by seed, step, layer and operand),
 *   so training stays deterministic per seed and the draws vectorise. Block
 *   scales are always computed deterministically (RNE of the E4M3 scale).
 * - **Sensitive layers in higher precision** (section 4.1: keep "a few
 *   sensitive linear layers in higher precision ... with the majority of
 *   high precision layers at the end of the network"; the 12B model keeps
 *   the first two and last eight blocks, ~16 %). For a small MLP that is
 *   the first and the last layer: both are kept in float64 by default
 *   (`keep_first`, `keep_last`, each an option). The first layer matters
 *   more here than in a language model: its input is a handful of physical
 *   features, which FP4 would round to ~3 significant bits each.
 *
 * Where bff differs from the paper, deliberately: the NVFP4 global scale of
 * an activation or gradient operand is one float32 per operand *row* (the
 * non-contracted index) rather than one per tensor, so a sample's forward
 * pass does not depend on its batch and inference (QuantizedNeuralNet,
 * W4A4) reproduces training's fprop exactly; the weights keep one global
 * scale per tensor, as in the paper. Accumulation is float (kernels) and the
 * bias, activation and loss are double.
 *
 * MXFP4 training is the same recipe with the MX format: E8M0 power-of-two
 * block scales (OCP rule, MlpFp4.h), blocks of 32 along K, 2-D weight tiles
 * of 32 x 32, no global scale; the Hadamard tiles stay 16 wide (they only
 * need to be orthogonal). The paper reports MXFP4 needing ~36 % more tokens
 * than NVFP4 for the same loss (section 5); that is the expectation here too.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPFP4TRAIN_H
#define IMPBFF_INTERNAL_MLPFP4TRAIN_H

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpFp4.h>
#include <IMP/bff/internal/MlpFp4Kernels.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpfp4 {
namespace train {

#if defined(__clang__)
#define IMPBFF_FP4_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define IMPBFF_FP4_UNROLL _Pragma("GCC unroll 16")
#else
#define IMPBFF_FP4_UNROLL
#endif

//! The fixed seed of the Hadamard sign vector (shared by all layers).
constexpr std::uint64_t kHadamardSeed = 0x48616461'6d617264ULL;  // "Hadamard"

//! `T = H_16 diag(s) / 4`: orthogonal (T T^T = I), H the Sylvester Hadamard
//! matrix (`H_ij = (-1)^popcount(i & j)`), s random signs. Applied as a fast
//! Walsh-Hadamard transform: `x_j s_j / 4` (exact), then four butterfly
//! stages `(a, b) -> (a + b, a - b)` -- adds only, so every build rounds
//! alike. `t` is the matrix (tests).
struct Hadamard16 {
    double t[16][16];
    double sq[16];  //!< s_j / 4
    explicit Hadamard16(std::uint64_t seed = kHadamardSeed) {
        SplitMix64 rng(seed);
        double s[16];
        for (double& v : s) v = (rng.next() >> 63) ? -1.0 : 1.0;
        for (int j = 0; j < 16; ++j) sq[j] = 0.25 * s[j];
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                int bits = 0;  // popcount(i & j)
                for (unsigned v = static_cast<unsigned>(i & j); v; v >>= 1) bits += static_cast<int>(v & 1u);
                t[i][j] = ((bits & 1) ? -0.25 : 0.25) * s[j];
            }
    }
    //! The butterflies on 16 vectors of `W` lanes (`v[i * W + c]`).
    template <int W>
    static void butterflies(double* v) {
        for (int h = 1; h < 16; h <<= 1)
            for (int i = 0; i < 16; ++i)
                if (!(i & h))
                    for (int c = 0; c < W; ++c) {
                        const double a = v[i * W + c], b = v[(i + h) * W + c];
                        v[i * W + c] = a + b;
                        v[(i + h) * W + c] = a - b;
                    }
    }
    //! `y = T x` for one 16-vector.
    void apply(const double* x, double* y) const {
        for (int j = 0; j < 16; ++j) y[j] = x[j] * sq[j];
        butterflies<1>(y);
    }
};

//! Every row of `M` (`rows x K`), zero padded to K16 = 16 ceil(K / 16) and
//! transformed tile by tile: `out` is `rows x K16`.
inline void rht_rows(const double* M, int rows, int K, const Hadamard16& h, std::vector<double>& out,
                     int& K16) {
    K16 = (K + 15) / 16 * 16;
    out.assign(static_cast<std::size_t>(rows) * K16, 0.0);
    double x[16];
    for (int r = 0; r < rows; ++r)
        for (int k0 = 0; k0 < K16; k0 += 16) {
            for (int j = 0; j < 16; ++j) x[j] = k0 + j < K ? M[static_cast<std::size_t>(r) * K + k0 + j] : 0.0;
            h.apply(x, out.data() + static_cast<std::size_t>(r) * K16 + k0);
        }
}

//! The transpose of `M` (`rows x n`), each of its `n` columns Hadamard-
//! transformed tile by tile along `rows` (zero padded to K16): `out` is
//! `n x K16`, bit for bit rht_rows() of the explicit transpose (the same
//! exact products and butterflies, vectorised over 8 columns at a time).
inline void rht_transpose(const double* M, int rows, int n, const Hadamard16& h, std::vector<double>& out, int& K16,
                          std::vector<double>& tile) {
    constexpr int kC = 8;
    K16 = (rows + 15) / 16 * 16;
    out.resize(static_cast<std::size_t>(n) * K16);
    tile.resize(16 * kC);
    double* v = tile.data();
    for (int r0 = 0; r0 < K16; r0 += 16) {
        const int nj = std::min(16, rows - r0);
        int c0 = 0;
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
        for (; c0 + 2 <= n; c0 += 2) {  // two columns, 16 registers
            float64x2_t w[16];
            IMPBFF_FP4_UNROLL
            for (int j = 0; j < 16; ++j)
                w[j] = j < nj ? vmulq_n_f64(vld1q_f64(M + static_cast<std::size_t>(r0 + j) * n + c0), h.sq[j])
                              : vdupq_n_f64(0.0);
            IMPBFF_FP4_UNROLL
            for (int hh = 1; hh < 16; hh <<= 1)
                IMPBFF_FP4_UNROLL
                for (int i = 0; i < 16; ++i)
                    if (!(i & hh)) {
                        const float64x2_t a = w[i], b = w[i + hh];
                        w[i] = vaddq_f64(a, b);
                        w[i + hh] = vsubq_f64(a, b);
                    }
            double* o0 = out.data() + static_cast<std::size_t>(c0) * K16 + r0;
            double* o1 = o0 + K16;
            IMPBFF_FP4_UNROLL
            for (int i = 0; i < 16; i += 2) {
                vst1q_f64(o0 + i, vzip1q_f64(w[i], w[i + 1]));
                vst1q_f64(o1 + i, vzip2q_f64(w[i], w[i + 1]));
            }
        }
#elif defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
        for (; c0 + 4 <= n; c0 += 4) {  // four columns
            __m256d w[16];
            IMPBFF_FP4_UNROLL
            for (int j = 0; j < 16; ++j)
                w[j] = j < nj ? _mm256_mul_pd(_mm256_loadu_pd(M + static_cast<std::size_t>(r0 + j) * n + c0),
                                              _mm256_set1_pd(h.sq[j]))
                              : _mm256_setzero_pd();
            IMPBFF_FP4_UNROLL
            for (int hh = 1; hh < 16; hh <<= 1)
                IMPBFF_FP4_UNROLL
                for (int i = 0; i < 16; ++i)
                    if (!(i & hh)) {
                        const __m256d a = w[i], b = w[i + hh];
                        w[i] = _mm256_add_pd(a, b);
                        w[i + hh] = _mm256_sub_pd(a, b);
                    }
            double* o = out.data() + static_cast<std::size_t>(c0) * K16 + r0;
            IMPBFF_FP4_UNROLL
            for (int i = 0; i < 16; i += 4) {  // 4 x 4 transposes
                const __m256d t0 = _mm256_unpacklo_pd(w[i], w[i + 1]), t1 = _mm256_unpackhi_pd(w[i], w[i + 1]);
                const __m256d t2 = _mm256_unpacklo_pd(w[i + 2], w[i + 3]), t3 = _mm256_unpackhi_pd(w[i + 2], w[i + 3]);
                _mm256_storeu_pd(o + i, _mm256_permute2f128_pd(t0, t2, 0x20));
                _mm256_storeu_pd(o + K16 + i, _mm256_permute2f128_pd(t1, t3, 0x20));
                _mm256_storeu_pd(o + 2 * K16 + i, _mm256_permute2f128_pd(t0, t2, 0x31));
                _mm256_storeu_pd(o + 3 * K16 + i, _mm256_permute2f128_pd(t1, t3, 0x31));
            }
        }
#endif
        for (; c0 < n; c0 += kC) {
            const int nc = std::min(kC, n - c0);
            for (int j = 0; j < 16; ++j) {
                const double* x = M + static_cast<std::size_t>(r0 + j) * n + c0;
                if (j < nj && nc == kC)
                    for (int c = 0; c < kC; ++c) v[j * kC + c] = x[c] * h.sq[j];
                else
                    for (int c = 0; c < kC; ++c) v[j * kC + c] = (j < nj && c < nc) ? x[c] * h.sq[j] : 0.0;
            }
            Hadamard16::butterflies<kC>(v);
            for (int c = 0; c < nc; ++c) {
                double* o = out.data() + static_cast<std::size_t>(c0 + c) * K16 + r0;
                for (int i = 0; i < 16; ++i) o[i] = v[i * kC + c];
            }
        }
    }
}

//! The plain transpose of `M` (`rows x n`) into `out` (`n x rows`).
inline void transpose(const double* M, int rows, int n, std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(n) * rows);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < n; ++c) out[static_cast<std::size_t>(c) * rows + r] = M[static_cast<std::size_t>(r) * n + c];
}

//! The recipe's switches.
struct Config {
    Format format = Format::NVFP4;
    bool hadamard = true;     //!< RHT on the wgrad inputs
    bool stochastic = true;   //!< stochastic rounding of the gradients
    bool keep_first = true;   //!< first layer in float64
    bool keep_last = true;    //!< last layer in float64
    bool fp4_layer(std::size_t l, std::size_t n_layers) const {
        return !((keep_first && l == 0) || (keep_last && l + 1 == n_layers));
    }
};

//! Per-step state: activations, pre-activations, and each FP4 layer's
//! quantised weights (fprop) and their exact transpose (dgrad).
struct Workspace {
    std::vector<std::vector<double>> a;  //!< a[0] input .. a[L] output, bs x width
    std::vector<std::vector<double>> z;  //!< z[l] = pre-activation of layer l
    //! Each FP4 layer's 2-D quantised weights packed for the kernel, as W
    //! (fprop) and as its exact transpose W^T (dgrad): one quantisation.
    std::vector<kern::PackedRight> pw, pwt;
    std::vector<double> dz, da, f1, dzt, at, tile;
    std::vector<std::uint8_t> wcodes, tcodes, tilecode;
    std::vector<float> hsc, hrow;
    kern::Q8Rows left;
    kern::PackedRight right;
    kern::QuantScratch qs;
    const std::vector<double>& output() const { return a.back(); }
};

//! quantize_2d() (MlpFp4.h, the reference) of `W` (`rows x cols`), packed
//! straight into the kernel's W and W^T operands: the same tile scale codes
//! and element codes, the vectorised encoder.
inline void quantize_2d_packed(const double* W, int rows, int cols, Format f, Workspace& ws,
                               kern::PackedRight& pw, kern::PackedRight& pwt) {
    const int b = block_size(f, cols), kpi = padded_cols(cols), kpo = padded_cols(rows);
    const int nb = kpi / b, nrt = (rows + b - 1) / b;
    const float g = f == Format::NVFP4
                            ? nvfp4_tensor_scale(kern::vq::absmax(W, rows * cols))
                            : 1.0f;
    // tile scale codes
    ws.tilecode.assign(static_cast<std::size_t>(nrt) * nb, 0);
    for (int rt = 0; rt < nrt; ++rt)
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = std::min(cols, kb * b), k1 = std::min(cols, k0 + b);
            double amax = 0.0;
            for (int r = rt * b; r < std::min(rows, rt * b + b); ++r)
                if (k1 > k0) amax = std::max(amax, kern::vq::absmax(W + static_cast<std::size_t>(r) * cols + k0, k1 - k0));
            ws.tilecode[static_cast<std::size_t>(rt) * nb + kb] = block_scale_code(f, amax, g);
        }
    // every tile's half scale (kd::half_scale), and the zero-block one past the last row tile
    ws.hsc.resize(static_cast<std::size_t>(nrt) * nb + 1);
    float* th = ws.hsc.data();
    for (std::size_t t = 0; t < static_cast<std::size_t>(nrt) * nb; ++t) th[t] = kern::kd::half_scale(f, ws.tilecode[t], g);
    th[static_cast<std::size_t>(nrt) * nb] = kern::kd::half_scale(f, block_scale_code(f, 0.0, g), g);
    std::vector<float>& hrow = ws.hrow;
    hrow.resize(static_cast<std::size_t>(std::max(kpi, kpo) / 16));
    // element codes, row by row (rows x kpi, unpacked)
    ws.wcodes.resize(static_cast<std::size_t>(rows) * kpi);
    pw.rows = 0;
    kern::pack_right_init(rows, kpi, pw);
    for (int r = 0; r < rows; ++r) {
        std::uint8_t* cr = ws.wcodes.data() + static_cast<std::size_t>(r) * kpi;
        std::fill(cr + cols, cr + kpi, std::uint8_t(0));
        const double* x = W + static_cast<std::size_t>(r) * cols;
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = std::min(cols, kb * b), k1 = std::min(cols, k0 + b);
            const std::uint8_t code = ws.tilecode[static_cast<std::size_t>(r / b) * nb + kb];
            const double d = block_scale_value(f, code, g);
            if (k1 <= k0) continue;
            if (!(d > 0.0)) {
                std::fill(cr + k0, cr + k1, std::uint8_t(0));
                continue;
            }
            int k = k0;
            const bool p2 = kern::vq::pow2_scale(d);
            for (; k + 16 <= k1; k += 16) kern::tq::group_exact(x + k, d, p2, cr + k);
            if (k < k1) kern::vq::encode_rne(x + k, k1 - k, d, cr + k);
        }
        const float* tr = th + static_cast<std::size_t>(r / b) * nb;
        for (int s2 = 0; s2 < kpi / 16; ++s2) hrow[static_cast<std::size_t>(s2)] = tr[s2 * 16 / b];
        kern::pack_right_row(pw, r, cr, hrow.data());
    }
    // the transpose: row k holds column k's codes; sub-block s's scale is
    // the tile (s * 16 / b, k / b), a zero-block code past the last row tile
    kern::pack_right_init(cols, kpo, pwt);
    ws.tcodes.assign(static_cast<std::size_t>(kpo), 0);
    for (int k = 0; k < cols; ++k) {
        const std::uint8_t* src = ws.wcodes.data() + k;
        for (int r = 0; r < rows; ++r) ws.tcodes[static_cast<std::size_t>(r)] = src[static_cast<std::size_t>(r) * kpi];
        for (int s2 = 0; s2 < kpo / 16; ++s2) {
            const int rt = s2 * 16 / b;
            hrow[static_cast<std::size_t>(s2)] =
                    rt * b < rows ? th[static_cast<std::size_t>(rt) * nb + k / b] : th[static_cast<std::size_t>(nrt) * nb];
        }
        kern::pack_right_row(pwt, k, ws.tcodes.data(), hrow.data());
    }
}

//! Quantised weights of every FP4 layer (2-D tiles), packed as W and W^T.
inline void quantize_weights(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws) {
    ws.pw.resize(layers.size());
    ws.pwt.resize(layers.size());
    for (std::size_t l = 0; l < layers.size(); ++l) {
        if (!c.fp4_layer(l, layers.size())) continue;
        if (c.format == Format::FP4) throw std::runtime_error("FP4: 2-D scaling needs mxfp4 or nvfp4");
        quantize_2d_packed(layers[l].weight.data(), layers[l].n_out, layers[l].n_in, c.format, ws, ws.pw[l], ws.pwt[l]);
    }
}

//! fprop of a batch (`X` is `bs x n_in`, standardised units): FP4 layers
//! `Z = Q(A) Q(W)^T + b` on the kernels, float64 layers through `Gemm`.
//! quantize_weights() must have run for the current weights.
template <class Gemm = mlpcore::PortableGemm>
inline void forward(const std::vector<DenseLayer>& layers, const Config& c, const double* X, int bs,
                    Workspace& ws) {
    const std::size_t L = layers.size();
    ws.a.resize(L + 1);
    ws.z.resize(L);
    ws.a[0].assign(X, X + static_cast<std::size_t>(bs) * layers.front().n_in);
    for (std::size_t l = 0; l < L; ++l) {
        const DenseLayer& ly = layers[l];
        std::vector<double>& z = ws.z[l];
        z.resize(static_cast<std::size_t>(bs) * ly.n_out);
        if (c.fp4_layer(l, L)) {
            kern::quantize_left(ws.a[l].data(), bs, ly.n_in, c.format, Rounding::NearestEven, nullptr, ws.left, ws.qs);
            kern::gemm_packed(ws.left, ws.pw[l], z.data(), ly.bias.data());
        } else {
            Gemm::nt(bs, ly.n_out, ly.n_in, ws.a[l].data(), ly.weight.data(), z.data());
            for (int r = 0; r < bs; ++r)
                for (int o = 0; o < ly.n_out; ++o)
                    z[static_cast<std::size_t>(r) * ly.n_out + o] += ly.bias[static_cast<std::size_t>(o)];
        }
        ws.a[l + 1].resize(z.size());
        mlpcore::act_apply(z.data(), ws.a[l + 1].data(), z.size(), ly.activation);
    }
}

//! The stochastic-rounding stream of a training run: SrKey (MlpFp4.h) of
//! (seed, step, layer, operand); backward() advances `step` once a call.
struct SrStream {
    std::uint64_t seed = 0;
    std::uint64_t step = 0;
    explicit SrStream(std::uint64_t s = 0) : seed(s) {}
    //! operand 0: dgrad's dZ, 1: wgrad's (T dZ)^T.
    SrKey key(int layer, int operand) const { return SrKey::make(seed, step, layer, operand); }
};

//! The operands wgrad multiplied, decoded (tests): `x` = Q(T dZ^T) as
//! `n_out x kp`, `y` = Q(T A^T) as `n_in x kp`.
struct WgradOperands {
    int kp = 0;
    std::vector<double> x, y;
};

//! The dequantised values of a left operand, `rows x kp`.
inline std::vector<double> decode_left(const kern::Q8Rows& L) {
    std::vector<double> out(static_cast<std::size_t>(L.rows) * L.kp);
    for (int r = 0; r < L.rows; ++r)
        for (int k = 0; k < L.kp; ++k)
            out[static_cast<std::size_t>(r) * L.kp + k] =
                    static_cast<double>(L.q[static_cast<std::size_t>(r) * L.kp + k]) *
                    static_cast<double>(L.sc[static_cast<std::size_t>(r) * L.n_sub() + k / 16]);
    return out;
}

//! wgrad of an FP4 layer: `dW (n_out x n_in) = dZ^T A` over the batch, both
//! operands (Hadamard-transformed along the batch when `c.hadamard`)
//! quantised along the batch with training's element rule
//! (kern::quantize_train); dZ stochastically rounded with `key` when
//! `c.stochastic`.
inline void wgrad(const double* dZ, const double* A, int bs, int n_out, int n_in, const Config& c, const SrKey& key,
                  const Hadamard16& h, double* dW, Workspace& ws, WgradOperands* keep = nullptr) {
    int K = bs;
    if (c.hadamard) {
        rht_transpose(dZ, bs, n_out, h, ws.dzt, K, ws.tile);
        rht_transpose(A, bs, n_in, h, ws.at, K, ws.tile);
    } else {
        transpose(dZ, bs, n_out, ws.dzt);
        transpose(A, bs, n_in, ws.at);
    }
    const SrKey* k = c.stochastic ? &key : nullptr;
    kern::quantize_train(ws.dzt.data(), n_out, K, static_cast<std::size_t>(K), c.format, k, &ws.left, nullptr, ws.qs);
    kern::quantize_train(ws.at.data(), n_in, K, static_cast<std::size_t>(K), c.format, nullptr, nullptr, &ws.right,
                         ws.qs);
    if (keep != nullptr) {
        keep->kp = ws.left.kp;
        keep->x = decode_left(ws.left);
        kern::Q8Rows y;
        kern::quantize_train(ws.at.data(), n_in, K, static_cast<std::size_t>(K), c.format, nullptr, &y, nullptr, ws.qs);
        keep->y = decode_left(y);
    }
    kern::gemm_packed(ws.left, ws.right, dW);
}

//! wgrad() with its own scratch (tests).
inline void wgrad(const double* dZ, const double* A, int bs, int n_out, int n_in, const Config& c, const SrKey& key,
                  const Hadamard16& h, double* dW, WgradOperands* keep = nullptr) {
    Workspace ws;
    wgrad(dZ, A, bs, n_out, n_in, c, key, h, dW, ws, keep);
}

//! Backward pass for `dY` (`bs x n_out`, the loss' adjoint of the output):
//! `grad` (flatten() layout) is overwritten.
template <class Gemm = mlpcore::PortableGemm>
inline void backward(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws, const double* dY,
                     int bs, SrStream& sr, const Hadamard16& h, double* grad) {
    const std::size_t L = layers.size();
    // offsets of each layer's weights in the flat vector
    std::vector<std::size_t> off(L);
    std::size_t p = 0;
    for (std::size_t l = 0; l < L; ++l) {
        off[l] = p;
        p += layers[l].weight.size() + layers[l].bias.size();
    }
    ws.da.assign(dY, dY + static_cast<std::size_t>(bs) * layers.back().n_out);
    for (std::size_t li = L; li-- > 0;) {
        const DenseLayer& ly = layers[li];
        const std::size_t nz = static_cast<std::size_t>(bs) * ly.n_out;
        ws.f1.resize(nz);
        mlpcore::act_derivs_n(ws.z[li].data(), ws.a[li + 1].data(), nz, ly.activation, 0, ws.f1.data(), nullptr,
                              nullptr);
        ws.dz.resize(nz);
        for (std::size_t i = 0; i < nz; ++i) ws.dz[i] = ws.da[i] * ws.f1[i];
        double* gW = grad + off[li];
        double* gb = gW + ly.weight.size();
        for (int o = 0; o < ly.n_out; ++o) gb[o] = 0.0;
        for (int r = 0; r < bs; ++r)
            for (int o = 0; o < ly.n_out; ++o) gb[o] += ws.dz[static_cast<std::size_t>(r) * ly.n_out + o];
        const bool fp4 = c.fp4_layer(li, L);
        if (fp4) {
            wgrad(ws.dz.data(), ws.a[li].data(), bs, ly.n_out, ly.n_in, c, sr.key(static_cast<int>(li), 1), h, gW, ws);
        } else {
            Gemm::tn(ly.n_out, ly.n_in, bs, ws.dz.data(), ws.a[li].data(), gW);
        }
        if (li == 0) break;
        ws.da.resize(static_cast<std::size_t>(bs) * ly.n_in);
        if (fp4) {
            const SrKey key = sr.key(static_cast<int>(li), 0);
            kern::quantize_train(ws.dz.data(), bs, ly.n_out, static_cast<std::size_t>(ly.n_out), c.format,
                                 c.stochastic ? &key : nullptr, &ws.left, nullptr, ws.qs);
            kern::gemm_packed(ws.left, ws.pwt[li], ws.da.data());
        } else {
            Gemm::nn(bs, ly.n_in, ly.n_out, ws.dz.data(), ly.weight.data(), ws.da.data());
        }
    }
    ++sr.step;
}

//! The inference network of an FP4-trained model: FP4 layers with exactly
//! fprop's 2-D scaled weights and W4A4 activations, the kept layers in
//! float64. QuantizedNeuralNet evaluates it as training's forward pass did.
inline Fp4Model to_model(const MlpModel& m, const Config& c) {
    Fp4Model out;
    out.format = c.format;
    out.quantize_activations = true;
    out.x_scaler = m.x_scaler;
    out.y_scaler = m.y_scaler;
    for (std::size_t l = 0; l < m.layers.size(); ++l) {
        const DenseLayer& d = m.layers[l];
        Fp4Layer q;
        q.n_in = d.n_in;
        q.n_out = d.n_out;
        q.activation = d.activation;
        q.bias = d.bias;
        if (c.fp4_layer(l, m.layers.size())) {
            q.weight = quantize_2d(d.weight.data(), d.n_out, d.n_in, c.format);
        } else {
            q.full_precision = true;
            q.weight_f64 = d.weight;
        }
        out.layers.push_back(std::move(q));
    }
    return out;
}

}  // namespace train
}  // namespace mlpfp4
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPFP4TRAIN_H

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
 *   wgrad) round to nearest even. The draws come from a SplitMix64 seeded by
 *   the training seed, so training stays deterministic per seed. Block
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

//! The fixed seed of the Hadamard sign vector (shared by all layers).
constexpr std::uint64_t kHadamardSeed = 0x48616461'6d617264ULL;  // "Hadamard"

//! `T = H_16 diag(s) / 4`: orthogonal (T T^T = I), H the Sylvester Hadamard
//! matrix (`H_ij = (-1)^popcount(i & j)`), s random signs.
struct Hadamard16 {
    double t[16][16];
    explicit Hadamard16(std::uint64_t seed = kHadamardSeed) {
        SplitMix64 rng(seed);
        double s[16];
        for (double& v : s) v = (rng.next() >> 63) ? -1.0 : 1.0;
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                int bits = 0;  // popcount(i & j)
                for (unsigned v = static_cast<unsigned>(i & j); v; v >>= 1) bits += static_cast<int>(v & 1u);
                t[i][j] = ((bits & 1) ? -0.25 : 0.25) * s[j];
            }
    }
    //! `y = T x` for one 16-vector.
    void apply(const double* x, double* y) const {
        for (int i = 0; i < 16; ++i) {
            double acc = 0.0;
            for (int j = 0; j < 16; ++j) acc += t[i][j] * x[j];
            y[i] = acc;
        }
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
//! `n x K16`, the same values as rht_rows() of the explicit transpose (the
//! sum over j runs in the same order; the zero padding adds nothing).
//! Vectorised over the columns.
inline void rht_transpose(const double* M, int rows, int n, const Hadamard16& h, std::vector<double>& out, int& K16,
                          std::vector<double>& tile) {
    K16 = (rows + 15) / 16 * 16;
    out.resize(static_cast<std::size_t>(n) * K16);
    tile.resize(static_cast<std::size_t>(16) * n);
    (void)tile;
#if defined(IMPBFF_FP4_AVX512)
    constexpr int kC = 8;  // columns a register block: 8 rows of T x kC accumulators
#else
    constexpr int kC = 4;
#endif
    (void)kC;
    for (int r0 = 0; r0 < K16; r0 += 16) {
        const int nj = std::min(16, rows - r0);
        int c0 = 0;
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON) || defined(IMPBFF_FP4_AVX2) || defined(IMPBFF_FP4_AVX512)
        for (; c0 + kC <= n; c0 += kC)
            for (int i0 = 0; i0 < 16; i0 += 8) {
#if defined(IMPBFF_FP4_NEON_DOTPROD) || defined(IMPBFF_FP4_NEON)
                float64x2_t acc[8][2];
                for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = vdupq_n_f64(0.0);
                for (int j = 0; j < nj; ++j) {
                    const double* x = M + static_cast<std::size_t>(r0 + j) * n + c0;
                    const float64x2_t x0 = vld1q_f64(x), x1 = vld1q_f64(x + 2);
                    // t x is exact (t = +-1/4), so the fused multiply-add rounds as
                    // acc + t x does (unless t x is subnormal)
                    for (int i = 0; i < 8; ++i) {
                        const double t = h.t[i0 + i][j];
                        acc[i][0] = vfmaq_n_f64(acc[i][0], x0, t);
                        acc[i][1] = vfmaq_n_f64(acc[i][1], x1, t);
                    }
                }
                double v[8][4];
                for (int i = 0; i < 8; ++i) {
                    vst1q_f64(v[i], acc[i][0]);
                    vst1q_f64(v[i] + 2, acc[i][1]);
                }
#elif defined(IMPBFF_FP4_AVX512)
                __m512d acc[8];
                for (int i = 0; i < 8; ++i) acc[i] = _mm512_setzero_pd();
                for (int j = 0; j < nj; ++j) {
                    const __m512d x0 = _mm512_loadu_pd(M + static_cast<std::size_t>(r0 + j) * n + c0);
                    for (int i = 0; i < 8; ++i)
                        acc[i] = _mm512_fmadd_pd(x0, _mm512_set1_pd(h.t[i0 + i][j]), acc[i]);
                }
                double v[8][8];
                for (int i = 0; i < 8; ++i) _mm512_storeu_pd(v[i], acc[i]);
#else
                __m256d acc[8];
                for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_pd();
                for (int j = 0; j < nj; ++j) {
                    const __m256d x0 = _mm256_loadu_pd(M + static_cast<std::size_t>(r0 + j) * n + c0);
                    for (int i = 0; i < 8; ++i)
#if defined(__FMA__)
                        acc[i] = _mm256_fmadd_pd(x0, _mm256_set1_pd(h.t[i0 + i][j]), acc[i]);
#else
                        acc[i] = _mm256_add_pd(acc[i], _mm256_mul_pd(x0, _mm256_set1_pd(h.t[i0 + i][j])));
#endif
                }
                double v[8][4];
                for (int i = 0; i < 8; ++i) _mm256_storeu_pd(v[i], acc[i]);
#endif
                for (int c = 0; c < kC; ++c) {
                    double* o = out.data() + static_cast<std::size_t>(c0 + c) * K16 + r0 + i0;
                    for (int i = 0; i < 8; ++i) o[i] = v[i][c];
                }
            }
#endif
        for (; c0 < n; ++c0) {
            double* o = out.data() + static_cast<std::size_t>(c0) * K16 + r0;
            for (int i = 0; i < 16; ++i) {
                double acc = 0.0;
                for (int j = 0; j < nj; ++j) acc += h.t[i][j] * M[static_cast<std::size_t>(r0 + j) * n + c0];
                o[i] = acc;
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
    std::vector<float> hsc;
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
    // element codes, row by row (rows x kpi, unpacked)
    ws.wcodes.assign(static_cast<std::size_t>(rows) * kpi, 0);
    ws.hsc.resize(static_cast<std::size_t>(std::max(kpi, kpo) / 16));
    pw.rows = 0;
    kern::pack_right_init(rows, kpi, pw);
    for (int r = 0; r < rows; ++r) {
        std::uint8_t* cr = ws.wcodes.data() + static_cast<std::size_t>(r) * kpi;
        const double* x = W + static_cast<std::size_t>(r) * cols;
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = std::min(cols, kb * b), k1 = std::min(cols, k0 + b);
            const std::uint8_t code = ws.tilecode[static_cast<std::size_t>(r / b) * nb + kb];
            const double d = block_scale_value(f, code, g);
            if (k1 > k0 && d > 0.0) kern::vq::encode_rne(x + k0, k1 - k0, d, cr + k0);
        }
        for (int s = 0; s < kpi / 16; ++s)
            ws.hsc[static_cast<std::size_t>(s)] =
                    kern::kd::half_scale(f, ws.tilecode[static_cast<std::size_t>(r / b) * nb + s * 16 / b], g);
        kern::pack_right_row(pw, r, cr, ws.hsc.data());
    }
    // the transpose: row k holds column k's codes; sub-block s's scale is
    // the tile (s * 16 / b, k / b), a zero-block code past the last row tile
    kern::pack_right_init(cols, kpo, pwt);
    ws.tcodes.assign(static_cast<std::size_t>(kpo), 0);
    const std::uint8_t zero_code = block_scale_code(f, 0.0, g);
    for (int k = 0; k < cols; ++k) {
        for (int r = 0; r < rows; ++r) ws.tcodes[static_cast<std::size_t>(r)] = ws.wcodes[static_cast<std::size_t>(r) * kpi + k];
        for (int s = 0; s < kpo / 16; ++s) {
            const int rt = s * 16 / b;
            ws.hsc[static_cast<std::size_t>(s)] = kern::kd::half_scale(
                    f, rt * b < rows ? ws.tilecode[static_cast<std::size_t>(rt) * nb + k / b] : zero_code, g);
        }
        kern::pack_right_row(pwt, k, ws.tcodes.data(), ws.hsc.data());
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

//! The operands wgrad multiplied, for tests: `x` = Q((T dZ^T)) rows, `y` =
//! Q((T A^T)) rows.
struct WgradOperands {
    kern::Fp4Rows x, y;
};

//! wgrad of an FP4 layer: `dW (n_out x n_in) = dZ^T A` over the batch, both
//! operands (Hadamard-transformed along the batch when `c.hadamard`)
//! quantised along the batch; dZ stochastically rounded when
//! `c.stochastic`.
inline void wgrad(const double* dZ, const double* A, int bs, int n_out, int n_in, const Config& c,
                  SplitMix64& rng, const Hadamard16& h, double* dW, Workspace& ws, WgradOperands* keep = nullptr) {
    int K = bs;
    if (c.hadamard) {
        rht_transpose(dZ, bs, n_out, h, ws.dzt, K, ws.tile);
        rht_transpose(A, bs, n_in, h, ws.at, K, ws.tile);
    } else {
        transpose(dZ, bs, n_out, ws.dzt);
        transpose(A, bs, n_in, ws.at);
    }
    const Rounding rz = c.stochastic ? Rounding::Stochastic : Rounding::NearestEven;
    if (keep != nullptr) {  // the same draws: a copy of the stream
        SplitMix64 r2 = rng;
        keep->x = kern::quantize_rows(ws.dzt.data(), n_out, K, c.format, rz, &r2);
        keep->y = kern::quantize_rows(ws.at.data(), n_in, K, c.format);
    }
    kern::quantize_left(ws.dzt.data(), n_out, K, c.format, rz, &rng, ws.left, ws.qs);
    kern::quantize_right(ws.at.data(), n_in, K, c.format, Rounding::NearestEven, nullptr, ws.right, ws.qs);
    kern::gemm_packed(ws.left, ws.right, dW);
}

//! wgrad() with its own scratch (tests).
inline void wgrad(const double* dZ, const double* A, int bs, int n_out, int n_in, const Config& c,
                  SplitMix64& rng, const Hadamard16& h, double* dW, WgradOperands* keep = nullptr) {
    Workspace ws;
    wgrad(dZ, A, bs, n_out, n_in, c, rng, h, dW, ws, keep);
}

//! Backward pass for `dY` (`bs x n_out`, the loss' adjoint of the output):
//! `grad` (flatten() layout) is overwritten.
template <class Gemm = mlpcore::PortableGemm>
inline void backward(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws, const double* dY,
                     int bs, SplitMix64& rng, const Hadamard16& h, double* grad) {
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
            wgrad(ws.dz.data(), ws.a[li].data(), bs, ly.n_out, ly.n_in, c, rng, h, gW, ws);
        } else {
            Gemm::tn(ly.n_out, ly.n_in, bs, ws.dz.data(), ws.a[li].data(), gW);
        }
        if (li == 0) break;
        ws.da.resize(static_cast<std::size_t>(bs) * ly.n_in);
        if (fp4) {
            kern::quantize_left(ws.dz.data(), bs, ly.n_out, c.format,
                                c.stochastic ? Rounding::Stochastic : Rounding::NearestEven, &rng, ws.left, ws.qs);
            kern::gemm_packed(ws.left, ws.pwt[li], ws.da.data());
        } else {
            Gemm::nn(bs, ly.n_in, ly.n_out, ws.dz.data(), ly.weight.data(), ws.da.data());
        }
    }
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

/**
 *  \file IMP/bff/internal/MlpFp4Train.h
 *  \brief FP4 training of a dense network: NVIDIA's NVFP4 pretraining recipe
 *         (and its MXFP4 analogue) on the integer FP4 kernels.
 *
 * Follows "Pretraining Large Language Models with NVFP4" (NVIDIA,
 * arXiv 2509.25149), section 4 and appendix B, for a dense regressor:
 *
 * - **All three GEMMs of an FP4 layer run in FP4** (section 4.1: "All GEMM
 *   operations quantize their inputs to NVFP4"), on the packed codes through
 *   MlpFp4Kernels.h's FP4 x FP4 kernel -- never dequantised to float:
 *   fprop `Z = Q(A) Q(W)^T`, dgrad `dA = Q(dZ) Q(W)`, wgrad
 *   `dW = Q(dZ)^T Q(A)`, each operand quantised along the GEMM's
 *   contraction dimension (1 x 16 blocks, appendix B; 1 x 32 for mxfp4).
 * - **Master weights and the Adam state stay float64**; the weights are
 *   re-quantised every step.
 * - **Weights use 2-D 16 x 16 block scaling** (section 4.3: "elements are
 *   grouped and scaled in 16x16 blocks"), so W in fprop and W^T in dgrad are
 *   the same quantised values (the chain rule then holds for the function
 *   actually evaluated). `quantize_2d` / `transpose_2d` in MlpFp4.h.
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
    std::vector<Fp4Tensor> wq, wqt;
    std::vector<double> dz, da, f1, dw;
    const std::vector<double>& output() const { return a.back(); }
};

//! Quantised weights of every FP4 layer (2-D tiles) and their transposes.
inline void quantize_weights(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws) {
    ws.wq.assign(layers.size(), Fp4Tensor());
    ws.wqt.assign(layers.size(), Fp4Tensor());
    for (std::size_t l = 0; l < layers.size(); ++l) {
        if (!c.fp4_layer(l, layers.size())) continue;
        ws.wq[l] = quantize_2d(layers[l].weight.data(), layers[l].n_out, layers[l].n_in, c.format);
        ws.wqt[l] = transpose_2d(ws.wq[l]);
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
            kern::gemm_fp4(kern::quantize_rows(ws.a[l].data(), bs, ly.n_in, c.format), kern::rows_of(ws.wq[l]),
                           z.data());
        } else {
            Gemm::nt(bs, ly.n_out, ly.n_in, ws.a[l].data(), ly.weight.data(), z.data());
        }
        for (int r = 0; r < bs; ++r)
            for (int o = 0; o < ly.n_out; ++o) z[static_cast<std::size_t>(r) * ly.n_out + o] += ly.bias[static_cast<std::size_t>(o)];
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
                  SplitMix64& rng, const Hadamard16& h, double* dW, WgradOperands* keep = nullptr) {
    std::vector<double> dzt(static_cast<std::size_t>(n_out) * bs), at(static_cast<std::size_t>(n_in) * bs);
    for (int r = 0; r < bs; ++r) {
        for (int o = 0; o < n_out; ++o) dzt[static_cast<std::size_t>(o) * bs + r] = dZ[static_cast<std::size_t>(r) * n_out + o];
        for (int k = 0; k < n_in; ++k) at[static_cast<std::size_t>(k) * bs + r] = A[static_cast<std::size_t>(r) * n_in + k];
    }
    int K = bs;
    if (c.hadamard) {
        std::vector<double> t;
        rht_rows(dzt.data(), n_out, bs, h, t, K);
        dzt.swap(t);
        rht_rows(at.data(), n_in, bs, h, t, K);
        at.swap(t);
    }
    WgradOperands ops;
    ops.x = kern::quantize_rows(dzt.data(), n_out, K, c.format,
                                c.stochastic ? Rounding::Stochastic : Rounding::NearestEven, &rng);
    ops.y = kern::quantize_rows(at.data(), n_in, K, c.format);
    kern::gemm_fp4(ops.x, ops.y, dW);
    if (keep != nullptr) *keep = std::move(ops);
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
            wgrad(ws.dz.data(), ws.a[li].data(), bs, ly.n_out, ly.n_in, c, rng, h, gW);
        } else {
            Gemm::tn(ly.n_out, ly.n_in, bs, ws.dz.data(), ws.a[li].data(), gW);
        }
        if (li == 0) break;
        ws.da.resize(static_cast<std::size_t>(bs) * ly.n_in);
        if (fp4) {
            kern::gemm_fp4(kern::quantize_rows(ws.dz.data(), bs, ly.n_out, c.format,
                                               c.stochastic ? Rounding::Stochastic : Rounding::NearestEven, &rng),
                           kern::rows_of(ws.wqt[li]), ws.da.data());
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

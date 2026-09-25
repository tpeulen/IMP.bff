/**
 *  \file IMP/bff/internal/MlpTernaryTrain.h
 *  \brief Quantisation-aware training of ternary networks (BitNet b1.58).
 *
 * The recipe of "The Era of 1-bit LLMs: Training Tips, Code and FAQ"
 * (Ma, Wang, Wei; the BitLinear reference code) on bff's MLP:
 *
 * - **Master weights** float64, Adam state float64 (train_neural_net's).
 * - **Forward** of a ternary layer: `W` quantised per step with the
 *   absmean quantiser (`weight_quant`, per tensor; MlpTernary.h), its input
 *   `A` per row to int8 with absmax (`activation_quant`), and
 *   `Z = Q(A) Q(W)^T + b` on the ternary x int8 kernel -- the same
 *   quantisers and kernel as QuantizedNeuralNet, so the trained network's
 *   ternary document (to_model()) predicts training's forward pass to the
 *   bit.
 * - **Backward**, the straight-through estimator of both quantisers
 *   (BitLinear's `x + (quant(x) - x).detach()`): the layer is treated as
 *   the linear map of the *dequantised* operands `Â = q_a / s_a` and
 *   `Ŵ = q / s`, and the quantisers as the identity. So
 *   `dW = dZ^T Â` (the gradient reaches every master weight, including those
 *   clipped to +-1 -- no clipping mask, as in BitNet), `dA = dZ Ŵ`, and no
 *   gradient flows into the scales `s`, `s_a` (they sit inside the detached
 *   term). `db = sum dZ`.
 * - **GEMMs.** fprop on the ternary x int8 kernel. dgrad (`dZ Ŵ`) and wgrad
 *   (`dZ^T Â`) in float64 through the training GEMM (MatGemm) on the
 *   dequantised operands: BitNet keeps the backward in high precision
 *   ("gradients and optimizer states in high precision"); an int8 dgrad on
 *   the ternary kernel would quantise dZ, which the recipe does not.
 * - **First and last layer** float64 by default (`keep_first`,
 *   `keep_last`; the options `ternary_keep_first_layer` /
 *   `ternary_keep_last_layer`): an MLP's input layer sees the physical
 *   inputs and its output layer is the regression head, and BitNet keeps
 *   its embedding and output head in high precision too.
 * - Deterministic: no randomness beyond train_neural_net's (no stochastic
 *   rounding in this recipe).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPTERNARYTRAIN_H
#define IMPBFF_INTERNAL_MLPTERNARYTRAIN_H

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpTernary.h>

#include <cstddef>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpternary {
namespace train {

struct Config {
    bool keep_first = true;
    bool keep_last = true;
    Scale scale = Scale::Tensor;
    //! Whether layer `l` of `n` is ternary.
    bool ternary_layer(std::size_t l, std::size_t n) const {
        return !((keep_first && l == 0) || (keep_last && l + 1 == n));
    }
};

struct Workspace {
    std::vector<TernaryTensor> tw;             //!< this step's trits, a layer
    std::vector<PackedT> pw;                   //!< the same in the kernel layout
    std::vector<std::vector<double>> wdq;      //!< Ŵ = q / s, float64 (dgrad)
    std::vector<std::vector<double>> a, z;     //!< layer inputs / pre-activations
    std::vector<std::vector<double>> ahat;     //!< Â = q_a / s_a of ternary layers' inputs (wgrad)
    A8Rows q8;
    std::vector<double> da, dz, f1;
    const std::vector<double>& output() const { return a.back(); }
};

//! Quantise every ternary layer's weights (once a step).
inline void quantize_weights(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws) {
    const std::size_t L = layers.size();
    ws.tw.resize(L);
    ws.pw.resize(L);
    ws.wdq.resize(L);
    for (std::size_t l = 0; l < L; ++l) {
        if (!c.ternary_layer(l, L)) continue;
        const DenseLayer& d = layers[l];
        quantize_into(d.weight.data(), d.n_out, d.n_in, c.scale, ws.tw[l]);
        pack_kernel_into(ws.tw[l], ws.pw[l]);
        ws.wdq[l].resize(d.weight.size());
        dequantize(ws.tw[l], ws.wdq[l].data());
    }
}

//! fprop of a batch (`X` is `bs x n_in`, standardised units).
template <class Gemm = mlpcore::PortableGemm>
inline void forward(const std::vector<DenseLayer>& layers, const Config& c, const double* X, int bs,
                    Workspace& ws) {
    const std::size_t L = layers.size();
    ws.a.resize(L + 1);
    ws.z.resize(L);
    ws.ahat.resize(L);
    ws.a[0].assign(X, X + static_cast<std::size_t>(bs) * layers.front().n_in);
    for (std::size_t l = 0; l < L; ++l) {
        const DenseLayer& ly = layers[l];
        std::vector<double>& z = ws.z[l];
        z.resize(static_cast<std::size_t>(bs) * ly.n_out);
        if (c.ternary_layer(l, L)) {
            quantize_rows(ws.a[l].data(), bs, ly.n_in, static_cast<std::size_t>(ly.n_in), ws.q8);
            gemm(ws.q8, ws.pw[l], z.data(), ly.bias.data());
            std::vector<double>& ah = ws.ahat[l];
            ah.resize(static_cast<std::size_t>(bs) * ly.n_in);
            for (int r = 0; r < bs; ++r) {
                const double st = ws.q8.step[static_cast<std::size_t>(r)];
                const std::int8_t* q = ws.q8.q.data() + static_cast<std::size_t>(r) * ws.q8.kp;
                double* o = ah.data() + static_cast<std::size_t>(r) * ly.n_in;
                for (int k = 0; k < ly.n_in; ++k) o[k] = static_cast<double>(q[k]) * st;
            }
        } else {  // exactly layer_forward()'s float64 branch
            Gemm::nt(bs, ly.n_out, ly.n_in, ws.a[l].data(), ly.weight.data(), z.data());
            for (int r = 0; r < bs; ++r) {
                double* zr = z.data() + static_cast<std::size_t>(r) * ly.n_out;
                for (int o = 0; o < ly.n_out; ++o) zr[o] += ly.bias[static_cast<std::size_t>(o)];
            }
        }
        ws.a[l + 1].resize(z.size());
        mlpcore::act_apply(z.data(), ws.a[l + 1].data(), z.size(), ly.activation);
    }
}

//! Backward pass for `dY` (`bs x n_out`): `grad` (flatten() layout) is
//! overwritten. forward() must have run on the same weights.
template <class Gemm = mlpcore::PortableGemm>
inline void backward(const std::vector<DenseLayer>& layers, const Config& c, Workspace& ws, const double* dY,
                     int bs, double* grad) {
    const std::size_t L = layers.size();
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
        const bool tern = c.ternary_layer(li, L);
        // STE: dW = dZ^T Â, dA = dZ Ŵ
        Gemm::tn(ly.n_out, ly.n_in, bs, ws.dz.data(), tern ? ws.ahat[li].data() : ws.a[li].data(), gW);
        if (li == 0) break;
        ws.da.resize(static_cast<std::size_t>(bs) * ly.n_in);
        Gemm::nn(bs, ly.n_in, ly.n_out, ws.dz.data(), tern ? ws.wdq[li].data() : ly.weight.data(), ws.da.data());
    }
}

//! The inference network of a ternary-trained model: the ternary layers
//! with exactly fprop's trits and scales, the kept layers float64.
inline TModel to_model(const MlpModel& m, const Config& c, Storage st = Storage::TQ2) {
    TModel out;
    out.scale = c.scale;
    out.storage = st;
    out.x_scaler = m.x_scaler;
    out.y_scaler = m.y_scaler;
    for (std::size_t l = 0; l < m.layers.size(); ++l) {
        const DenseLayer& d = m.layers[l];
        TLayer q;
        q.n_in = d.n_in;
        q.n_out = d.n_out;
        q.activation = d.activation;
        q.bias = d.bias;
        if (c.ternary_layer(l, m.layers.size())) {
            q.weight = quantize(d.weight.data(), d.n_out, d.n_in, c.scale);
        } else {
            q.full_precision = true;
            q.weight_f64 = d.weight;
        }
        out.layers.push_back(std::move(q));
    }
    return out;
}

}  // namespace train
}  // namespace mlpternary
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPTERNARYTRAIN_H

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
 * - **GEMMs.** fprop on the ternary x int8 kernel. The backward GEMMs by
 *   `Config::backward` (option `ternary_backward`):
 *   - `Float64` (default): dgrad (`dZ Ŵ`) and wgrad (`dZ^T Â`) in float64
 *     through the training GEMM (MatGemm) on the dequantised operands --
 *     BitNet's recipe ("gradients and optimizer states in high precision").
 *   - `Int8Dgrad` (SwitchBack, Wortsman et al. 2023, arXiv 2304.13013):
 *     dgrad on the ternary x int8 kernel, `dZ` int8 per row (round to
 *     nearest even); wgrad float64.
 *   - `Int8` (Jetfire-style, Xi et al. 2024, arXiv 2403.12422): dgrad as
 *     above, and wgrad on the int8 x int8 kernel -- fprop's int8 codes of
 *     `A` as they are, `dZ` times the activation steps int8 per block of
 *     32 batch rows with stochastic rounding (SrKey of (seed, step, layer,
 *     operand 2)).
 *   MlpTernaryGrad.h states the quantisers and kernels. The STE, the master
 *   weights, the bias gradient and the kept float64 layers are the same in
 *   all three; only these GEMMs' operands are quantised. BitNet's FAQ
 *   ("Training acceleration?") names low-precision GEMM kernels for
 *   BitLinear's forward and backward as the way to speed up training.
 * - **First and last layer** float64 by default (`keep_first`,
 *   `keep_last`; the options `ternary_keep_first_layer` /
 *   `ternary_keep_last_layer`): an MLP's input layer sees the physical
 *   inputs and its output layer is the regression head, and BitNet keeps
 *   its embedding and output head in high precision too.
 * - Deterministic: `Float64` and `Int8Dgrad` draw nothing; `Int8`'s draws
 *   are counter-based (a function of seed, step, layer and element).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPTERNARYTRAIN_H
#define IMPBFF_INTERNAL_MLPTERNARYTRAIN_H

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpTernary.h>
#include <IMP/bff/internal/MlpTernaryGrad.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpternary {
namespace train {

//! The backward GEMMs of a ternary layer (the STE is the same in all three).
enum class Backward {
    Float64,    //!< dgrad and wgrad in float64 on the dequantised operands (BitNet's recipe)
    Int8Dgrad,  //!< SwitchBack: dgrad int8 (dZ per row) on the ternary kernel, wgrad float64
    Int8        //!< + wgrad int8 x int8, dZ per 32-row block, stochastic rounding (Jetfire-style)
};

//! "float64", "int8_dgrad", "int8"; throws otherwise.
inline Backward backward_from_string(const std::string& s) {
    if (s == "float64") return Backward::Float64;
    if (s == "int8_dgrad") return Backward::Int8Dgrad;
    if (s == "int8") return Backward::Int8;
    throw std::runtime_error("ternary_backward must be float64, int8_dgrad or int8, not '" + s + "'");
}

struct Config {
    bool keep_first = true;
    bool keep_last = true;
    Scale scale = Scale::Tensor;
    Backward backward = Backward::Float64;
    std::uint64_t seed = 0;  //!< keys wgrad's stochastic rounding (Backward::Int8)
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
    // int8 backward: fprop's int8 inputs a layer, W^T's trits, scratch, the
    // step counter of the stochastic rounding (backward() advances it)
    std::vector<A8Rows> q8l;
    std::vector<PackedT> pwt;
    std::vector<double> T, buf;
    G8Blocks gl;
    PackedI8 pr;
    A8Rows qg;
    std::uint64_t step = 0;
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
        if (c.backward == Backward::Float64) {
            ws.wdq[l].resize(d.weight.size());
            dequantize(ws.tw[l], ws.wdq[l].data());
        } else {
            ws.pwt.resize(L);
            pack_transposed_into(ws.tw[l], ws.pwt[l]);
        }
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
    ws.q8l.resize(L);
    ws.a[0].assign(X, X + static_cast<std::size_t>(bs) * layers.front().n_in);
    for (std::size_t l = 0; l < L; ++l) {
        const DenseLayer& ly = layers[l];
        std::vector<double>& z = ws.z[l];
        z.resize(static_cast<std::size_t>(bs) * ly.n_out);
        if (c.ternary_layer(l, L)) {
            A8Rows& q8 = ws.q8l[l];
            quantize_rows(ws.a[l].data(), bs, ly.n_in, static_cast<std::size_t>(ly.n_in), q8);
            gemm(q8, ws.pw[l], z.data(), ly.bias.data());
            if (c.backward != Backward::Int8) {  // Â for the float64 wgrad
                std::vector<double>& ah = ws.ahat[l];
                ah.resize(static_cast<std::size_t>(bs) * ly.n_in);
                for (int r = 0; r < bs; ++r) {
                    const double st = q8.step[static_cast<std::size_t>(r)];
                    const std::int8_t* q = q8.q.data() + static_cast<std::size_t>(r) * q8.kp;
                    double* o = ah.data() + static_cast<std::size_t>(r) * ly.n_in;
                    for (int k = 0; k < ly.n_in; ++k) o[k] = static_cast<double>(q[k]) * st;
                }
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
        if (tern && c.backward == Backward::Int8)
            wgrad(ws.dz.data(), bs, ly.n_out, ws.q8l[li], ly.n_in, SrKey::make(c.seed, ws.step, static_cast<int>(li), 2),
                  ws.T, ws.gl, ws.pr, gW);
        else
            Gemm::tn(ly.n_out, ly.n_in, bs, ws.dz.data(), tern ? ws.ahat[li].data() : ws.a[li].data(), gW);
        if (li == 0) break;
        ws.da.resize(static_cast<std::size_t>(bs) * ly.n_in);
        if (tern && c.backward != Backward::Float64)
            dgrad(ws.dz.data(), bs, ws.tw[li], ws.pwt[li], ws.buf, ws.qg, ws.da.data());
        else
            Gemm::nn(bs, ly.n_in, ly.n_out, ws.dz.data(), tern ? ws.wdq[li].data() : ly.weight.data(), ws.da.data());
    }
    ++ws.step;
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

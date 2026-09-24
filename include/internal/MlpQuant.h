/**
 *  \file IMP/bff/internal/MlpQuant.h
 *  \brief Dynamic-range int8 inference for a dense network.
 *
 * **A reconstruction.** tttrlib had an `MlpQuant.h` that was lost before it
 * was committed; only its description survives (tttrlib's
 * `modules/math/README.md` in a dangling stash). This file is rebuilt from
 * that spec, not recovered code:
 *
 *   > Dynamic-range int8 inference for the network -- weights quantised once
 *   > (per-tensor symmetric), activations requantised between layers, int32
 *   > accumulation. ~2x inference against the double portable path at batch
 *   > 256 (3x64 net, measured), 8x smaller weights; error bounded by the
 *   > activation absmax (<= 2% on the parity fixtures), so it is a
 *   > deployment path, not a training or derivative path. Concepts from
 *   > ermig1979/Simd's quantised inner-product scheme, re-expressed std-only.
 *
 * The scheme, as rebuilt:
 *
 * - **Weights**, once, per tensor, symmetric: `s_w = max|W| / 127`,
 *   `q_w = round(W / s_w)` in [-127, 127] (int8, one byte per weight against
 *   eight for a double -- the 8x). Biases stay double; they are `n_out`
 *   numbers per layer.
 * - **Activations**, dynamically, per row (per sample) and per layer: the
 *   layer input `a` is requantised with `s_a = max|a| / 127`. Per row rather
 *   than per batch so a sample's result does not depend on what else is in
 *   the batch.
 * - **Accumulation** in int32: `acc = sum_k q_w[j,k] q_a[k]` (exact: |term|
 *   <= 127^2, so 2^31 is reached only past 133 000 inputs), then
 *   `z = acc s_w s_a + b` and the activation in double.
 *
 * Error: each weight is off by at most `s_w / 2` and each activation by at
 * most `s_a / 2 = max|a| / 254`, so the pre-activation error of output j is
 * bounded by `(s_w / 2) sum_k |a_k| + (s_a / 2) sum_k |W_jk|` plus their
 * product -- proportional to the activation absmax, as the spec says.
 * Scalers are applied in double, outside the quantised layers.
 *
 * std-only; the kernels are plain loops the compiler vectorises (integer
 * sums may be reassociated, so the int32 dot vectorises where the double
 * one, without -ffast-math, does not). No derivative, no training: use the
 * double network for those.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPQUANT_H
#define IMPBFF_INTERNAL_MLPQUANT_H

#include <IMP/bff/internal/MlpCore.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpquant {

//! One dense layer with int8 weights and a per-tensor scale.
struct QuantLayer {
    int n_in = 0;
    int n_out = 0;
    Activation activation = Activation::Identity;
    std::vector<std::int8_t> weight;  //!< row-major n_out x n_in
    double weight_scale = 1.0;        //!< W ~= weight * weight_scale
    std::vector<double> bias;         //!< n_out, double
};

//! The quantised layers plus the (double) scalers of the source model.
struct QuantModel {
    std::vector<QuantLayer> layers;
    StandardScaler x_scaler;
    StandardScaler y_scaler;

    int n_inputs() const { return layers.empty() ? 0 : layers.front().n_in; }
    int n_outputs() const { return layers.empty() ? 0 : layers.back().n_out; }
    //! Bytes the int8 weights take (one per weight).
    std::size_t weight_bytes() const {
        std::size_t n = 0;
        for (const auto& l : layers) n += l.weight.size() * sizeof(std::int8_t);
        return n;
    }
};

//! Symmetric int8 quantisation of `n` values; returns the scale.
/*! `s = max|v| / 127` (1 when all are zero), `q = round(v / s)`. */
inline double quantize_symmetric(const double* v, std::size_t n, std::int8_t* q) {
    double amax = 0.0;
    for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::abs(v[i]));
    const double s = amax > 0.0 ? amax / 127.0 : 1.0;
    const double inv = 1.0 / s;
    for (std::size_t i = 0; i < n; ++i) {
        const long r = std::lround(v[i] * inv);
        q[i] = static_cast<std::int8_t>(std::max(-127L, std::min(127L, r)));
    }
    return s;
}

//! Quantise a model's weights once.
inline QuantModel quantize(const MlpModel& m) {
    m.validate();
    QuantModel out;
    out.x_scaler = m.x_scaler;
    out.y_scaler = m.y_scaler;
    for (const DenseLayer& l : m.layers) {
        QuantLayer q;
        q.n_in = l.n_in;
        q.n_out = l.n_out;
        q.activation = l.activation;
        q.bias = l.bias;
        q.weight.resize(l.weight.size());
        q.weight_scale = quantize_symmetric(l.weight.data(), l.weight.size(), q.weight.data());
        out.layers.push_back(std::move(q));
    }
    return out;
}

//! int8 . int8 with an int32 accumulator.
inline std::int32_t dot_i8(const std::int8_t* a, const std::int8_t* b, int n) {
    std::int32_t acc = 0;
    for (int k = 0; k < n; ++k)
        acc += static_cast<std::int32_t>(a[k]) * static_cast<std::int32_t>(b[k]);
    return acc;
}

//! Forward pass of a batch, in the model's physical units.
/*! `X` is `n_rows x n_inputs`, row-major; `y` becomes `n_rows x n_outputs`. */
inline void predict(const QuantModel& m, const double* X, int n_rows, std::vector<double>& y) {
    y.clear();
    if (n_rows <= 0 || m.layers.empty()) return;
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    std::vector<double> a(X, X + rows * static_cast<std::size_t>(m.n_inputs())), z;
    mlpcore::detail::scale_in(a, n_rows, m.n_inputs(), m.x_scaler);
    std::vector<std::int8_t> qa;
    std::vector<double> sa(rows);
    for (const QuantLayer& l : m.layers) {
        const std::size_t n_in = static_cast<std::size_t>(l.n_in);
        const std::size_t n_out = static_cast<std::size_t>(l.n_out);
        qa.resize(rows * n_in);
        for (std::size_t r = 0; r < rows; ++r)
            sa[r] = quantize_symmetric(a.data() + r * n_in, n_in, qa.data() + r * n_in);
        z.resize(rows * n_out);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::int8_t* qrow = qa.data() + r * n_in;
            const double s = sa[r] * l.weight_scale;
            double* zrow = z.data() + r * n_out;
            for (std::size_t j = 0; j < n_out; ++j)
                zrow[j] = static_cast<double>(dot_i8(qrow, l.weight.data() + j * n_in, l.n_in)) * s +
                          l.bias[j];
        }
        a.resize(z.size());
        mlpcore::act_apply(z.data(), a.data(), z.size(), l.activation);
    }
    y.swap(a);
    mlpcore::detail::unscale_out(y, n_rows, m.n_outputs(), m.y_scaler);
}

}  // namespace mlpquant
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPQUANT_H

/**
 *  \file IMP/bff/internal/NetworkGguf.h
 *  \brief The mapping between bff's networks and GGUF files.
 *
 * GGUF is the interchange format of llama.cpp and MLX; `GgufIo.h` is its
 * std-only container and block codecs. This header maps bff's three network
 * forms onto it -- the float `MlpModel`, the quantised
 * `QuantizedNeuralNet` formats and the ternary `TModel` -- with one eye on
 * bit-exact round trips (bff to GGUF to bff reproduces the network) and one
 * on files a foreign tool can read.
 *
 * **Tensor layout.** A layer's weights are one tensor `blk.{i}.weight`
 * (ggml convention: `ne[0]` = n_in, `ne[1]` = n_out, row-major, blocks run
 * along the input dimension) and its biases `blk.{i}.bias` (F32, or F64 in
 * an F64 file). The float layers' scales ride in metadata (`bff.x_scaler.*`
 * / `bff.y_scaler.*`, F64 arrays), the layer shapes and activations in
 * `bff.dims` / `bff.activations`, the architecture tag is `bff.mlp`.
 *
 * **Type mapping** (quantised export, `QuantizedNeuralNet::to_gguf`):
 *
 *  - `"int8"` -> GGUF **I8** (raw int8 bytes) with the per-tensor scale in
 *    metadata `bff.int8.scales` (one F64 a layer). Exact round trip; a
 *    foreign reader sees the int codes (dequantise with the metadata or
 *    treat as raw bytes).
 *  - `"mxfp4"` -> GGUF **MXFP4**. ggml halves the E8M0 scale (a stored code
 *    e decodes to 2^(e - 128) times E2M1) where bff decodes 2^(e - 127), so
 *    the export writes each scale byte plus one and the import takes one
 *    off: values, codes and scales round trip bit-exactly with no metadata,
 *    and a foreign reader decodes the exact bff weights. (Code 254
 *    saturates; unreachable for real weights.)
 *  - `"nvfp4"` -> GGUF **NVFP4** with bff's float32 global scale folded
 *    into the block scales (`new = fp32_to_ue4m3(2 decode(old) g)`, since
 *    ggml's ue4m3 convention carries a factor 2). The codes are exact; the
 *    folded scale rounds to E4M3 -- at most half a mantissa step (1/16
 *    relative) a block, quantisation noise. ggml has no slot for a global
 *    scale; `bff.nvfp4.tensor_scale` records the original. Import restores
 *    `tensor_scale = 1` on the folded scales. **The only approximate
 *    mapping in this header.**
 *  - `"ternary"` / `"ternary_row"` -> GGUF **TQ2_0**, `"ternary_tq1"` /
 *    `"ternary_tq1_row"` -> GGUF **TQ1_0**. The packed trits are exact; the
 *    absmean scales (bff: one F64 a tensor or row) come back from metadata
 *    `bff.ternary.gamma` (F64 array) + `bff.ternary.gamma_counts` (U32, one
 *    count a quantised layer). Without the metadata (a foreign TQ file)
 *    the gamma is `max` of the block scales of the tensor/row -- fp16
 *    rounded, so a bff->GGUF->bff round trip without metadata lands within
 *    one fp16 step of the weights. Block padding to 256 uses trit 0.
 *  - `"fp4"` (bff's per-row-scale E2M1) has **no GGUF type**; exported
 *    dequantised as F32 (`full_precision` layers likewise), so the file is
 *    correct but the quantised form does not survive.
 *  - float networks (`NeuralNet::to_gguf`) take the type as an argument:
 *    "f32" (default), "f16", "bf16", "q8_0", "q4_0", "mxfp4", "nvfp4",
 *    "tq2_0"/"tq1_0", "q1_0" (sign-only, 1.125 bits a weight) or "f64" --
 *    the one bit-exact choice for a float network.
 *
 * **Import** (`NeuralNet::from_gguf`) reads a `bff.mlp` file of any of the
 * types above: quantised tensors are dequantised (mxfp4 scale convention
 * reversed), biases F32/F64, scalers from metadata. Foreign architectures
 * are refused with a message that names what was found -- a llama.cpp file
 * is not an MLP. `QuantizedNeuralNet::from_gguf` rebuilds the *quantised*
 * network from `bff.quantization` + the exact codes (int8, mxfp4, ternary;
 * nvfp4 folded; fp4 arrives as float layers).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NETWORK_GGUF_H
#define IMPBFF_INTERNAL_NETWORK_GGUF_H

#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/internal/GgufIo.h>
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpFp4.h>
#include <IMP/bff/internal/MlpQuant.h>
#include <IMP/bff/internal/MlpTernary.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {
namespace netgguf {

using gguf::Type;

//! A layer count / dims / activation list -> metadata, scalers -> metadata.
inline void put_model_kv(gguf::Writer& w, const MlpModel& m, const std::string& weight_type) {
    w.add_str("general.architecture", "bff.mlp");
    w.add_u32("bff.layer_count", static_cast<std::uint32_t>(m.layers.size()));
    std::vector<std::uint32_t> dims;
    std::vector<std::string> acts;
    for (const DenseLayer& l : m.layers) {
        dims.push_back(static_cast<std::uint32_t>(l.n_in));
        acts.push_back(activation_to_string(l.activation));
    }
    dims.push_back(static_cast<std::uint32_t>(m.layers.back().n_out));
    w.add_array_u32("bff.dims", dims);
    w.add_array_str("bff.activations", acts);
    w.add_str("bff.weight_type", weight_type);
    if (m.x_scaler.active()) {
        w.add_array_f64("bff.x_scaler.mean", m.x_scaler.mean);
        w.add_array_f64("bff.x_scaler.scale", m.x_scaler.scale);
    }
    if (m.y_scaler.active()) {
        w.add_array_f64("bff.y_scaler.mean", m.y_scaler.mean);
        w.add_array_f64("bff.y_scaler.scale", m.y_scaler.scale);
    }
}

//! The scaler metadata of a file -> scalers.
inline void get_model_scalers(const gguf::Reader& r, MlpModel& m) {
    if (r.has("bff.x_scaler.mean")) {
        m.x_scaler.mean = r.get_array_f64("bff.x_scaler.mean");
        m.x_scaler.scale = r.get_array_f64("bff.x_scaler.scale");
    }
    if (r.has("bff.y_scaler.mean")) {
        m.y_scaler.mean = r.get_array_f64("bff.y_scaler.mean");
        m.y_scaler.scale = r.get_array_f64("bff.y_scaler.scale");
    }
}

//! Dequantise a tensor of a file into float64 `out` (n elements). F64
//! tensors are read verbatim (bit-exact); the rest widen their float32.
inline void tensor_to_doubles(const gguf::Reader& r, const gguf::TensorInfo& t,
                              std::vector<double>& out) {
    const std::size_t n = static_cast<std::size_t>(t.n_elements());
    const std::vector<std::uint8_t> raw = r.tensor_bytes(t);
    out.assign(n, 0.0);
    gguf::decode_row_d(t.type, raw.data(), static_cast<int>(n), out.data());
}

//! Whether the MXFP4 scale bytes must shift by one (ggml's half convention).
/*! See the file comment: export +1, import -1. */
constexpr int kMxfp4Shift = 1;

//! One MXFP4 row of `n` elements: shift every 17-byte block's E8M0 byte.
inline void shift_mxfp4(std::vector<std::uint8_t>& bytes, int n, int delta) {
    const std::size_t blocks = static_cast<std::size_t>(n) / 32;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint8_t& e = bytes[b * 17];
        const int v = static_cast<int>(e) + delta;
        e = static_cast<std::uint8_t>(std::max(0, std::min(254, v)));
    }
}

// ---------------------------------------------------------------- export

//! The float model as a GGUF file image; `type_name` picks the weight type.
inline std::vector<std::uint8_t> model_to_gguf(const MlpModel& m, const std::string& type_name) {
    m.validate();
    const Type type = gguf::type_from_string(type_name);
    const bool f64_biases = type == Type::F64;
    gguf::Writer w;
    put_model_kv(w, m, gguf::type_to_string(type));
    std::vector<float> buf;
    for (std::size_t i = 0; i < m.layers.size(); ++i) {
        const DenseLayer& l = m.layers[i];
        const int n = l.n_in * l.n_out;
        // an F64 file keeps the weights bit-exact: no float detour
        std::vector<double> buf64;
        if (type == Type::F64) {
            buf64.assign(l.weight.begin(), l.weight.begin() + n);
        } else {
            buf.resize(static_cast<std::size_t>(n));
            for (int k = 0; k < n; ++k) buf[static_cast<std::size_t>(k)] = static_cast<float>(l.weight[static_cast<std::size_t>(k)]);
        }
        // quantised block types run along ne[0] (the input dimension):
        // pad the row to the block size with zero-value codes.
        int bs, bb;
        gguf::type_shape(type, bs, bb);
        const int padded = bs > 1 ? (l.n_in + bs - 1) / bs * bs : l.n_in;
        std::vector<std::uint8_t> enc(gguf::row_bytes(type, padded * l.n_out));
        std::vector<float> padded_w;
        std::vector<double> padded_w64;
        if (padded != l.n_in) {
            if (type == Type::F64) {
                padded_w64.assign(static_cast<std::size_t>(padded) * l.n_out, 0.0);
                for (int o = 0; o < l.n_out; ++o)
                    std::copy_n(buf64.data() + static_cast<std::size_t>(o) * l.n_in, l.n_in,
                                padded_w64.data() + static_cast<std::size_t>(o) * padded);
            } else {
                padded_w.assign(static_cast<std::size_t>(padded) * l.n_out, 0.0f);
                for (int o = 0; o < l.n_out; ++o)
                    std::copy_n(buf.data() + static_cast<std::size_t>(o) * l.n_in, l.n_in,
                                padded_w.data() + static_cast<std::size_t>(o) * padded);
            }
        }
        if (type == Type::F64) {
            gguf::encode_row_d(type, padded_w64.empty() ? buf64.data() : padded_w64.data(),
                               padded * l.n_out, enc.data());
        } else {
            const float* src = padded_w.empty() ? buf.data() : padded_w.data();
            gguf::encode_row(type, src, padded * l.n_out, enc.data());
        }
        // no MXFP4 scale shift: bff's e8m0 convention already matches ggml's
        w.add_tensor("blk." + std::to_string(i) + ".weight",
                     static_cast<std::uint64_t>(padded), static_cast<std::uint64_t>(l.n_out),
                     type, enc.data());
        // biases: F32, or F64 in an F64 file
        std::vector<std::uint8_t> enc_b(l.n_out * (f64_biases ? 8 : 4));
        for (int o = 0; o < l.n_out; ++o) {
            if (f64_biases) {
                std::uint64_t u;
                std::memcpy(&u, &l.bias[static_cast<std::size_t>(o)], 8);
                for (int b = 0; b < 8; ++b)
                    enc_b[static_cast<std::size_t>(8 * o + b)] = static_cast<std::uint8_t>(u >> (8 * b));
            } else {
                const float v = static_cast<float>(l.bias[static_cast<std::size_t>(o)]);
                std::uint32_t u;
                std::memcpy(&u, &v, 4);
                for (int b = 0; b < 4; ++b)
                    enc_b[static_cast<std::size_t>(4 * o + b)] = static_cast<std::uint8_t>(u >> (8 * b));
            }
        }
        w.add_tensor("blk." + std::to_string(i) + ".bias", static_cast<std::uint64_t>(l.n_out), 1,
                     f64_biases ? Type::F64 : Type::F32, enc_b.data());
    }
    return w.finish();
}

//! Re-group one packed-FP4 block from bff's sequential nibble pairs
//! (element 2i low / 2i + 1 high) to ggml's split layout (element j low /
//! element `n/2` + j high), in place on `out` (code_bytes long both ways).
inline void regroup_fp4_block(const std::uint8_t* src, std::uint8_t* out,
                              int code_bytes) {
    const auto nib = [&src](int e) {
        return static_cast<std::uint8_t>((src[e >> 1] >> (4 * (e & 1))) & 0x0F);
    };
    for (int j = 0; j < code_bytes; ++j)
        out[j] = static_cast<std::uint8_t>(nib(j) | (nib(code_bytes + j) << 4));
}

//! The quantised (FP4/int8) model as a GGUF file image.
/*! `format` is "int8", "fp4", "mxfp4" or "nvfp4"; the ternary formats go
    through ternary_to_gguf() below. */
inline std::vector<std::uint8_t> fp4_to_gguf(const std::string& format,
                                             const mlpquant::QuantModel& q8,
                                             const mlpfp4::Fp4Model& f4) {
    const bool is_int8 = format == "int8";
    const mlpfp4::Format ff = is_int8 ? mlpfp4::Format::FP4 : mlpfp4::format_from_string(format);
    gguf::Writer w;
    w.add_str("general.architecture", "bff.mlp");
    w.add_str("bff.quantization", format);
    w.add_bool("bff.quantize_activations",
               is_int8 ? true : f4.quantize_activations);
    w.add_str("bff.weight_type", is_int8 ? "I8" : gguf::type_to_string(
                                   ff == mlpfp4::Format::MXFP4 ? Type::MXFP4
                                   : ff == mlpfp4::Format::NVFP4 ? Type::NVFP4
                                                                 : Type::F32));
    const std::size_t nl = is_int8 ? q8.layers.size() : f4.layers.size();
    w.add_u32("bff.layer_count", static_cast<std::uint32_t>(nl));
    std::vector<std::uint32_t> dims;
    std::vector<std::string> acts;
    std::vector<double> int8_scales;
    for (std::size_t i = 0; i < nl; ++i) {
        const int n_in = is_int8 ? q8.layers[i].n_in : f4.layers[i].n_in;
        const int n_out = is_int8 ? q8.layers[i].n_out : f4.layers[i].n_out;
        const Activation act = is_int8 ? q8.layers[i].activation : f4.layers[i].activation;
        dims.push_back(static_cast<std::uint32_t>(n_in));
        acts.push_back(activation_to_string(act));
    }
    if (nl) dims.push_back(static_cast<std::uint32_t>(
               is_int8 ? q8.layers.back().n_out : f4.layers.back().n_out));
    w.add_array_u32("bff.dims", dims);
    w.add_array_str("bff.activations", acts);
    const StandardScaler& xs = is_int8 ? q8.x_scaler : f4.x_scaler;
    const StandardScaler& ys = is_int8 ? q8.y_scaler : f4.y_scaler;
    if (xs.active()) {
        w.add_array_f64("bff.x_scaler.mean", xs.mean);
        w.add_array_f64("bff.x_scaler.scale", xs.scale);
    }
    if (ys.active()) {
        w.add_array_f64("bff.y_scaler.mean", ys.mean);
        w.add_array_f64("bff.y_scaler.scale", ys.scale);
    }

    for (std::size_t i = 0; i < nl; ++i) {
        const std::string base = "blk." + std::to_string(i);
        if (is_int8) {
            const mlpquant::QuantLayer& l = q8.layers[i];
            std::vector<std::uint8_t> enc(l.weight.size());
            for (std::size_t k = 0; k < l.weight.size(); ++k)
                enc[k] = static_cast<std::uint8_t>(l.weight[k]);
            w.add_tensor(base + ".weight", static_cast<std::uint64_t>(l.n_in),
                         static_cast<std::uint64_t>(l.n_out), Type::I8, enc.data());
            int8_scales.push_back(l.weight_scale);
            std::vector<std::uint8_t> enc_b(l.bias.size() * 8);
            for (std::size_t o = 0; o < l.bias.size(); ++o) {
                std::uint64_t u;
                std::memcpy(&u, &l.bias[o], 8);
                for (int b = 0; b < 8; ++b)
                    enc_b[o * 8 + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
            }
            w.add_tensor(base + ".bias", static_cast<std::uint64_t>(l.bias.size()), 1,
                         Type::F64, enc_b.data());
            continue;
        }
        const mlpfp4::Fp4Layer& l = f4.layers[i];
        // bias, always F64 in a quantised document
        std::vector<std::uint8_t> enc_b(l.bias.size() * 8);
        for (std::size_t o = 0; o < l.bias.size(); ++o) {
            std::uint64_t u;
            std::memcpy(&u, &l.bias[o], 8);
            for (int b = 0; b < 8; ++b)
                enc_b[o * 8 + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
        }
        w.add_tensor(base + ".bias", static_cast<std::uint64_t>(l.bias.size()), 1,
                     Type::F64, enc_b.data());
        if (l.full_precision || ff == mlpfp4::Format::FP4) {
            // no GGUF type for per-row-scale FP4: dequantised F32
            std::vector<double> wgt(static_cast<std::size_t>(l.n_in) * l.n_out);
            if (l.full_precision)
                wgt = l.weight_f64;
            else
                mlpfp4::dequantize(l.weight, wgt.data());
            std::vector<float> wf(wgt.size());
            for (std::size_t k = 0; k < wgt.size(); ++k) wf[k] = static_cast<float>(wgt[k]);
            std::vector<std::uint8_t> enc(wf.size() * 4);
            gguf::encode_row(Type::F32, wf.data(), static_cast<int>(wf.size()), enc.data());
            w.add_tensor(base + ".weight", static_cast<std::uint64_t>(l.n_in),
                         static_cast<std::uint64_t>(l.n_out), Type::F32, enc.data());
            continue;
        }
        // mxfp4 / nvfp4: bff keeps codes and scales apart; GGUF interleaves
        // them. MXFP4: 17-byte blocks of one E8M0 scale + 16 code bytes.
        // NVFP4: 36-byte SUPER-blocks of 64 elements — 4 E4M3 scale bytes
        // first, then all 32 code bytes (gguf's layout, NOT interleaved).
        // A GGUF NVFP4 row must hold whole 64-element super-blocks; when
        // bff's padded row (multiple of 32) is not a multiple of 64, emit
        // the tensor with ne[0] padded to 64 and zero sub-blocks for the
        // pad columns (the reader strips them via bff.dims).
        const bool nv = ff == mlpfp4::Format::NVFP4;
        const int block = nv ? 16 : 32;
        const int code_bytes = block / 2;
        const int sub_per_super = nv ? 4 : 1;
        int file_cols = l.weight.padded();
        if (nv && file_cols % 64 != 0) file_cols = (file_cols + 63) / 64 * 64;
        const int bpr = file_cols / block;  // sub-blocks a file row holds
        std::vector<std::uint8_t> enc;
        enc.reserve(static_cast<std::size_t>(l.n_out) * (file_cols / block) *
                    (code_bytes + sub_per_super));
        if (nv)
            w.add_f64("bff.nvfp4." + base + ".tensor_scale", l.weight.tensor_scale);
        // fold the scales once: the file value is kvalues2[c] * ue4m3_half(B)
        // = kE2M1[c] * ue4m3_raw(B), so ue4m3_raw(B) must equal the bff
        // scale e4m3(s) * g — no factor 2 anywhere
        std::vector<std::uint8_t> folded_scales(
                static_cast<std::size_t>(l.n_out) * bpr);
        const int real_bpr =
                static_cast<int>(l.weight.padded()) / block;  // bff's own rows
        for (int r = 0; r < l.n_out; ++r) {
            const std::uint8_t* sr =
                    l.weight.scales.data() + static_cast<std::size_t>(r) * l.weight.row_scale_bytes();
            for (int b = 0; b < real_bpr; ++b) {
                std::uint8_t s = sr[b];
                if (nv) {
                    const double raw = mlpfp4::e4m3_decode(s);
                    s = gguf::detail::value_to_ue4m3(static_cast<float>(raw * l.weight.tensor_scale));
                }
                folded_scales[static_cast<std::size_t>(r) * bpr + b] = s;
            }
            // pad sub-blocks: scale code of value 0 (any works with zero
            // codes, but keep it defined)
            for (int b = real_bpr; b < bpr; ++b)
                folded_scales[static_cast<std::size_t>(r) * bpr + b] = 0;
        }
        if (!nv) {
            for (int r = 0; r < l.n_out; ++r) {
                const std::uint8_t* cr =
                        l.weight.codes.data() + static_cast<std::size_t>(r) * l.weight.row_bytes();
                for (int b = 0; b < bpr; ++b) {
                    enc.push_back(folded_scales[static_cast<std::size_t>(r) * bpr + b]);
                    std::vector<std::uint8_t> blk(static_cast<std::size_t>(code_bytes));
                    regroup_fp4_block(cr + static_cast<std::size_t>(b) * code_bytes,
                                      blk.data(), code_bytes);
                    enc.insert(enc.end(), blk.begin(), blk.end());
                }
            }
        } else {  // NVFP4 super-blocks: row = 4 scales, then 32 code bytes
            for (int r = 0; r < l.n_out; ++r) {
                const std::uint8_t* cr =
                        l.weight.codes.data() + static_cast<std::size_t>(r) * l.weight.row_bytes();
                for (int b = 0; b < bpr; ++b)
                    enc.push_back(folded_scales[static_cast<std::size_t>(r) * bpr + b]);
                std::vector<std::uint8_t> blk(
                        static_cast<std::size_t>(bpr) * code_bytes, 0);
                for (int b = 0; b < real_bpr; ++b)
                    regroup_fp4_block(cr + static_cast<std::size_t>(b) * code_bytes,
                                      blk.data() + static_cast<std::size_t>(b) * code_bytes,
                                      code_bytes);
                enc.insert(enc.end(), blk.begin(), blk.end());
            }
        }
        w.add_tensor(base + ".weight", static_cast<std::uint64_t>(file_cols),
                     static_cast<std::uint64_t>(l.n_out),
                     nv ? Type::NVFP4 : Type::MXFP4, enc.data());
    }
    if (!int8_scales.empty()) w.add_array_f64("bff.int8.scales", int8_scales);
    return w.finish();
}

//! The ternary model as a GGUF file image (TQ2_0 or TQ1_0 tensors).
inline std::vector<std::uint8_t> ternary_to_gguf(const mlpternary::TModel& tm) {
    gguf::Writer w;
    w.add_str("general.architecture", "bff.mlp");
    w.add_str("bff.quantization", mlpternary::format_name(tm.scale, tm.storage));
    w.add_bool("bff.quantize_activations", true);
    w.add_str("bff.weight_type", tm.storage == mlpternary::Storage::TQ1 ? "TQ1_0" : "TQ2_0");
    w.add_u32("bff.layer_count", static_cast<std::uint32_t>(tm.layers.size()));
    std::vector<std::uint32_t> dims, gamma_counts;
    std::vector<std::string> acts;
    std::vector<double> gamma_all;
    for (const mlpternary::TLayer& l : tm.layers) {
        dims.push_back(static_cast<std::uint32_t>(l.n_in));
        acts.push_back(activation_to_string(l.activation));
        if (!l.full_precision) {
            gamma_all.insert(gamma_all.end(), l.weight.gamma.begin(), l.weight.gamma.end());
            gamma_counts.push_back(static_cast<std::uint32_t>(l.weight.gamma.size()));
        }
    }
    if (!tm.layers.empty())
        dims.push_back(static_cast<std::uint32_t>(tm.layers.back().n_out));
    w.add_array_u32("bff.dims", dims);
    w.add_array_str("bff.activations", acts);
    w.add_array_u32("bff.ternary.gamma_counts", gamma_counts);
    w.add_array_f64("bff.ternary.gamma", gamma_all);
    if (tm.x_scaler.active()) {
        w.add_array_f64("bff.x_scaler.mean", tm.x_scaler.mean);
        w.add_array_f64("bff.x_scaler.scale", tm.x_scaler.scale);
    }
    if (tm.y_scaler.active()) {
        w.add_array_f64("bff.y_scaler.mean", tm.y_scaler.mean);
        w.add_array_f64("bff.y_scaler.scale", tm.y_scaler.scale);
    }
    const gguf::Type type = tm.storage == mlpternary::Storage::TQ1 ? Type::TQ1_0 : Type::TQ2_0;
    for (std::size_t i = 0; i < tm.layers.size(); ++i) {
        const mlpternary::TLayer& l = tm.layers[i];
        const std::string base = "blk." + std::to_string(i);
        std::vector<std::uint8_t> enc_b(l.bias.size() * 8);
        for (std::size_t o = 0; o < l.bias.size(); ++o) {
            std::uint64_t u;
            std::memcpy(&u, &l.bias[o], 8);
            for (int b = 0; b < 8; ++b)
                enc_b[o * 8 + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
        }
        w.add_tensor(base + ".bias", static_cast<std::uint64_t>(l.bias.size()), 1,
                     Type::F64, enc_b.data());
        if (l.full_precision) {
            std::vector<float> wf(l.weight_f64.size());
            for (std::size_t k = 0; k < wf.size(); ++k) wf[k] = static_cast<float>(l.weight_f64[k]);
            std::vector<std::uint8_t> enc(wf.size() * 4);
            gguf::encode_row(Type::F32, wf.data(), static_cast<int>(wf.size()), enc.data());
            w.add_tensor(base + ".weight", static_cast<std::uint64_t>(l.n_in),
                         static_cast<std::uint64_t>(l.n_out), Type::F32, enc.data());
            continue;
        }
        // dequantise exactly (q * gamma), then encode with ggml's blocks;
        // the fp16 block scales are as coarse as the format allows, the
        // exact gammas ride in the metadata
        const std::size_t nw = static_cast<std::size_t>(l.n_in) * l.n_out;
        std::vector<double> wgt(nw);
        mlpternary::dequantize(l.weight, wgt.data());
        const int bs = 256;
        const int padded = (l.n_in + bs - 1) / bs * bs;
        std::vector<float> wf(static_cast<std::size_t>(padded) * l.n_out, 0.0f);
        for (int o = 0; o < l.n_out; ++o)
            std::copy_n(wgt.data() + static_cast<std::size_t>(o) * l.n_in, l.n_in,
                        wf.data() + static_cast<std::size_t>(o) * padded);
        std::vector<std::uint8_t> enc(gguf::row_bytes(type, padded * l.n_out));
        gguf::encode_row(type, wf.data(), padded * l.n_out, enc.data());
        w.add_tensor(base + ".weight", static_cast<std::uint64_t>(padded),
                     static_cast<std::uint64_t>(l.n_out), type, enc.data());
    }
    return w.finish();
}

// ---------------------------------------------------------------- import

//! A `bff.mlp` file image -> the float model (quantised tensors dequantised).
/*! The weights are read as stored; for MXFP4 bff's writer shifted the scale
    bytes one up (ggml's half-scale convention), which is undone when
    `bff.weight_type` says the file is bff's own. Rows padded to a block
    size are truncated to `bff.dims`' layer widths. */
inline MlpModel model_from_gguf(const std::vector<std::uint8_t>& bytes) {
    const gguf::Reader r(bytes);
    const std::string arch = r.get_str("general.architecture");
    if (arch != "bff.mlp")
        throw std::runtime_error("NeuralNet::from_gguf: the file is a '" + arch +
                                 "' model, not a bff.mlp network");
    const std::string weight_type = r.get_str("bff.weight_type");
    const bool bff_mxfp4 = weight_type == "MXFP4" || weight_type == "mxfp4";
    const std::uint64_t nl = r.get_u64("bff.layer_count");
    std::vector<std::uint32_t> dims;
    std::vector<std::string> acts;
    if (r.has("bff.dims")) {
        const gguf::Value* v = r.find("bff.dims");
        dims = v->au32;
    }
    if (r.has("bff.activations")) {
        const gguf::Value* v = r.find("bff.activations");
        acts = v->astr;
    }
    MlpModel m;
    get_model_scalers(r, m);
    for (std::uint64_t i = 0; i < nl; ++i) {
        const std::string base = "blk." + std::to_string(i);
        const gguf::TensorInfo* wt = r.find_tensor(base + ".weight");
        if (wt == nullptr) throw std::runtime_error("NeuralNet::from_gguf: no tensor '" + base + ".weight'");
        const gguf::TensorInfo* bt = r.find_tensor(base + ".bias");
        if (bt == nullptr) throw std::runtime_error("NeuralNet::from_gguf: no tensor '" + base + ".bias'");
        DenseLayer l;
        l.n_in = static_cast<int>(wt->ne[0]);
        l.n_out = static_cast<int>(wt->ne[1]);
        if (i + 1 < dims.size()) l.n_in = static_cast<int>(dims[i]);  // true width (padding stripped)
        if (i < acts.size()) l.activation = activation_from_string(acts[i]);
        if (l.n_in <= 0 || l.n_out <= 0 || l.n_in > (1 << 24) || l.n_out > (1 << 24))
            throw std::runtime_error("NeuralNet::from_gguf: layer size out of range");
        // weight, dequantised, row-major with rows of wt->ne[0]; strip pad.
        // An F64 file keeps the weights bit-exact: no float detour.
        const std::size_t stored = static_cast<std::size_t>(wt->n_elements());
        std::vector<std::uint8_t> raw = r.tensor_bytes(*wt);
        // no MXFP4 un-shift: bff's e8m0 convention already matches ggml's
        const std::size_t cols = static_cast<std::size_t>(wt->ne[0]);
        const std::size_t keep = static_cast<std::size_t>(l.n_in);
        l.weight.resize(keep * static_cast<std::size_t>(l.n_out));
        if (wt->type == Type::F64) {
            // stored rows may be padded out to cols; read into a flat
            // buffer then strip the pad column-wise
            std::vector<double> xw(stored);
            gguf::decode_row_d(Type::F64, raw.data(), static_cast<int>(stored), xw.data());
            for (int o = 0; o < l.n_out; ++o)
                for (std::size_t k = 0; k < keep; ++k)
                    l.weight[static_cast<std::size_t>(o) * keep + k] =
                        xw[static_cast<std::size_t>(o) * cols + k];
        } else {
            std::vector<float> x(stored);
            gguf::decode_row(wt->type, raw.data(), static_cast<int>(stored), x.data());
            for (int o = 0; o < l.n_out; ++o)
                for (std::size_t k = 0; k < keep; ++k)
                    l.weight[static_cast<std::size_t>(o) * keep + k] =
                        static_cast<double>(x[static_cast<std::size_t>(o) * cols + k]);
        }
        // int8 weights are raw codes; scale them by the metadata
        if (wt->type == Type::I8 && r.has("bff.int8.scales")) {
            const gguf::Value* v = r.find("bff.int8.scales");
            if (i < v->af64.size() && (*v).af64[static_cast<std::size_t>(i)] != 1.0) {
                const double s = v->af64[static_cast<std::size_t>(i)];
                for (double& wv : l.weight) wv *= s;
            }
        }
        // biases
        std::vector<double> bias;
        tensor_to_doubles(r, *bt, bias);
        if (bias.size() != static_cast<std::size_t>(l.n_out))
            throw std::runtime_error("NeuralNet::from_gguf: bias length mismatch");
        l.bias = std::move(bias);
        m.layers.push_back(std::move(l));
    }
    m.validate();
    return m;
}

//! The quantised GGUF image -> the (format, q8, f4, tm) it encodes.
/*! Reads what fp4_to_gguf() / ternary_to_gguf() wrote: the format from
    `bff.quantization`, the codes from the tensors (int8 I8 + the scales
    from metadata, mxfp4 with the scale shift undone, nvfp4 unfolded back
    onto `tensor_scale = 1`, ternary re-quantised against the metadata
    gamma). `quantize_activations` comes from the metadata. \throws on a
    foreign architecture or a file whose format is not one of bff's. */
inline std::string quantized_from_gguf(const std::vector<std::uint8_t>& bytes,
                                       bool& quantize_activations,
                                       mlpquant::QuantModel& q8, mlpfp4::Fp4Model& f4,
                                       mlpternary::TModel& tm) {
    const gguf::Reader r(bytes);
    const std::string arch = r.get_str("general.architecture");
    if (arch != "bff.mlp")
        throw std::runtime_error("QuantizedNeuralNet::from_gguf: the file is a '" + arch +
                                 "' model, not a bff.mlp network");
    const std::string format = r.get_str("bff.quantization");
    quantize_activations = true;
    if (r.has("bff.quantize_activations")) {
        const gguf::Value* v = r.find("bff.quantize_activations");
        if (v->kind == gguf::Value::BOOL) quantize_activations = v->b;
    }
    const bool is_int8 = format == "int8";
    const bool is_fp4 = format == "fp4";
    mlpternary::Scale tsc = mlpternary::Scale::Tensor;
    mlpternary::Storage tst = mlpternary::Storage::TQ2;
    const bool is_ter = mlpternary::parse_format(format, tsc, tst);
    mlpfp4::Format ff = mlpfp4::Format::FP4;
    if (!is_int8 && !is_ter && !is_fp4)
        ff = mlpfp4::format_from_string(format);
    if (is_ter && !quantize_activations)
        throw std::runtime_error("QuantizedNeuralNet::from_gguf: ternary needs quantize_activations");
    std::vector<std::uint32_t> dims;
    std::vector<std::string> acts;
    if (r.has("bff.dims")) dims = r.find("bff.dims")->au32;
    if (r.has("bff.activations")) acts = r.find("bff.activations")->astr;
    q8 = mlpquant::QuantModel();
    f4 = mlpfp4::Fp4Model();
    tm = mlpternary::TModel();
    tm.scale = tsc;
    tm.storage = tst;
    f4.format = ff;
    f4.quantize_activations = quantize_activations;
    const gguf::Value* scales_v = r.find("bff.int8.scales");
    const gguf::Value* gamma_counts = r.find("bff.ternary.gamma_counts");
    const gguf::Value* gamma_all = r.find("bff.ternary.gamma");
    std::size_t gamma_at = 0;
    const std::uint64_t nl = r.get_u64("bff.layer_count");
    for (std::uint64_t i = 0; i < nl; ++i) {
        const std::string base = "blk." + std::to_string(i);
        const gguf::TensorInfo* wt = r.find_tensor(base + ".weight");
        if (wt == nullptr)
            throw std::runtime_error("QuantizedNeuralNet::from_gguf: no tensor '" + base + ".weight'");
        const gguf::TensorInfo* bt = r.find_tensor(base + ".bias");
        if (bt == nullptr)
            throw std::runtime_error("QuantizedNeuralNet::from_gguf: no tensor '" + base + ".bias'");
        const int n_in = static_cast<int>(i + 1 < dims.size() ? static_cast<int>(dims[i]) : wt->ne[0]);
        const int n_out = static_cast<int>(wt->ne[1]);
        if (n_in <= 0 || n_out <= 0 || n_in > (1 << 24) || n_out > (1 << 24))
            throw std::runtime_error("QuantizedNeuralNet::from_gguf: layer size out of range");
        Activation act = Activation::Identity;
        if (i < acts.size()) act = activation_from_string(acts[i]);
        std::vector<double> bias;
        tensor_to_doubles(r, *bt, bias);
        if (bias.size() != static_cast<std::size_t>(n_out))
            throw std::runtime_error("QuantizedNeuralNet::from_gguf: bias length mismatch");
        if (is_ter) {
            mlpternary::TLayer l;
            l.n_in = n_in;
            l.n_out = n_out;
            l.activation = act;
            l.bias = std::move(bias);
            // decode the stored weights
            const std::size_t stored = static_cast<std::size_t>(wt->n_elements());
            const std::size_t cols = static_cast<std::size_t>(wt->ne[0]);
            std::vector<float> x(stored);
            {
                std::vector<std::uint8_t> raw = r.tensor_bytes(*wt);
                gguf::decode_row(wt->type, raw.data(), static_cast<int>(stored), x.data());
            }
            const std::size_t keep = static_cast<std::size_t>(n_in);
            std::vector<double> wgt(keep * static_cast<std::size_t>(n_out));
            for (int o = 0; o < n_out; ++o)
                for (std::size_t k = 0; k < keep; ++k)
                    wgt[static_cast<std::size_t>(o) * keep + k] =
                        static_cast<double>(x[static_cast<std::size_t>(o) * cols + k]);
            if (wt->type == Type::F32) {
                l.full_precision = true;
                l.weight_f64 = std::move(wgt);
            } else {
                // quantise against the exact metadata gamma (not the fp16
                // block scales the file carried)
                l.weight.rows = n_out;
                l.weight.cols = n_in;
                l.weight.scale = tsc;
                double* W = wgt.data();
                mlpternary::quantize_into(W, n_out, n_in, tsc, l.weight);
                // overwrite the recomputed gamma with the stored one
                std::size_t want = tsc == mlpternary::Scale::Tensor ? 1 : static_cast<std::size_t>(n_out);
                if (gamma_counts != nullptr && gamma_all != nullptr) {
                    std::uint64_t got = gamma_counts->au32.size() > i ? gamma_counts->au32[i] : 0;
                    if (got != want)
                        throw std::runtime_error("QuantizedNeuralNet::from_gguf: ternary gamma count");
                    for (std::size_t g = 0; g < want; ++g)
                        l.weight.gamma[g] = gamma_all->af64[gamma_at + g];
                    gamma_at += want;
                }
                // trits must match the file's codes, not a re-quantisation:
                // the file's weights are q gamma exactly (encode_row wrote
                // q * fp16_block_scale though). Recover the trits from the
                // fp16-scaled values using the stored gamma.
                for (int o = 0; o < n_out; ++o) {
                    const double g = l.weight.gamma[tsc == mlpternary::Scale::Tensor ? 0 : static_cast<std::size_t>(o)];
                    for (int k = 0; k < n_in; ++k) {
                        const double v = wgt[static_cast<std::size_t>(o) * keep + k] / g;
                        l.weight.q[static_cast<std::size_t>(o) * keep + k] =
                            static_cast<std::int8_t>((v > 0.25 ? 1 : 0) - (v < -0.25 ? 1 : 0));
                    }
                }
            }
            tm.layers.push_back(std::move(l));
            continue;
        }
        if (is_int8) {
            mlpquant::QuantLayer l;
            l.n_in = n_in;
            l.n_out = n_out;
            l.activation = act;
            l.bias = std::move(bias);
            const std::size_t nw = static_cast<std::size_t>(n_in) * static_cast<std::size_t>(n_out);
            if (wt->n_elements() != static_cast<std::uint64_t>(nw))
                throw std::runtime_error("QuantizedNeuralNet::from_gguf: int8 weight size");
            std::vector<std::uint8_t> raw = r.tensor_bytes(*wt);
            l.weight.resize(nw);
            for (std::size_t k = 0; k < nw; ++k)
                l.weight[k] = static_cast<std::int8_t>(raw[k]);
            l.weight_scale = scales_v && static_cast<std::size_t>(i) < scales_v->af64.size()
                                 ? scales_v->af64[static_cast<std::size_t>(i)]
                                 : 1.0;
            q8.layers.push_back(std::move(l));
            continue;
        }
        // fp4 / mxfp4 / nvfp4
        mlpfp4::Fp4Layer l;
        l.n_in = n_in;
        l.n_out = n_out;
        l.activation = act;
        l.bias = std::move(bias);
        const std::size_t cols = static_cast<std::size_t>(wt->ne[0]);
        const std::size_t stored = static_cast<std::size_t>(wt->n_elements());
        std::vector<std::uint8_t> raw = r.tensor_bytes(*wt);
        std::vector<float> x(stored);
        gguf::decode_row(wt->type, raw.data(), static_cast<int>(stored), x.data());
        const std::size_t keep = static_cast<std::size_t>(n_in);
        std::vector<double> wgt(keep * static_cast<std::size_t>(n_out));
        for (int o = 0; o < n_out; ++o)
            for (std::size_t k = 0; k < keep; ++k)
                wgt[static_cast<std::size_t>(o) * keep + k] =
                    static_cast<double>(x[static_cast<std::size_t>(o) * cols + k]);
        if (is_fp4) {
            // bff fp4: the F32 payload is exactly on the fp4 grid (e2m1 *
            // row scale), so re-quantising returns the original codes and
            // row scales bit-for-bit — the layer stays quantised fp4
            l.weight = mlpfp4::quantize(wgt.data(), n_out, n_in, mlpfp4::Format::FP4);
        } else if (wt->type == Type::F32 || wt->type == Type::F16 ||
                   wt->type == Type::BF16 || wt->type == Type::F64) {
            // a kept float layer
            l.full_precision = true;
            l.weight_f64 = std::move(wgt);
        } else if (format == "mxfp4") {
            // re-encode from the dequantised values: mxfp4's value grid is
            // exact in float32, so the codes come back identically
            l.weight = mlpfp4::quantize(wgt.data(), n_out, n_in, mlpfp4::Format::MXFP4);
        } else if (format == "nvfp4") {
            // re-encode from the dequantised values; the requantised global
            // scale is consistent with the rebuilt block codes (the file's
            // folded scales served their readers, the metadata keeps the
            // original for reference)
            l.weight = mlpfp4::quantize(wgt.data(), n_out, n_in, mlpfp4::Format::NVFP4);
        } else {  // bff fp4: the F32 payload is exactly on the fp4 grid
            // (e2m1 * row scale), so quantising returns the original codes
            // and scales bit-for-bit
            l.weight = mlpfp4::quantize(wgt.data(), n_out, n_in, mlpfp4::Format::FP4);
        }
        f4.layers.push_back(std::move(l));
    }
    const StandardScaler xs{r.get_array_f64("bff.x_scaler.mean"), r.get_array_f64("bff.x_scaler.scale")};
    const StandardScaler ys{r.get_array_f64("bff.y_scaler.mean"), r.get_array_f64("bff.y_scaler.scale")};
    q8.x_scaler = f4.x_scaler = tm.x_scaler = xs;
    q8.y_scaler = f4.y_scaler = tm.y_scaler = ys;
    return format;
}

}  // namespace netgguf
}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_NETWORK_GGUF_H
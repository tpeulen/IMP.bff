/**
 *  \file IMP/bff/internal/GgufIo.h
 *  \brief The GGUF v3 container and the block codecs bff reads and writes.
 *
 * GGUF (github.com/ggml-org/ggml, docs/gguf.md) is llama.cpp's and MLX's
 * model file format: a magic + version header, key-value metadata, a table
 * of tensor infos, and raw tensor data aligned to `general.alignment`
 * (32 unless the file says otherwise). This header is a std-only writer and
 * reader of that container plus the block layouts of the tensor types bff
 * maps to its networks -- F32, F16, BF16, F64, Q8_0, Q4_0, Q1_0, MXFP4,
 * NVFP4, TQ1_0 and TQ2_0. The mapping between those and bff's network
 * formats lives in `NetworkGguf.h`; this file knows nothing about networks.
 *
 * Everything here follows the ggml reference implementations
 * (ggml-org/llama.cpp `ggml/src/ggml-common.h`, `ggml-quants.c`) and their
 * gguf-py ports, checked type by type:
 *
 *  - **F16** IEEE binary16, round to nearest even. **BF16** the upper half
 *    of an IEEE float32, RNE, NaN quieted. **F64** plain IEEE doubles.
 *  - **Q8_0** blocks of 32: `d = fp16(max|x| / 127)` (the absmax computed
 *    in float32), `q = rint(x / d)` in [-128, 127]. x = q d.
 *  - **Q4_0** blocks of 32, nibbles, ggml's quirk: `d = fp16(v / -8)` where
 *    `v` is the *signed* value of the largest-magnitude element, and
 *    `q = trunc(x / d + 8.5) & 0xF`; a byte holds element j (low) and
 *    j + 16 (high). x = (q - 8) d.
 *  - **Q1_0** blocks of 128, one bit per element set when `x >= 0`
 *    (element j is bit j of byte j / 8): a sign-only code with
 *    `d = fp16(mean |x|)` -- every element decodes to +-d. 1.125 bits a
 *    weight; bff's use is interchange with tools that want it.
 *  - **MXFP4** blocks of 32: one E8M0 byte (`floor(log2 max|x|) - 2 + 127`)
 *    and 16 bytes of E2M1 codes -- ggml pairs element j (low nibble) with
 *    j + 16 (high) of each block, *not* bff's 2i / 2i + 1; `NetworkGguf.h`
 *    converts. The multiplier of a code is E2M1 x 2^(floor(log2 max|x|)-2)
 *    (ggml stores doubled k-values against a halved scale, the same thing).
 *    Codes chosen by nearest of the sixteen candidates, computed in float32.
 *  - **NVFP4** super-blocks of 64 = four 16-element sub-blocks: four
 *    unsigned E4M3 scale bytes (bias 7, no sign) then 32 bytes of E2M1
 *    codes -- within a sub-block, element e lives in byte e % 8, low nibble
 *    for e < 8, high for e >= 8; the multiplier of a sub-block's codes is
 *    ue4m3(scale) x 2 (the doubled k-value convention), i.e. E2M1 x
 *    ue4m3_half(scale). No global scale: a writer that wants NVIDIA's
 *    tensor scale keeps it in metadata (bff writes `bff.nvfp4.tensor_scale`).
 *  - **TQ1_0** blocks of 256: 51 "qs" bytes then 4 "qh" bytes then fp16 d.
 *    Five trits a byte, `byte = ceil(v 256 / 243)` with
 *    `v = sum_n (t_n + 1) 3^(4-n)`; the trits of a byte are strided, not
 *    contiguous: bytes 0..31 hold elements m + 32n (m = 0..31, n = 0..4),
 *    bytes 32..47 elements 160 + m + 16n, qh[m] elements 240 + m + 4n
 *    (four trits, weights 3^(3-n)). `d = fp16(max |x|)` of the block.
 *  - **TQ2_0** blocks of 256: 64 bytes, two bits a element -- byte j holds
 *    elements 4j..4j+3 in bit pairs 0, 2, 4, 6 (code q + 1) -- then fp16
 *    `d = fp16(max |x|)`. A code of 3 is invalid.
 *
 *   A ternary code d of a block decodes from a stored byte b as
 *   `(((b 3^n) mod 256) 3) >> 8 - 1`, n the digit index (bff's own TQ1
 *   formula; it is the exact inverse of the ceil encoding above).
 *
 *   All quantisers take float32 values the way ggml does (a float64 weight
 *   is converted first); rint is round-half-to-even, like gguf-py's
 *   np_roundf. The little-endian container is read and written assuming a
 *   little-endian host, as ggml's own code does.
 *
 *  Sources: ggml-org/ggml docs/gguf.md (container, MIT); ggml-org/llama.cpp
 *  ggml-common.h and ggml-quants.c (block layouts and reference quantisers,
 *  MIT); ggml-org/gguf (the gguf-py package, MIT) whose dequantize/quantize
 *  these were checked against byte for byte. No code copied.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_GGUFIO_H
#define IMPBFF_INTERNAL_GGUFIO_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "IqTables.h"

namespace IMP {
namespace bff {
namespace internal {
namespace gguf {

//! The tensor types bff maps, by their GGUF v3 numbers.
enum class Type : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42
};

//! Elements a block covers and bytes a block occupies, per type.
inline void type_shape(Type t, int& elements, int& bytes) {
    switch (t) {
        case Type::F32: elements = 1; bytes = 4; return;
        case Type::F16: elements = 1; bytes = 2; return;
        case Type::BF16: elements = 1; bytes = 2; return;
        case Type::F64: elements = 1; bytes = 8; return;
        case Type::I8: elements = 1; bytes = 1; return;
        case Type::I16: elements = 1; bytes = 2; return;
        case Type::I32: elements = 1; bytes = 4; return;
        case Type::I64: elements = 1; bytes = 8; return;
        case Type::Q4_0: elements = 32; bytes = 18; return;
        case Type::Q4_1: elements = 32; bytes = 20; return;
        case Type::Q5_0: elements = 32; bytes = 22; return;
        case Type::Q5_1: elements = 32; bytes = 24; return;
        case Type::Q8_0: elements = 32; bytes = 34; return;
        case Type::Q8_1: elements = 32; bytes = 36; return;
        case Type::Q2_K: elements = 256; bytes = 84; return;
        case Type::Q3_K: elements = 256; bytes = 110; return;
        case Type::Q4_K: elements = 256; bytes = 144; return;
        case Type::Q5_K: elements = 256; bytes = 176; return;
        case Type::Q6_K: elements = 256; bytes = 210; return;
        case Type::Q8_K: elements = 256; bytes = 292; return;
        case Type::IQ2_XXS: elements = 256; bytes = 66; return;
        case Type::IQ2_XS: elements = 256; bytes = 74; return;
        case Type::IQ3_XXS: elements = 256; bytes = 98; return;
        case Type::IQ1_S: elements = 256; bytes = 50; return;
        case Type::IQ4_NL: elements = 32; bytes = 18; return;
        case Type::IQ3_S: elements = 256; bytes = 110; return;
        case Type::IQ2_S: elements = 256; bytes = 82; return;
        case Type::IQ4_XS: elements = 256; bytes = 136; return;
        case Type::Q2_0: elements = 64; bytes = 18; return;
        case Type::IQ1_M: elements = 256; bytes = 56; return;
        case Type::Q1_0: elements = 128; bytes = 18; return;   // 2 + 16
        case Type::MXFP4: elements = 32; bytes = 17; return;   // 1 + 16
        case Type::NVFP4: elements = 64; bytes = 36; return;   // 4 + 32
        case Type::TQ1_0: elements = 256; bytes = 54; return; // 48 + 4 + 2
        case Type::TQ2_0: elements = 256; bytes = 66; return;  // 64 + 2
    }
    throw std::runtime_error("gguf: unknown tensor type");
}

//! "F32", "q8_0", "mxfp4", ... (case-insensitive) -> Type; throws otherwise.
inline Type type_from_string(const std::string& name) {
    std::string s;
    for (char c : name) s.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
    if (s == "f32") return Type::F32;
    if (s == "f16") return Type::F16;
    if (s == "bf16") return Type::BF16;
    if (s == "f64") return Type::F64;
    if (s == "i8") return Type::I8;
    if (s == "i16") return Type::I16;
    if (s == "i32") return Type::I32;
    if (s == "i64") return Type::I64;
    if (s == "q8_0") return Type::Q8_0;
    if (s == "q8_1") return Type::Q8_1;
    if (s == "q5_0") return Type::Q5_0;
    if (s == "q5_1") return Type::Q5_1;
    if (s == "q2_k") return Type::Q2_K;
    if (s == "q3_k") return Type::Q3_K;
    if (s == "q4_k") return Type::Q4_K;
    if (s == "q5_k") return Type::Q5_K;
    if (s == "q6_k") return Type::Q6_K;
    if (s == "q8_k") return Type::Q8_K;
    if (s == "iq2_xxs") return Type::IQ2_XXS;
    if (s == "iq3_xxs") return Type::IQ3_XXS;
    if (s == "iq1_s") return Type::IQ1_S;
    if (s == "iq4_nl") return Type::IQ4_NL;
    if (s == "iq3_s") return Type::IQ3_S;
    if (s == "iq2_s") return Type::IQ2_S;
    if (s == "iq4_xs") return Type::IQ4_XS;
    if (s == "q2_0") return Type::Q2_0;
    if (s == "iq2_xs") return Type::IQ2_XS;
    if (s == "iq1_m") return Type::IQ1_M;
    if (s == "q4_0") return Type::Q4_0;
    if (s == "q4_1") return Type::Q4_1;
    if (s == "q1_0") return Type::Q1_0;
    if (s == "mxfp4") return Type::MXFP4;
    if (s == "nvfp4") return Type::NVFP4;
    if (s == "tq1_0") return Type::TQ1_0;
    if (s == "tq2_0") return Type::TQ2_0;
    throw std::runtime_error("gguf: unknown tensor type '" + name +
                             "' (F32, F16, BF16, F64, I8, Q8_0, Q4_0, Q1_0, "
                             "MXFP4, NVFP4, TQ1_0, TQ2_0, I16, I32, I64, Q4_1, Q5_0, Q5_1, Q8_1, IQ*)");
}

//! The canonical ggml spelling of a type.
inline std::string type_to_string(Type t) {
    switch (t) {
        case Type::F32: return "F32";
        case Type::F16: return "F16";
        case Type::BF16: return "BF16";
        case Type::F64: return "F64";
        case Type::I8: return "I8";
        case Type::I16: return "I16";
        case Type::I32: return "I32";
        case Type::I64: return "I64";
        case Type::Q4_0: return "Q4_0";
        case Type::Q4_1: return "Q4_1";
        case Type::Q5_0: return "Q5_0";
        case Type::Q5_1: return "Q5_1";
        case Type::Q8_0: return "Q8_0";
        case Type::Q8_1: return "Q8_1";
        case Type::Q2_K: return "Q2_K";
        case Type::Q3_K: return "Q3_K";
        case Type::Q4_K: return "Q4_K";
        case Type::Q5_K: return "Q5_K";
        case Type::Q6_K: return "Q6_K";
        case Type::Q8_K: return "Q8_K";
        case Type::IQ2_XXS: return "IQ2_XXS";
        case Type::IQ3_XXS: return "IQ3_XXS";
        case Type::IQ1_S: return "IQ1_S";
        case Type::IQ4_NL: return "IQ4_NL";
        case Type::IQ3_S: return "IQ3_S";
        case Type::IQ2_S: return "IQ2_S";
        case Type::IQ4_XS: return "IQ4_XS";
        case Type::Q2_0: return "Q2_0";
        case Type::IQ2_XS: return "IQ2_XS";
        case Type::IQ1_M: return "IQ1_M";
        case Type::Q1_0: return "Q1_0";
        case Type::MXFP4: return "MXFP4";
        case Type::NVFP4: return "NVFP4";
        case Type::TQ1_0: return "TQ1_0";
        case Type::TQ2_0: return "TQ2_0";
    }
    return "F32";
}

//! Whether the type stores float values one element at a time (no blocks).
inline bool is_plain(Type t) {
    return t == Type::F32 || t == Type::F16 || t == Type::BF16 || t == Type::F64 ||
           t == Type::I8 || t == Type::I16 || t == Type::I32 || t == Type::I64;
}

// ------------------------------------------------------------------ floats

//! IEEE float32 -> binary16, round to nearest even (ggml_fp32_to_fp16).
inline std::uint16_t f32_to_f16(float value) {
    std::uint32_t x;
    std::memcpy(&x, &value, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t rest = x & 0x7FFFFFFFu;
    if (rest >= 0x7F800000u)  // inf / nan
        return static_cast<std::uint16_t>(sign | 0x7C00u | (rest > 0x7F800000u ? 0x200u : 0));
    std::uint32_t be = rest >> 23;
    std::uint32_t m = (rest >> 13) & 0x3FFu;  // tentative mantissa
    if (be == 0) {                            // float32 zero / subnormal -> 0
        return static_cast<std::uint16_t>(sign);
    }
    if (be >= 0x71 && be <= 0x8E) {           // normal half range
        const std::uint32_t rem = rest & 0x1FFFu;
        const std::uint32_t half = 0x1000u;
        if (rem > half || (rem == half && (m & 1u))) ++m;  // RNE
        if (m == 0x400u) { m = 0; ++be; }
        return static_cast<std::uint16_t>(sign | ((be - 112) << 10) | m);
    }
    if (be < 0x71) {                          // -> half subnormal
        // value = rest / 2^133; keep the top bits with RNE at bit 10^-24
        const std::uint32_t shift = 126 - be + 14;  // >= 14
        const std::uint32_t mant = (rest & 0x7FFFFFu) | 0x800000u;
        std::uint32_t q = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;
        return static_cast<std::uint16_t>(sign | q);
    }
    return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow -> infinity
}

//! binary16 -> float32, exact.
inline float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t be = (h >> 10) & 0x1Fu;
    const std::uint32_t m = h & 0x3FFu;
    std::uint32_t x;
    if (be == 0) {
        if (m == 0) x = sign;
        else {  // subnormal half
            std::uint32_t e = -1;
            std::uint32_t v = m;
            while (!(v & 0x400u)) { v <<= 1; --e; }
            v &= 0x3FFu;
            x = sign | ((127 - 15 + e + 1) << 23) | (v << 13);
        }
    } else if (be == 0x1F) {
        x = sign | 0x7F800000u | (m << 13);
    } else {
        x = sign | ((be - 15 + 127) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &x, 4);
    return out;
}

//! float32 -> bf16, round to nearest even, NaN quieted (ggml convention).
inline std::uint16_t f32_to_bf16(float value) {
    std::uint32_t x;
    std::memcpy(&x, &value, 4);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u)  // NaN -> quiet
        return static_cast<std::uint16_t>((x >> 16) | 0x0040u);
    std::uint32_t bias = 0x7FFFu + ((x >> 16) & 1u);  // RNE
    return static_cast<std::uint16_t>((x + bias) >> 16);
}

//! bf16 -> float32, exact.
inline float bf16_to_f32(std::uint16_t h) {
    std::uint32_t x = static_cast<std::uint32_t>(h) << 16;
    float out;
    std::memcpy(&out, &x, 4);
    return out;
}

// ------------------------------------------------------------------ LE ints

inline void put_u16(std::vector<std::uint8_t>& o, std::uint16_t v) {
    o.push_back(static_cast<std::uint8_t>(v));
    o.push_back(static_cast<std::uint8_t>(v >> 8));
}
inline void put_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
inline void put_u64(std::vector<std::uint8_t>& o, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
inline void put_f32(std::vector<std::uint8_t>& o, float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    put_u32(o, u);
}
inline void put_f64(std::vector<std::uint8_t>& o, double v) {
    std::uint64_t u;
    std::memcpy(&u, &v, 8);
    put_u64(o, u);
}

//! A bounds-checked little-endian cursor over a file image.
struct Cursor {
    const std::uint8_t* p;
    std::size_t n;
    std::size_t at = 0;
    Cursor(const std::uint8_t* data, std::size_t size) : p(data), n(size) {}
    void need(std::size_t k) const {
        if (at + k > n) throw std::runtime_error("gguf: truncated file");
    }
    std::uint8_t u8() {
        need(1);
        return p[at++];
    }
    std::uint16_t u16() {
        need(2);
        std::uint16_t v = static_cast<std::uint16_t>(p[at] | (p[at + 1] << 8));
        at += 2;
        return v;
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p[at + i]) << (8 * i);
        at += 4;
        return v;
    }
    std::uint64_t u64() {
        need(8);
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[at + i]) << (8 * i);
        at += 8;
        return v;
    }
    float f32() {
        const std::uint32_t u = u32();
        float v;
        std::memcpy(&v, &u, 4);
        return v;
    }
    double f64() {
        const std::uint64_t u = u64();
        double v;
        std::memcpy(&v, &u, 8);
        return v;
    }
    std::uint64_t raw(const std::uint8_t*& out, std::size_t k) {
        need(k);
        out = p + at;
        at += k;
        return k;
    }
};

// ------------------------------------------------------------------ codecs

//! Quantisers and dequantisers, one row at a time.
/*! Every function covers exactly `n` elements, `n` a multiple of the
    type's block size; the caller pads. Quantise takes float32 values (a
    float64 weight is converted by the caller) and writes `out_bytes(n)`;
    dequantise reads it and writes `n` float32 values. Byte layouts are the
    ggml ones -- including MXFP4's j / j + 16 nibble pairing and TQ1_0's
    strided trits -- so these write and read interchange bytes directly. */

inline std::size_t row_bytes(Type t, int n) {
    int e, b;
    type_shape(t, e, b);
    return static_cast<std::size_t>(n / e) * b;
}

namespace detail {

inline std::uint16_t f16_of(float v) { return f32_to_f16(v); }
inline float half_of(std::uint16_t h) { return f16_to_f32(h); }
inline float rintf32(float v) { return std::rint(v); }  // half to even

// ---- Q8_0
inline void q8_0_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 32, x += 32, out += 34) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        const std::uint16_t h = f16_of(d);
        out[0] = static_cast<std::uint8_t>(h & 0xFF);
        out[1] = static_cast<std::uint8_t>(h >> 8);
        for (int j = 0; j < 32; ++j) {
            float q = rintf32(x[j] * id);
            q = std::max(-128.f, std::min(127.f, q));
            out[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
    }
}
inline void q8_0_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 32, x += 32, in += 34) {
        const std::uint16_t h = static_cast<std::uint16_t>(in[0] | (in[1] << 8));
        const float d = half_of(h);
        for (int j = 0; j < 32; ++j)
            x[j] = static_cast<float>(static_cast<std::int8_t>(in[2 + j])) * d;
    }
}

// ---- Q8_1: Q8_0 payload with an additional fp16 d*sum(q) field.
inline void q8_1_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 32, x += 32, out += 36) {
        float amax = 0.f; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f/d : 0.f;
        const std::uint16_t dh = f16_of(d); out[0] = static_cast<std::uint8_t>(dh); out[1] = static_cast<std::uint8_t>(dh >> 8);
        int sum = 0;
        for (int j = 0; j < 32; ++j) { int q = static_cast<int>(std::round(x[j]*id)); q = std::max(-128, std::min(127, q)); out[4+j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q)); sum += q; }
        const std::uint16_t sh = f16_of(sum*d); out[2] = static_cast<std::uint8_t>(sh); out[3] = static_cast<std::uint8_t>(sh >> 8);
    }
}
inline void q8_1_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 32, x += 32, in += 36) { const float d = half_of(static_cast<std::uint16_t>(in[0] | (in[1] << 8))); for (int j = 0; j < 32; ++j) x[j] = d * static_cast<float>(static_cast<std::int8_t>(in[4+j])); }
}

// ---- Q2_K: sixteen affine 16-value groups inside a 256-value super-block.
inline void q2_k_encode(const float* x, int n, std::uint8_t* out) {
    for (int base = 0; base < n; base += 256, x += 256, out += 84) {
        float ds[16], ms[16], maxd = 0.f, maxm = 0.f;
        for (int g = 0; g < 16; ++g) { float lo=x[16*g], hi=lo; for (int j=1;j<16;++j) { lo=std::min(lo,x[16*g+j]); hi=std::max(hi,x[16*g+j]); } ds[g]=(hi-lo)/3.f; ms[g]=std::max(0.f,-lo); maxd=std::max(maxd,ds[g]); maxm=std::max(maxm,ms[g]); }
        const float d=maxd/15.f, dm=maxm/15.f; const std::uint16_t dh=f16_of(d), mh=f16_of(dm);
        for (int g=0;g<16;++g) { const int sd=d>0?std::max(0,std::min(15,static_cast<int>(std::round(ds[g]/d)))):0; const int sm=dm>0?std::max(0,std::min(15,static_cast<int>(std::round(ms[g]/dm)))):0; out[g]=static_cast<std::uint8_t>(sd | (sm<<4)); }
        std::fill(out+16,out+80,0); for (int g=0;g<16;++g) { const float dg=d*(out[g]&15), mg=dm*(out[g]>>4); for(int j=0;j<16;++j) { const int q=dg>0?std::max(0,std::min(3,static_cast<int>(std::round((x[16*g+j]+mg)/dg)))):0; const int chunk=g/8, sub=g%8, l=(sub%2)*16+j; out[16+chunk*32+l] |= static_cast<std::uint8_t>(q << (2*(sub/2))); } }
        out[80]=static_cast<std::uint8_t>(dh); out[81]=static_cast<std::uint8_t>(dh>>8); out[82]=static_cast<std::uint8_t>(mh); out[83]=static_cast<std::uint8_t>(mh>>8);
    }
}
inline void q2_k_decode(const std::uint8_t* in, int n, float* x) {
    for (int base=0;base<n;base+=256,x+=256,in+=84) { const float d=half_of(static_cast<std::uint16_t>(in[80]|(in[81]<<8))), dm=half_of(static_cast<std::uint16_t>(in[82]|(in[83]<<8))); for(int g=0;g<16;++g) { const float dg=d*(in[g]&15), mg=dm*(in[g]>>4); const int chunk=g/8, sub=g%8; for(int j=0;j<16;++j) { const int l=(sub%2)*16+j; const int q=(in[16+chunk*32+l]>>(2*(sub/2)))&3; x[16*g+j]=dg*q-mg; } } }
}

// ---- Q3_K: sixteen signed 3-bit 16-value groups; 6-bit scales share d.
inline void q3_k_encode(const float* x, int n, std::uint8_t* out) {
    for (int base=0;base<n;base+=256,x+=256,out+=110) { float gs[16], mx=0.f; for(int g=0;g<16;++g){ float a=0.f; for(int j=0;j<16;++j)a=std::max(a,std::fabs(x[16*g+j])); gs[g]=a/3.f; mx=std::max(mx,gs[g]); } const float d=mx/31.f; std::fill(out,out+110,0); for(int g=0;g<16;++g){ const int sc=d>0?std::max(1,std::min(31,static_cast<int>(std::round(gs[g]/d)))):0; const int raw=sc+32; if(g<8) out[96+g]|=raw&15; else out[96+g-8]|=(raw&15)<<4; out[104+g%4]|=static_cast<std::uint8_t>((raw>>4)<<(2*(g/4))); for(int j=0;j<16;++j){ int q=sc?std::max(-4,std::min(3,static_cast<int>(std::round(x[16*g+j]/(d*sc))))):0; int code=q+4, p=16*g+j; if(code>3){out[p%32]|=static_cast<std::uint8_t>(1u<<(p/32));code-=4;} const int chunk=p/128, l=p%32; out[32+chunk*32+l]|=static_cast<std::uint8_t>(code<<(2*((p%128)/32))); } } const std::uint16_t h=f16_of(d);out[108]=static_cast<std::uint8_t>(h);out[109]=static_cast<std::uint8_t>(h>>8); }
}
inline void q3_k_decode(const std::uint8_t* in, int n, float* x) {
    for(int base=0;base<n;base+=256,x+=256,in+=110){ const float d=half_of(static_cast<std::uint16_t>(in[108]|(in[109]<<8))); for(int g=0;g<16;++g){ const int raw=(g<8?(in[96+g]&15):(in[96+g-8]>>4)) | (((in[104+g%4]>>(2*(g/4)))&3)<<4); const float s=d*(raw-32); for(int j=0;j<16;++j){const int p=16*g+j,chunk=p/128,l=p%32; int q=(in[32+chunk*32+l]>>(2*((p%128)/32)))&3; if(!(in[p%32]&(1u<<(p/32))))q-=4; x[p]=s*q;}}}
}

// ---- Q4_K: eight affine 32-value groups, six-bit scale and minimum codes.
inline void q4_k_encode(const float* x, int n, std::uint8_t* out) {
    for(int base=0;base<n;base+=256,x+=256,out+=144){ float ds[8],ms[8],md=0.f,mm=0.f; for(int g=0;g<8;++g){float lo=x[32*g],hi=lo;for(int j=1;j<32;++j){lo=std::min(lo,x[32*g+j]);hi=std::max(hi,x[32*g+j]);}ds[g]=(hi-lo)/15.f;ms[g]=std::max(0.f,-lo);md=std::max(md,ds[g]);mm=std::max(mm,ms[g]);}const float d=md/63.f,dm=mm/63.f;std::fill(out,out+144,0);for(int g=0;g<8;++g){const int sd=d?std::max(0,std::min(63,static_cast<int>(std::round(ds[g]/d)))):0,sm=dm?std::max(0,std::min(63,static_cast<int>(std::round(ms[g]/dm)))):0;if(g<4){out[4+g]=sd;out[8+g]=sm;}else{out[8+g]=(sd&15)|((sm&15)<<4);out[4+g-4]|=(sd>>4)<<6;out[4+g]|=(sm>>4)<<6;}}for(int g=0;g<8;++g){const int sd=(g<4?out[4+g]&63:((out[8+g]&15)|((out[g]>>6)<<4))),sm=(g<4?out[8+g]&63:((out[8+g]>>4)|((out[4+g]>>6)<<4)));const float dg=d*sd,mg=dm*sm;for(int j=0;j<32;++j){const int q=dg?std::max(0,std::min(15,static_cast<int>(std::round((x[32*g+j]+mg)/dg)))):0;const int pair=g/2,l=(g%2)*32+j;out[16+pair*32+(l%32)]|=static_cast<std::uint8_t>(q<<(4*(g%2)));}}const std::uint16_t dh=f16_of(d),mh=f16_of(dm);out[0]=static_cast<std::uint8_t>(dh);out[1]=static_cast<std::uint8_t>(dh>>8);out[2]=static_cast<std::uint8_t>(mh);out[3]=static_cast<std::uint8_t>(mh>>8);}
}
inline void q4_k_decode(const std::uint8_t* in,int n,float* x){for(int base=0;base<n;base+=256,x+=256,in+=144){const float d=half_of(static_cast<std::uint16_t>(in[0]|(in[1]<<8))),dm=half_of(static_cast<std::uint16_t>(in[2]|(in[3]<<8)));for(int g=0;g<8;++g){const int sd=g<4?in[4+g]&63:((in[8+g]&15)|((in[g]>>6)<<4)),sm=g<4?in[8+g]&63:((in[8+g]>>4)|((in[4+g]>>6)<<4));for(int j=0;j<32;++j){const int pair=g/2;const int q=(in[16+pair*32+j]>>(4*(g%2)))&15;x[32*g+j]=d*sd*q-dm*sm;}}}}

// ---- Q5_K: Q4_K affine groups with one high-bit plane per quant.
inline void q5_k_encode(const float* x,int n,std::uint8_t*out){for(int base=0;base<n;base+=256,x+=256,out+=176){float ds[8],ms[8],md=0,mm=0;for(int g=0;g<8;++g){float lo=x[32*g],hi=lo;for(int j=1;j<32;++j){lo=std::min(lo,x[32*g+j]);hi=std::max(hi,x[32*g+j]);}ds[g]=(hi-lo)/31.f;ms[g]=std::max(0.f,-lo);md=std::max(md,ds[g]);mm=std::max(mm,ms[g]);}const float d=md/63.f,dm=mm/63.f;std::fill(out,out+176,0);for(int g=0;g<8;++g){const int sd=d?std::max(0,std::min(63,static_cast<int>(std::round(ds[g]/d)))):0,sm=dm?std::max(0,std::min(63,static_cast<int>(std::round(ms[g]/dm)))):0;if(g<4){out[4+g]=sd;out[8+g]=sm;}else{out[8+g]=(sd&15)|((sm&15)<<4);out[g]|=(sd>>4)<<6;out[4+g]|=(sm>>4)<<6;}}for(int g=0;g<8;++g){const int sd=g<4?out[4+g]&63:((out[8+g]&15)|((out[g]>>6)<<4)),sm=g<4?out[8+g]&63:((out[8+g]>>4)|((out[4+g]>>6)<<4));const float dg=d*sd,mg=dm*sm;for(int j=0;j<32;++j){const int q=dg?std::max(0,std::min(31,static_cast<int>(std::round((x[32*g+j]+mg)/dg)))):0,pair=g/2;out[48+pair*32+j]|=static_cast<std::uint8_t>((q&15)<<(4*(g%2)));if(q&16)out[16+j]|=static_cast<std::uint8_t>(1u<<(2*pair+(g%2)));}}const std::uint16_t dh=f16_of(d),mh=f16_of(dm);out[0]=static_cast<std::uint8_t>(dh);out[1]=static_cast<std::uint8_t>(dh>>8);out[2]=static_cast<std::uint8_t>(mh);out[3]=static_cast<std::uint8_t>(mh>>8);}}
inline void q5_k_decode(const std::uint8_t*in,int n,float*x){for(int base=0;base<n;base+=256,x+=256,in+=176){const float d=half_of(static_cast<std::uint16_t>(in[0]|(in[1]<<8))),dm=half_of(static_cast<std::uint16_t>(in[2]|(in[3]<<8)));for(int g=0;g<8;++g){const int sd=g<4?in[4+g]&63:((in[8+g]&15)|((in[g]>>6)<<4)),sm=g<4?in[8+g]&63:((in[8+g]>>4)|((in[4+g]>>6)<<4)),pair=g/2;for(int j=0;j<32;++j){int q=(in[48+pair*32+j]>>(4*(g%2)))&15;if(in[16+j]&(1u<<(2*pair+(g%2))))q|=16;x[32*g+j]=d*sd*q-dm*sm;}}}}

// ---- Q6_K: signed six-bit groups; low nibbles plus a two-bit plane.
inline void q6_k_encode(const float*x,int n,std::uint8_t*out){for(int base=0;base<n;base+=256,x+=256,out+=210){float gs[16],mx=0;for(int g=0;g<16;++g){float a=0;for(int j=0;j<16;++j)a=std::max(a,std::fabs(x[16*g+j]));gs[g]=a/31.f;mx=std::max(mx,gs[g]);}const float d=mx/127.f;std::fill(out,out+210,0);for(int g=0;g<16;++g){const int sc=d?std::max(1,std::min(127,static_cast<int>(std::round(gs[g]/d)))):0;out[192+g]=static_cast<std::uint8_t>(static_cast<std::int8_t>(sc));for(int j=0;j<16;++j){int q=sc?std::max(-32,std::min(31,static_cast<int>(std::round(x[16*g+j]/(d*sc))))):0;const int code=q+32,p=16*g+j,chunk=p/128,l=p%32,slot=(p%128)/32;const int qli=chunk*64+l+(slot==1||slot==3?32:0);out[qli]|=static_cast<std::uint8_t>((code&15)<<(slot>=2?4:0));out[128+chunk*32+l]|=static_cast<std::uint8_t>((code>>4)<<(2*slot));}}const std::uint16_t h=f16_of(d);out[208]=static_cast<std::uint8_t>(h);out[209]=static_cast<std::uint8_t>(h>>8);}}
inline void q6_k_decode(const std::uint8_t*in,int n,float*x){for(int base=0;base<n;base+=256,x+=256,in+=210){const float d=half_of(static_cast<std::uint16_t>(in[208]|(in[209]<<8)));for(int p=0;p<256;++p){const int chunk=p/128,l=p%32,slot=(p%128)/32,qli=chunk*64+l+(slot==1||slot==3?32:0);const int code=((in[qli]>>(slot>=2?4:0))&15)|(((in[128+chunk*32+l]>>(2*slot))&3)<<4);x[p]=d*static_cast<std::int8_t>(in[192+p/16])*(code-32);}}}

// ---- Q8_K: float32 scale, int8 codes, and int16 sums for 16-value groups.
inline void q8_k_encode(const float*x,int n,std::uint8_t*out){for(int base=0;base<n;base+=256,x+=256,out+=292){float a=0;for(int i=0;i<256;++i)a=std::max(a,std::fabs(x[i]));const float d=a/127.f;std::uint32_t du;std::memcpy(&du,&d,4);for(int b=0;b<4;++b)out[b]=static_cast<std::uint8_t>(du>>(8*b));for(int g=0;g<16;++g){int sum=0;for(int j=0;j<16;++j){int q=d?std::max(-128,std::min(127,static_cast<int>(std::round(x[16*g+j]/d)))):0;out[4+16*g+j]=static_cast<std::uint8_t>(static_cast<std::int8_t>(q));sum+=q;}const std::uint16_t s=static_cast<std::uint16_t>(static_cast<std::int16_t>(sum));out[260+2*g]=static_cast<std::uint8_t>(s);out[261+2*g]=static_cast<std::uint8_t>(s>>8);}}}
inline void q8_k_decode(const std::uint8_t*in,int n,float*x){for(int base=0;base<n;base+=256,x+=256,in+=292){std::uint32_t u=0;for(int b=0;b<4;++b)u|=static_cast<std::uint32_t>(in[b])<<(8*b);float d;std::memcpy(&d,&u,4);for(int i=0;i<256;++i)x[i]=d*static_cast<float>(static_cast<std::int8_t>(in[4+i]));}}

// ---- IQ2_XXS: 2-bit importance-grid quants (ggml block_iq2_xxs).
inline std::uint8_t iq2_signs(std::uint8_t index) {
    // gguf-py's IQ2_XXS.ksigns stores the seven low sign bits and makes
    // the eighth bit their parity (an even number of negative signs).
    std::uint8_t parity = 0;
    for (int bit = 0; bit < 7; ++bit) parity ^= (index >> bit) & 1u;
    return static_cast<std::uint8_t>(index | (parity << 7));
}
inline void iq2_xxs_decode(const std::uint8_t*in,int n,float*x){for(;n;n-=256,in+=66,x+=256){const float d=half_of(static_cast<std::uint16_t>(in[0]|(in[1]<<8)));for(int b=0;b<8;++b){std::uint32_t lo=0,hi=0;for(int j=0;j<4;++j){lo|=static_cast<std::uint32_t>(in[2+8*b+j])<<(8*j);hi|=static_cast<std::uint32_t>(in[6+8*b+j])<<(8*j);}const float db=d*(.5f+(hi>>28))*.25f;for(int l=0;l<4;++l){const std::uint64_t grid=kIq2XxsGrid[(lo>>(8*l))&255];const std::uint8_t signs=iq2_signs((hi>>(7*l))&127);for(int j=0;j<8;++j){const float v=static_cast<float>((grid>>(8*j))&255);x[32*b+8*l+j]=(signs&(1u<<j))?-db*v:db*v;}}}}}
inline void iq2_xxs_encode(const float*x,int n,std::uint8_t*out){for(;n;n-=256,x+=256,out+=66){float a=0;for(int i=0;i<256;++i)a=std::max(a,std::fabs(x[i]));const std::uint16_t dh=f16_of(a/166.625f);out[0]=static_cast<std::uint8_t>(dh);out[1]=static_cast<std::uint8_t>(dh>>8);const float d=half_of(dh), db=d*3.875f;for(int b=0;b<8;++b){std::uint32_t lo=0,hi=15u<<28;for(int l=0;l<4;++l){const float* v=x+32*b+8*l;float best=std::numeric_limits<float>::infinity();int bi=0,bs=0;for(int g=0;g<256;++g){const std::uint64_t grid=kIq2XxsGrid[g];for(int s=0;s<128;++s){const std::uint8_t sign=iq2_signs(static_cast<std::uint8_t>(s));float e=0;for(int j=0;j<8;++j){float q=db*static_cast<float>((grid>>(8*j))&255);if(sign&(1u<<j))q=-q;const float z=v[j]-q;e+=z*z;}if(e<best){best=e;bi=g;bs=s;}}}lo|=static_cast<std::uint32_t>(bi)<<(8*l);hi|=static_cast<std::uint32_t>(bs)<<(7*l);}for(int j=0;j<4;++j){out[2+8*b+j]=static_cast<std::uint8_t>(lo>>(8*j));out[6+8*b+j]=static_cast<std::uint8_t>(hi>>(8*j));}}}}

// ---- IQ2_XS: 2-bit importance-grid quants with paired 4-bit scales.
inline void iq2_xs_decode(const std::uint8_t*in,int n,float*x){for(;n;n-=256,in+=74,x+=256){const float d=half_of(static_cast<std::uint16_t>(in[0]|(in[1]<<8)));const std::uint8_t*qs=in+2;const std::uint8_t*sc=in+66;for(int b=0;b<8;++b){const float db[2]={d*(.5f+(sc[b]&15))*.25f,d*(.5f+(sc[b]>>4))*.25f};for(int l=0;l<4;++l){const std::uint16_t q=static_cast<std::uint16_t>(qs[8*b+2*l]|(qs[8*b+2*l+1]<<8));const std::uint64_t grid=kIq2XsGrid[q&511];const std::uint8_t signs=iq2_signs(q>>9);for(int j=0;j<8;++j){float z=db[l/2]*static_cast<float>((grid>>(8*j))&255);x[32*b+8*l+j]=(signs&(1u<<j))?-z:z;}}}}}
inline void iq2_xs_encode(const float*x,int n,std::uint8_t*out){for(;n;n-=256,x+=256,out+=74){float a=0;for(int i=0;i<256;++i)a=std::max(a,std::fabs(x[i]));const std::uint16_t dh=f16_of(a/166.625f);out[0]=static_cast<std::uint8_t>(dh);out[1]=static_cast<std::uint8_t>(dh>>8);const float d=half_of(dh),db=d*3.875f;for(int b=0;b<8;++b){out[66+b]=255;for(int l=0;l<4;++l){const float*v=x+32*b+8*l;float best=std::numeric_limits<float>::infinity();int bi=0,bs=0;for(int g=0;g<512;++g){const std::uint64_t grid=kIq2XsGrid[g];for(int s=0;s<128;++s){const std::uint8_t sign=iq2_signs(static_cast<std::uint8_t>(s));float e=0;for(int j=0;j<8;++j){float q=db*static_cast<float>((grid>>(8*j))&255);if(sign&(1u<<j))q=-q;float z=v[j]-q;e+=z*z;}if(e<best){best=e;bi=g;bs=s;}}}const std::uint16_t q=static_cast<std::uint16_t>(bi|(bs<<9));out[2+8*b+2*l]=static_cast<std::uint8_t>(q);out[3+8*b+2*l]=static_cast<std::uint8_t>(q>>8);}}}}

// Remaining IQ formats are import-only. Their grid and field layouts follow
// gguf-py quants.py and ggml's block_iq* definitions.
inline std::uint16_t iq_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
inline std::uint32_t iq_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(iq_u16(p)) |
           (static_cast<std::uint32_t>(iq_u16(p + 2)) << 16);
}
inline float iq_grid8(std::uint64_t grid, int j) {
    return static_cast<float>((grid >> (8 * j)) & 0xffu);
}
inline float iq_grid4(std::uint32_t grid, int j) {
    return static_cast<float>((grid >> (8 * j)) & 0xffu);
}

inline void iq3_xxs_decode(const std::uint8_t* in, int n, float* x) {
    for (; n; n -= 256, in += 98, x += 256) {
        const float d = half_of(iq_u16(in));
        const std::uint8_t* qs = in + 2;
        const std::uint8_t* scales = in + 66;
        for (int b = 0; b < 8; ++b) {
            const std::uint32_t word = iq_u32(scales + 4 * b);
            const float db = d * (0.5f + static_cast<float>(word >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = iq2_signs((word >> (7 * l)) & 127u);
                for (int j = 0; j < 8; ++j) {
                    const std::uint32_t grid = kIq3XxsGrid[qs[8 * b + 2 * l + j / 4]];
                    const float value = db * iq_grid4(grid, j % 4);
                    x[32 * b + 8 * l + j] = (signs & (1u << j)) ? -value : value;
                }
            }
        }
    }
}

inline void iq1_s_decode(const std::uint8_t* in, int n, float* x) {
    for (; n; n -= 256, in += 50, x += 256) {
        const float d = half_of(iq_u16(in));
        const std::uint8_t* qs = in + 2;
        const std::uint8_t* qh = in + 34;
        for (int b = 0; b < 8; ++b) {
            const std::uint16_t high = iq_u16(qh + 2 * b);
            const float db = d * static_cast<float>(2 * ((high >> 12) & 7u) + 1);
            const float delta = (high & 0x8000u) ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                const int index = qs[4 * b + l] | (((high >> (3 * l)) & 7u) << 8);
                const std::uint64_t grid = kIq1SGrid[index];
                for (int j = 0; j < 8; ++j)
                    x[32 * b + 8 * l + j] = db *
                        (static_cast<float>(static_cast<std::int8_t>((grid >> (8 * j)) & 255u)) + delta);
            }
        }
    }
}

inline void iq4_nl_decode(const std::uint8_t* in, int n, float* x) {
    static constexpr std::int8_t values[16] =
        {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (; n; n -= 32, in += 18, x += 32) {
        const float d = half_of(iq_u16(in));
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t q = in[2 + j];
            x[j] = d * static_cast<float>(values[q & 15u]);
            x[j + 16] = d * static_cast<float>(values[q >> 4]);
        }
    }
}

inline void iq3_s_decode(const std::uint8_t* in, int n, float* x) {
    for (; n; n -= 256, in += 110, x += 256) {
        const float d = half_of(iq_u16(in));
        const std::uint8_t* qs = in + 2;
        const std::uint8_t* qh = in + 66;
        const std::uint8_t* signs = in + 74;
        const std::uint8_t* scales = in + 106;
        for (int b = 0; b < 8; ++b) {
            const int scale = (scales[b / 2] >> (4 * (b % 2))) & 15;
            const float db = d * static_cast<float>(1 + 2 * scale);
            for (int l = 0; l < 8; ++l) {
                const int group = 8 * b + l;
                const int index = qs[group] | (((qh[group / 8] >> (group % 8)) & 1u) << 8);
                const std::uint32_t grid = kIq3SGrid[index];
                for (int j = 0; j < 4; ++j) {
                    const float value = db * iq_grid4(grid, j);
                    x[32 * b + 4 * l + j] = (signs[4 * b + l / 2] & (1u << (4 * (l % 2) + j))) ? -value : value;
                }
            }
        }
    }
}

inline void iq2_s_decode(const std::uint8_t* in, int n, float* x) {
    for (; n; n -= 256, in += 82, x += 256) {
        const float d = half_of(iq_u16(in));
        const std::uint8_t* qs = in + 2;
        const std::uint8_t* signs = in + 34;
        const std::uint8_t* qh = in + 66;
        const std::uint8_t* scales = in + 74;
        for (int b = 0; b < 16; ++b) {
            const int scale = (scales[b / 2] >> (4 * (b % 2))) & 15;
            const float db = d * (0.5f + static_cast<float>(scale)) * 0.25f;
            for (int l = 0; l < 2; ++l) {
                const int group = 2 * b + l;
                const int index = qs[group] | (((qh[group / 4] >> (2 * (group % 4))) & 3u) << 8);
                const std::uint64_t grid = kIq2SGrid[index];
                for (int j = 0; j < 8; ++j) {
                    const float value = db * iq_grid8(grid, j);
                    x[16 * b + 8 * l + j] = (signs[group] & (1u << j)) ? -value : value;
                }
            }
        }
    }
}

inline void iq4_xs_decode(const std::uint8_t* in, int n, float* x) {
    static constexpr std::int8_t values[16] =
        {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (; n; n -= 256, in += 136, x += 256) {
        const float d = half_of(iq_u16(in));
        const std::uint16_t scales_h = iq_u16(in + 2);
        const std::uint8_t* scales_l = in + 4;
        const std::uint8_t* qs = in + 8;
        for (int b = 0; b < 8; ++b) {
            const int scale = ((scales_l[b / 2] >> (4 * (b % 2))) & 15) |
                              (((scales_h >> (2 * b)) & 3) << 4);
            const float db = d * static_cast<float>(scale - 32);
            for (int j = 0; j < 16; ++j) {
                const std::uint8_t q = qs[16 * b + j];
                x[32 * b + j] = db * static_cast<float>(values[q & 15u]);
                x[32 * b + 16 + j] = db * static_cast<float>(values[q >> 4]);
            }
        }
    }
}

inline void iq1_m_decode(const std::uint8_t* in, int n, float* x) {
    for (; n; n -= 256, in += 56, x += 256) {
        const std::uint8_t* qs = in;
        const std::uint8_t* qh = in + 32;
        const std::uint8_t* scales = in + 48;
        const std::uint16_t s0 = iq_u16(scales), s1 = iq_u16(scales + 2);
        const std::uint16_t s2 = iq_u16(scales + 4), s3 = iq_u16(scales + 6);
        const std::uint16_t dh = static_cast<std::uint16_t>(
            ((s0 >> 12) & 15u) | ((s1 >> 8) & 0xf0u) |
            ((s2 >> 4) & 0xf00u) | (s3 & 0xf000u));
        const float d = half_of(dh);
        for (int b = 0; b < 16; ++b) {
            const std::uint16_t packed = iq_u16(scales + 2 * (b / 4));
            const int scale = (packed >> (3 * (b % 4))) & 7;
            const float db = d * static_cast<float>(2 * scale + 1);
            for (int l = 0; l < 2; ++l) {
                const int group = 2 * b + l;
                const int high = (qh[group / 2] >> (4 * (group % 2))) & 15;
                const int index = qs[group] | ((high & 7) << 8);
                const float delta = (high & 8) ? -0.125f : 0.125f;
                const std::uint64_t grid = kIq1SGrid[index];
                for (int j = 0; j < 8; ++j)
                    x[16 * b + 8 * l + j] = db *
                        (static_cast<float>(static_cast<std::int8_t>((grid >> (8 * j)) & 255u)) + delta);
            }
        }
    }
}

// ---- Q2_0: GGML's 64-value two-bit block.
inline void q2_0_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 64, x += 64, out += 18) {
        float amax = 0.f;
        for (int j = 0; j < 64; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float id = amax > 0.f ? 1.f / amax : 0.f;
        const std::uint16_t h = f16_of(amax);
        out[0] = static_cast<std::uint8_t>(h);
        out[1] = static_cast<std::uint8_t>(h >> 8);
        std::fill(out + 2, out + 18, 0);
        for (int j = 0; j < 64; ++j) {
            const int q = std::max(0, std::min(3, static_cast<int>(std::round(x[j] * id)) + 1));
            out[2 + j / 4] |= static_cast<std::uint8_t>(q << (2 * (j % 4)));
        }
    }
}
inline void q2_0_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 64, x += 64, in += 18) {
        const float d = half_of(static_cast<std::uint16_t>(in[0] | (in[1] << 8)));
        for (int j = 0; j < 64; ++j) x[j] = d * (((in[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
    }
}

// ---- Q4_0 (ggml's signed-max negative-delta quirk, j / j + 16 pairing)
inline void q4_0_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 32, x += 32, out += 18) {
        float amax = 0.f;
        float vmax = 0.f;
        for (int j = 0; j < 32; ++j) {
            const float a = std::fabs(x[j]);
            if (a > amax) { amax = a; vmax = x[j]; }
        }
        const float d = vmax / -8.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        const std::uint16_t h = f16_of(d);
        out[0] = static_cast<std::uint8_t>(h & 0xFF);
        out[1] = static_cast<std::uint8_t>(h >> 8);
        std::fill(out + 2, out + 18, static_cast<std::uint8_t>(0));
        for (int j = 0; j < 32; ++j) {
            float v = std::trunc(x[j] * id + 8.5f);
            int q = static_cast<int>(v);
            q = std::max(0, std::min(15, q));
            const int byte = j < 16 ? j : j - 16;
            if (j < 16) out[2 + byte] |= static_cast<std::uint8_t>(q);
            else out[2 + byte] |= static_cast<std::uint8_t>(q << 4);
        }
    }
}
inline void q4_0_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 32, x += 32, in += 18) {
        const std::uint16_t h = static_cast<std::uint16_t>(in[0] | (in[1] << 8));
        const float d = half_of(h);
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t b = in[2 + j];
            x[j] = static_cast<float>(static_cast<int>(b & 0xF) - 8) * d;
            x[j + 16] = static_cast<float>(static_cast<int>(b >> 4) - 8) * d;
        }
    }
}

// ---- Q4_1: affine 4-bit blocks, x = min + d * q, q in [0, 15].
inline void q4_1_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 32, x += 32, out += 20) {
        float lo = x[0], hi = x[0];
        for (int j = 1; j < 32; ++j) { lo = std::min(lo, x[j]); hi = std::max(hi, x[j]); }
        const float d = (hi - lo) / 15.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        const std::uint16_t dh = f16_of(d), mh = f16_of(lo);
        out[0] = static_cast<std::uint8_t>(dh); out[1] = static_cast<std::uint8_t>(dh >> 8);
        out[2] = static_cast<std::uint8_t>(mh); out[3] = static_cast<std::uint8_t>(mh >> 8);
        for (int j = 0; j < 16; ++j) {
            const int q0 = std::max(0, std::min(15, static_cast<int>(std::trunc((x[j] - lo) * id + .5f))));
            const int q1 = std::max(0, std::min(15, static_cast<int>(std::trunc((x[j + 16] - lo) * id + .5f))));
            out[4 + j] = static_cast<std::uint8_t>(q0 | (q1 << 4));
        }
    }
}
inline void q4_1_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 32, x += 32, in += 20) {
        const float d = half_of(static_cast<std::uint16_t>(in[0] | (in[1] << 8)));
        const float m = half_of(static_cast<std::uint16_t>(in[2] | (in[3] << 8)));
        for (int j = 0; j < 16; ++j) { const std::uint8_t q = in[4 + j]; x[j] = m + d * (q & 15); x[j + 16] = m + d * (q >> 4); }
    }
}

// ---- Q5_0/Q5_1: five-bit legacy blocks; the high bit-plane is qh[0..3].
inline void q5_encode(const float* x, int n, std::uint8_t* out, bool affine) {
    const int stride = affine ? 24 : 22;
    for (int i = 0; i < n; i += 32, x += 32, out += stride) {
        float lo = x[0], hi = x[0], amax = 0.f, vmax = 0.f;
        for (int j = 0; j < 32; ++j) { lo = std::min(lo, x[j]); hi = std::max(hi, x[j]); if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); vmax = x[j]; } }
        const float d = affine ? (hi - lo) / 31.f : vmax / -16.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        const std::uint16_t dh = f16_of(d), mh = f16_of(lo);
        out[0] = static_cast<std::uint8_t>(dh); out[1] = static_cast<std::uint8_t>(dh >> 8);
        int off = 2;
        if (affine) { out[2] = static_cast<std::uint8_t>(mh); out[3] = static_cast<std::uint8_t>(mh >> 8); off = 4; }
        std::uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const float a = affine ? (x[j] - lo) * id : x[j] * id;
            const float b = affine ? (x[j + 16] - lo) * id : x[j + 16] * id;
            const int q0 = std::max(0, std::min(31, static_cast<int>(std::trunc(a + (affine ? .5f : 16.5f)))));
            const int q1 = std::max(0, std::min(31, static_cast<int>(std::trunc(b + (affine ? .5f : 16.5f)))));
            out[off + 4 + j] = static_cast<std::uint8_t>((q0 & 15) | ((q1 & 15) << 4));
            qh |= static_cast<std::uint32_t>((q0 >> 4) | ((q1 >> 4) << 16)) << j;
        }
        for (int b = 0; b < 4; ++b) out[off + b] = static_cast<std::uint8_t>(qh >> (8*b));
    }
}
inline void q5_decode(const std::uint8_t* in, int n, float* x, bool affine) {
    const int stride = affine ? 24 : 22;
    for (int i = 0; i < n; i += 32, x += 32, in += stride) {
        const float d = half_of(static_cast<std::uint16_t>(in[0] | (in[1] << 8)));
        const float m = affine ? half_of(static_cast<std::uint16_t>(in[2] | (in[3] << 8))) : 0.f;
        const int off = affine ? 4 : 2;
        std::uint32_t qh = 0; for (int b = 0; b < 4; ++b) qh |= static_cast<std::uint32_t>(in[off+b]) << (8*b);
        for (int j = 0; j < 16; ++j) { const std::uint8_t q = in[off + 4 + j]; const int q0 = (q & 15) | (((qh >> j) & 1u) << 4); const int q1 = (q >> 4) | (((qh >> (j+16)) & 1u) << 4); x[j] = affine ? m + d*q0 : d*(q0-16); x[j+16] = affine ? m + d*q1 : d*(q1-16); }
    }
}

// ---- Q1_0: sign bits, d = mean |x| of the 128-block
inline void q1_0_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 128, x += 128, out += 18) {
        float sum = 0.f;
        for (int j = 0; j < 128; ++j) sum += std::fabs(x[j]);
        const float d = sum / 128.f;
        const std::uint16_t h = f16_of(d);
        out[0] = static_cast<std::uint8_t>(h & 0xFF);
        out[1] = static_cast<std::uint8_t>(h >> 8);
        std::fill(out + 2, out + 18, static_cast<std::uint8_t>(0));
        for (int j = 0; j < 128; ++j)
            if (x[j] >= 0.f) out[2 + j / 8] |= static_cast<std::uint8_t>(1u << (j % 8));
    }
}
inline void q1_0_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 128, x += 128, in += 18) {
        const std::uint16_t h = static_cast<std::uint16_t>(in[0] | (in[1] << 8));
        const float d = half_of(h);
        for (int j = 0; j < 128; ++j)
            x[j] = (in[2 + j / 8] >> (j % 8) & 1u) ? d : -d;
    }
}

// ---- MXFP4: E8M0 a 32-block, codes nearest of the 16 doubled k-values,
//      nibbles paired j (low) / j + 16 (high).
const float kMXValues[16] = {0.f,  1.f,  2.f,  3.f,  4.f,  6.f,  8.f,  12.f,
                             0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

inline float e8m0_half(std::uint8_t e) {  // ggml_e8m0_to_fp32_half
    if (e >= 2) {
        const std::uint32_t bits = static_cast<std::uint32_t>(e - 1) << 23;
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }
    const std::uint32_t bits = 0x00200000u << e;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

inline void mxfp4_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 32, x += 32, out += 17) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
        int e = 0;
        if (amax > 0.f) {
            // floor(log2 amax), exact: frexp gives amax = m 2^be, 0.5 <= m < 1
            int be = 0;
            std::frexp(amax, &be);
            e = (be - 1) - 2 + 127;
            e = std::max(0, std::min(254, e));
        }
        out[0] = static_cast<std::uint8_t>(e);
        const float d = e8m0_half(static_cast<std::uint8_t>(e));
        std::fill(out + 1, out + 17, static_cast<std::uint8_t>(0));
        for (int j = 0; j < 32; ++j) {
            int best = 0;
            float err = std::fabs(d * kMXValues[0] - x[j]);
            for (int k = 1; k < 16; ++k) {
                const float e2 = std::fabs(d * kMXValues[k] - x[j]);
                if (e2 < err) { err = e2; best = k; }
            }
            const int byte = j < 16 ? j : j - 16;
            if (j < 16) out[1 + byte] |= static_cast<std::uint8_t>(best);
            else out[1 + byte] |= static_cast<std::uint8_t>(best << 4);
        }
    }
}
inline void mxfp4_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 32, x += 32, in += 17) {
        const float d = e8m0_half(in[0]);
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t b = in[1 + j];
            x[j] = d * kMXValues[b & 0xF];
            x[j + 16] = d * kMXValues[b >> 4];
        }
    }
}

// ---- NVFP4: four ue4m3 scale bytes a 64-block; sub-block of 16 stores
//      element e in byte e % 8, low nibble for e < 8, high for e >= 8.
inline float ue4m3_value(std::uint8_t s) {  // raw (pre-halving) ue4m3
    const std::uint32_t exp = (s >> 3) & 0xFu;
    const std::uint32_t man = s & 7u;
    if (s == 0x7F) return 0.f;  // ggml's NaN marker reads as zero
    if (exp == 0) return static_cast<float>(man) * (1.f / 512.f);  // man 2^-9
    const std::uint32_t bits = ((exp + 120u) << 23) | (man << 20);  // (1+man/8) 2^(e-7)
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

inline std::uint8_t value_to_ue4m3(float v) {  // nearest, bias 7, [0, 448]
    if (!(v > 0.f)) return 0;
    if (v > 448.f) return 0x7E;
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    int be = static_cast<int>((bits >> 23) & 0xFF) - 127 + 7;
    int man = static_cast<int>((bits >> 20) & 7);
    const int round = static_cast<int>((bits >> 19) & 1);
    man += round;
    if (man > 7) { man = 0; ++be; }
    if (be >= 15) return 0x7E;
    if (be <= 0) {  // subnormal: v = man/8 2^-9 (pre-halving); nearest man
        const float sub = v * 512.f;
        const int sm = static_cast<int>(std::floor(sub + 0.5f));
        return static_cast<std::uint8_t>(std::max(0, std::min(7, sm)));
    }
    return static_cast<std::uint8_t>((be << 3) | man);
}

inline void nvfp4_encode(const float* x, int n, std::uint8_t* out) {
    for (int i = 0; i < n; i += 64, x += 64, out += 36) {
        for (int sb = 0; sb < 4; ++sb) {
            float amax = 0.f;
            for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(x[sb * 16 + j]));
            // doubled k-values against a halved scale: pick scale = amax / 6
            out[sb] = value_to_ue4m3(amax / 6.f);
        }
        std::fill(out + 4, out + 36, static_cast<std::uint8_t>(0));
        for (int sb = 0; sb < 4; ++sb) {
            const float d = ue4m3_value(out[sb]) * 0.5f;
            for (int j = 0; j < 16; ++j) {
                int best = 0;
                float err = std::fabs(d * kMXValues[0] - x[sb * 16 + j]);
                for (int k = 1; k < 16; ++k) {
                    const float e2 = std::fabs(d * kMXValues[k] - x[sb * 16 + j]);
                    if (e2 < err) { err = e2; best = k; }
                }
                std::uint8_t* bytes = out + 4 + sb * 8;
                if (j < 8) bytes[j] |= static_cast<std::uint8_t>(best);
                else bytes[j - 8] |= static_cast<std::uint8_t>(best << 4);
            }
        }
    }
}
inline void nvfp4_decode(const std::uint8_t* in, int n, float* x) {
    for (int i = 0; i < n; i += 64, x += 64, in += 36) {
        for (int sb = 0; sb < 4; ++sb) {
            const float d = ue4m3_value(in[sb]) * 0.5f;
            for (int j = 0; j < 16; ++j) {
                const std::uint8_t b = in[4 + sb * 8 + j % 8];
                const std::uint8_t c = j < 8 ? static_cast<std::uint8_t>(b & 0xF)
                                             : static_cast<std::uint8_t>(b >> 4);
                x[sb * 16 + j] = d * kMXValues[c];
            }
        }
    }
}

// ---- TQ1_0 / TQ2_0 (ternary). Codes are q + 1 in {0, 1, 2}; digit n of a
//      stored byte decodes as (((b 3^n) & 255) * 3) >> 8 - 1.
inline int tq_digit(std::uint8_t b, int n) {
    const unsigned m = (static_cast<unsigned>(b) * (n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u)) & 0xFFu;
    return static_cast<int>((m * 3u) >> 8) - 1;
}
inline std::uint8_t tq_byte5(const int* c) {  // five codes, weights 3^(4-n)
    unsigned v = 0;
    for (int n = 0; n < 5; ++n) v = 3 * v + static_cast<unsigned>(c[n] + 1);
    return static_cast<std::uint8_t>((v * 256u + 242u) / 243u);
}

inline void tq_encode(const float* x, int n, std::uint8_t* out, bool tq1) {
    const int nb = n / 256;
    for (int i = 0; i < nb; ++i, x += 256) {
        float amax = 0.f;
        for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax;
        const float id = d > 0.f ? 1.f / d : 0.f;
        int q[256];
        for (int j = 0; j < 256; ++j) {
            float v = rintf32(x[j] * id);
            v = std::max(-1.f, std::min(1.f, v));
            q[j] = static_cast<int>(v);
        }
        std::uint8_t* dst = out + static_cast<std::size_t>(i) * (tq1 ? 54 : 66);
        if (!tq1) {
            // TQ2_0: byte b = 32 g + c holds elements 128 g + c + 32 r
            // (r = 0..3) in bit pairs 2 r.
            for (int b = 0; b < 64; ++b) {
                const int g = b / 32, c = b % 32;
                std::uint8_t byte = 0;
                for (int r = 0; r < 4; ++r)
                    byte |= static_cast<std::uint8_t>((q[128 * g + c + 32 * r] + 1) << (2 * r));
                dst[b] = byte;
            }
            const std::uint16_t h = f16_of(d);
            dst[64] = static_cast<std::uint8_t>(h & 0xFF);
            dst[65] = static_cast<std::uint8_t>(h >> 8);
            continue;
        }
        // TQ1_0: qs[0..31]: element m + 32n; qs[32..47]: 160 + m + 16n;
        // qh[0..3] (bytes 48..51): element 240 + m + 4n, encoded as a
        // five-trit byte with a trailing zero trit (weights 81, 27, 9, 3).
        int c[5];
        for (int m = 0; m < 32; ++m) {
            for (int t = 0; t < 5; ++t) c[t] = q[m + 32 * t];
            dst[m] = tq_byte5(c);
        }
        for (int m = 0; m < 16; ++m) {
            for (int t = 0; t < 5; ++t) c[t] = q[160 + m + 16 * t];
            dst[32 + m] = tq_byte5(c);
        }
        for (int b = 0; b < 4; ++b) {
            // digit a of byte b is element 240 + 4a + b; a trailing zero
            // trit (weight 1) completes the five-trit byte.
            int c4[5];
            for (int a = 0; a < 4; ++a) c4[a] = q[240 + 4 * a + b];
            c4[4] = -1;  // the pad trit 0, code 0 (tq_byte5 adds the +1)
            dst[48 + b] = tq_byte5(c4);
        }
        const std::uint16_t h = f16_of(d);
        dst[52] = static_cast<std::uint8_t>(h & 0xFF);
        dst[53] = static_cast<std::uint8_t>(h >> 8);
    }
}

inline void tq_decode(const std::uint8_t* in, int n, float* x, bool tq1) {
    const int nb = n / 256;
    for (int i = 0; i < nb; ++i, x += 256) {
        const std::uint8_t* src = in + static_cast<std::size_t>(i) * (tq1 ? 54 : 66);
        float d;
        if (tq1) {
            const std::uint16_t h = static_cast<std::uint16_t>(src[52] | (src[53] << 8));
            d = half_of(h);
            for (int m = 0; m < 32; ++m)
                for (int t = 0; t < 5; ++t) x[m + 32 * t] = static_cast<float>(tq_digit(src[m], t)) * d;
            for (int m = 0; m < 16; ++m)
                for (int t = 0; t < 5; ++t) x[160 + m + 16 * t] = static_cast<float>(tq_digit(src[32 + m], t)) * d;
            for (int b = 0; b < 4; ++b)
                for (int a = 0; a < 4; ++a)
                    x[240 + 4 * a + b] = static_cast<float>(tq_digit(src[48 + b], a)) * d;
        } else {
            const std::uint16_t h = static_cast<std::uint16_t>(src[64] | (src[65] << 8));
            d = half_of(h);
            for (int b = 0; b < 64; ++b) {
                const int g = b / 32, c = b % 32;
                for (int r = 0; r < 4; ++r) {
                    const int u = static_cast<int>((src[b] >> (2 * r)) & 3u);
                    x[128 * g + c + 32 * r] = static_cast<float>(u - 1) * d;
                }
            }
        }
    }
}

}  // namespace detail

//! Bytes a quantised/dequantised row needs.
inline std::size_t encoded_bytes(Type t, int n) { return row_bytes(t, n); }

//! Quantise `n` float32 values (n a multiple of the block size) to `out`.
inline void encode_row(Type t, const float* x, int n, std::uint8_t* out) {
    switch (t) {
        case Type::F32: std::memcpy(out, x, static_cast<std::size_t>(n) * 4); break;
        case Type::F16: for (int i = 0; i < n; ++i) { const std::uint16_t h = detail::f16_of(x[i]); out[2 * i] = static_cast<std::uint8_t>(h & 0xFF); out[2 * i + 1] = static_cast<std::uint8_t>(h >> 8); } break;
        case Type::BF16: for (int i = 0; i < n; ++i) { const std::uint16_t h = f32_to_bf16(x[i]); out[2 * i] = static_cast<std::uint8_t>(h & 0xFF); out[2 * i + 1] = static_cast<std::uint8_t>(h >> 8); } break;
        case Type::F64: for (int i = 0; i < n; ++i) { const double v = static_cast<double>(x[i]); std::uint64_t u; std::memcpy(&u, &v, 8); for (int b = 0; b < 8; ++b) out[8 * i + b] = static_cast<std::uint8_t>(u >> (8 * b)); } break;
        case Type::I8: for (int i = 0; i < n; ++i) { float q = detail::rintf32(x[i]); q = std::max(-128.f, std::min(127.f, q)); out[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q)); } break;
        case Type::I16: for (int i = 0; i < n; ++i) { const long long q = std::max(-32768LL, std::min(32767LL, std::llround(x[i]))); const std::uint16_t u = static_cast<std::uint16_t>(static_cast<std::int16_t>(q)); out[2*i] = static_cast<std::uint8_t>(u); out[2*i+1] = static_cast<std::uint8_t>(u >> 8); } break;
        case Type::I32: for (int i = 0; i < n; ++i) { const long long q = std::max(-2147483648LL, std::min(2147483647LL, std::llround(x[i]))); const std::uint32_t u = static_cast<std::uint32_t>(static_cast<std::int32_t>(q)); for (int b = 0; b < 4; ++b) out[4*i+b] = static_cast<std::uint8_t>(u >> (8*b)); } break;
        case Type::I64: for (int i = 0; i < n; ++i) { const std::int64_t q = static_cast<std::int64_t>(std::llround(x[i])); const std::uint64_t u = static_cast<std::uint64_t>(q); for (int b = 0; b < 8; ++b) out[8*i+b] = static_cast<std::uint8_t>(u >> (8*b)); } break;
        case Type::Q8_0: detail::q8_0_encode(x, n, out); break;
        case Type::Q8_1: detail::q8_1_encode(x, n, out); break;
        case Type::Q2_K: detail::q2_k_encode(x, n, out); break;
        case Type::Q3_K: detail::q3_k_encode(x, n, out); break;
        case Type::Q4_K: detail::q4_k_encode(x, n, out); break;
        case Type::Q5_K: detail::q5_k_encode(x, n, out); break;
        case Type::Q6_K: detail::q6_k_encode(x, n, out); break;
        case Type::Q8_K: detail::q8_k_encode(x, n, out); break;
        case Type::IQ2_XXS: detail::iq2_xxs_encode(x, n, out); break;
        case Type::IQ3_XXS: case Type::IQ1_S: case Type::IQ4_NL:
        case Type::IQ3_S: case Type::IQ2_S: case Type::IQ4_XS:
        case Type::IQ1_M:
            throw std::runtime_error("gguf: tensor type " + type_to_string(t) +
                                     " is import only; no reference-verified encoder is available");
        case Type::Q2_0: detail::q2_0_encode(x, n, out); break;
        case Type::IQ2_XS: detail::iq2_xs_encode(x, n, out); break;
        case Type::Q4_0: detail::q4_0_encode(x, n, out); break;
        case Type::Q4_1: detail::q4_1_encode(x, n, out); break;
        case Type::Q5_0: detail::q5_encode(x, n, out, false); break;
        case Type::Q5_1: detail::q5_encode(x, n, out, true); break;
        case Type::Q1_0: detail::q1_0_encode(x, n, out); break;
        case Type::MXFP4: detail::mxfp4_encode(x, n, out); break;
        case Type::NVFP4: detail::nvfp4_encode(x, n, out); break;
        case Type::TQ1_0: detail::tq_encode(x, n, out, true); break;
        case Type::TQ2_0: detail::tq_encode(x, n, out, false); break;
    }
}

inline void decode_row(Type t, const std::uint8_t* in, int n, float* x);

//! Encode `n` double values. Identical to encode_row() except F64, which
//! stores the values verbatim (bit-exact); every other type narrows first.
inline void encode_row_d(Type t, const double* x, int n, std::uint8_t* out) {
    if (t == Type::F64) {
        for (int i = 0; i < n; ++i) {
            std::uint64_t u;
            std::memcpy(&u, x + i, 8);
            for (int b = 0; b < 8; ++b) out[8 * i + b] = static_cast<std::uint8_t>(u >> (8 * b));
        }
        return;
    }
    std::vector<float> f(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) f[static_cast<std::size_t>(i)] = static_cast<float>(x[i]);
    encode_row(t, f.data(), n, out);
}

//! Decode `in` (n elements, encoded) to `n` doubles. F64 reads verbatim;
//! everything else widens the float32 result.
inline void decode_row_d(Type t, const std::uint8_t* in, int n, double* x) {
    if (t == Type::F64) {
        for (int i = 0; i < n; ++i) {
            std::uint64_t u = 0;
            for (int b = 0; b < 8; ++b) u |= static_cast<std::uint64_t>(in[8 * i + b]) << (8 * b);
            std::memcpy(x + i, &u, 8);
        }
        return;
    }
    std::vector<float> f(static_cast<std::size_t>(n));
    decode_row(t, in, n, f.data());
    for (int i = 0; i < n; ++i) x[i] = static_cast<double>(f[static_cast<std::size_t>(i)]);
}

//! Dequantise `in` (n elements, encoded) to `n` float32 values.
inline void decode_row(Type t, const std::uint8_t* in, int n, float* x) {
    switch (t) {
        case Type::F32: std::memcpy(x, in, static_cast<std::size_t>(n) * 4); break;
        case Type::F16: for (int i = 0; i < n; ++i) x[i] = detail::half_of(static_cast<std::uint16_t>(in[2 * i] | (in[2 * i + 1] << 8))); break;
        case Type::BF16: for (int i = 0; i < n; ++i) x[i] = bf16_to_f32(static_cast<std::uint16_t>(in[2 * i] | (in[2 * i + 1] << 8))); break;
        case Type::F64: for (int i = 0; i < n; ++i) { std::uint64_t u = 0; for (int b = 0; b < 8; ++b) u |= static_cast<std::uint64_t>(in[8 * i + b]) << (8 * b); double v; std::memcpy(&v, &u, 8); x[i] = static_cast<float>(v); } break;
        case Type::I8: for (int i = 0; i < n; ++i) x[i] = static_cast<float>(static_cast<std::int8_t>(in[i])); break;
        case Type::I16: for (int i = 0; i < n; ++i) x[i] = static_cast<float>(static_cast<std::int16_t>(static_cast<std::uint16_t>(in[2*i] | (in[2*i+1] << 8)))); break;
        case Type::I32: for (int i = 0; i < n; ++i) { std::uint32_t u = 0; for (int b = 0; b < 4; ++b) u |= static_cast<std::uint32_t>(in[4*i+b]) << (8*b); x[i] = static_cast<float>(static_cast<std::int32_t>(u)); } break;
        case Type::I64: for (int i = 0; i < n; ++i) { std::uint64_t u = 0; for (int b = 0; b < 8; ++b) u |= static_cast<std::uint64_t>(in[8*i+b]) << (8*b); x[i] = static_cast<float>(static_cast<std::int64_t>(u)); } break;
        case Type::Q8_0: detail::q8_0_decode(in, n, x); break;
        case Type::Q8_1: detail::q8_1_decode(in, n, x); break;
        case Type::Q2_K: detail::q2_k_decode(in, n, x); break;
        case Type::Q3_K: detail::q3_k_decode(in, n, x); break;
        case Type::Q4_K: detail::q4_k_decode(in, n, x); break;
        case Type::Q5_K: detail::q5_k_decode(in, n, x); break;
        case Type::Q6_K: detail::q6_k_decode(in, n, x); break;
        case Type::Q8_K: detail::q8_k_decode(in, n, x); break;
        case Type::IQ2_XXS: detail::iq2_xxs_decode(in, n, x); break;
        case Type::IQ3_XXS: detail::iq3_xxs_decode(in, n, x); break;
        case Type::IQ1_S: detail::iq1_s_decode(in, n, x); break;
        case Type::IQ4_NL: detail::iq4_nl_decode(in, n, x); break;
        case Type::IQ3_S: detail::iq3_s_decode(in, n, x); break;
        case Type::IQ2_S: detail::iq2_s_decode(in, n, x); break;
        case Type::IQ4_XS: detail::iq4_xs_decode(in, n, x); break;
        case Type::Q2_0: detail::q2_0_decode(in, n, x); break;
        case Type::IQ2_XS: detail::iq2_xs_decode(in, n, x); break;
        case Type::IQ1_M: detail::iq1_m_decode(in, n, x); break;
        case Type::Q4_0: detail::q4_0_decode(in, n, x); break;
        case Type::Q4_1: detail::q4_1_decode(in, n, x); break;
        case Type::Q5_0: detail::q5_decode(in, n, x, false); break;
        case Type::Q5_1: detail::q5_decode(in, n, x, true); break;
        case Type::Q1_0: detail::q1_0_decode(in, n, x); break;
        case Type::MXFP4: detail::mxfp4_decode(in, n, x); break;
        case Type::NVFP4: detail::nvfp4_decode(in, n, x); break;
        case Type::TQ1_0: detail::tq_decode(in, n, x, true); break;
        case Type::TQ2_0: detail::tq_decode(in, n, x, false); break;
    }
}

// -------------------------------------------------------------- container

//! One tensor: logical 2-D shape (ne[0] = columns, ne[1] = rows, ggml's
//! convention; the data are row-major with rows of ne[0] elements).
struct TensorInfo {
    std::string name;
    std::uint64_t ne[2] = {0, 0};  //!< columns, rows
    Type type = Type::F32;
    std::uint64_t offset = 0;      //!< bytes from the start of the data section
    std::uint64_t n_elements() const { return ne[0] * ne[1]; }
    std::size_t n_bytes() const {
        int e, b;
        type_shape(type, e, b);
        return static_cast<std::size_t>(n_elements() / static_cast<std::uint64_t>(e)) * b;
    }
};

//! Metadata values bff reads (a subset of GGUF's types -- enough for the
//! keys bff writes and the ones general tools write).
struct Value {
    enum Kind { UINT32, UINT64, INT64, FLOAT32, FLOAT64, BOOL, STRING, ARRAY_U32, ARRAY_I64, ARRAY_F32, ARRAY_F64, ARRAY_STR } kind = UINT64;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    float f = 0.f;
    double d = 0.0;
    bool b = false;
    std::string s;
    std::vector<std::uint32_t> au32;
    std::vector<std::int64_t> ai64;
    std::vector<float> af32;
    std::vector<double> af64;
    std::vector<std::string> astr;
};

inline void put_str_bytes(std::vector<std::uint8_t>& o, const std::string& s) {
    put_u64(o, s.size());
    o.insert(o.end(), s.begin(), s.end());
}

//! One KV of any supported kind, already encoded (type byte + payload) by
//! the writer; the reader decodes into Value.
struct KVPair {
    std::string key;
    Value value;
};

//! Streaming writer: add metadata and tensors, then serialise with
//! finish(). Tensor data are copied in add_tensor and laid out at the end,
//! each offset a multiple of the alignment.
class Writer {
public:
    explicit Writer(std::uint32_t alignment = 32) : alignment_(alignment) {
        if (alignment < 8 || alignment % 8 != 0)
            throw std::runtime_error("gguf: alignment must be a multiple of 8 >= 8");
    }

    void add_u32(const std::string& k, std::uint32_t v) { kvs_.push_back({k, simple(Value::UINT32, v)}); }
    void add_u64(const std::string& k, std::uint64_t v) { kvs_.push_back({k, simple(Value::UINT64, v)}); }
    void add_i64(const std::string& k, std::int64_t v) {
        Value v2;
        v2.kind = Value::INT64;
        v2.i = v;
        kvs_.push_back({k, v2});
    }
    void add_f32(const std::string& k, float v) {
        Value v2;
        v2.kind = Value::FLOAT32;
        v2.f = v;
        kvs_.push_back({k, v2});
    }
    void add_f64(const std::string& k, double v) {
        Value v2;
        v2.kind = Value::FLOAT64;
        v2.d = v;
        kvs_.push_back({k, v2});
    }
    void add_bool(const std::string& k, bool v) {
        Value v2;
        v2.kind = Value::BOOL;
        v2.b = v;
        kvs_.push_back({k, v2});
    }
    void add_str(const std::string& k, const std::string& v) {
        Value v2;
        v2.kind = Value::STRING;
        v2.s = v;
        kvs_.push_back({k, v2});
    }
    void add_array_u32(const std::string& k, const std::vector<std::uint32_t>& v) {
        Value val;
        val.kind = Value::ARRAY_U32;
        val.au32 = v;
        kvs_.push_back({k, val});
    }
    void add_array_i64(const std::string& k, const std::vector<std::int64_t>& v) {
        Value val;
        val.kind = Value::ARRAY_I64;
        val.ai64 = v;
        kvs_.push_back({k, val});
    }
    void add_array_f32(const std::string& k, const std::vector<float>& v) {
        Value val;
        val.kind = Value::ARRAY_F32;
        val.af32 = v;
        kvs_.push_back({k, val});
    }
    void add_array_f64(const std::string& k, const std::vector<double>& v) {
        Value val;
        val.kind = Value::ARRAY_F64;
        val.af64 = v;
        kvs_.push_back({k, val});
    }
    void add_array_str(const std::string& k, const std::vector<std::string>& v) {
        Value val;
        val.kind = Value::ARRAY_STR;
        val.astr = v;
        kvs_.push_back({k, val});
    }

    //! Add a tensor. `ne` is {columns, rows}; `n` = ne[0] * ne[1] elements;
    //! the data must be `row_bytes(type, n)` of encoded bytes.
    void add_tensor(const std::string& name, std::uint64_t ne0, std::uint64_t ne1,
                    Type type, const std::uint8_t* data) {
        TensorInfo t;
        t.name = name;
        t.ne[0] = ne0;
        t.ne[1] = ne1;
        t.type = type;
        const std::uint64_t n = ne0 * ne1;
        int e, b;
        type_shape(type, e, b);
        if (e > 1 && n % static_cast<std::uint64_t>(e) != 0)
            throw std::runtime_error("gguf: tensor '" + name + "' has " +
                                     std::to_string(n) + " elements, not a multiple of the " +
                                     type_to_string(type) + " block size " + std::to_string(e));
        if (name.size() > 64)
            throw std::runtime_error("gguf: tensor name longer than 64 bytes");
        const std::uint64_t pad = data_at_ % alignment_ == 0
                                      ? 0
                                      : alignment_ - data_at_ % alignment_;
        tensor_data_.insert(tensor_data_.end(), pad, 0);  // keep offsets true
        t.offset = data_at_ + pad;  // the data, not the pad, starts here
        tensors_.push_back(t);
        const std::size_t bytes = static_cast<std::size_t>(n / e) * b;
        tensor_data_.insert(tensor_data_.end(), data, data + bytes);
        data_at_ += bytes + pad;  // next tensor's offset stays aligned
        if (e > 1 && !has_quantization_version_) {
            add_u32("general.quantization_version", 2);
            has_quantization_version_ = true;
        }
    }

    //! Serialise the file image.
    std::vector<std::uint8_t> finish() const {
        std::vector<std::uint8_t> o;
        put_u32(o, 0x46554747u);  // "GGUF" little-endian
        put_u32(o, 3);
        put_u64(o, tensors_.size());
        put_u64(o, kvs_.size());
        for (const KVPair& kv : kvs_) {
            put_str_bytes(o, kv.key);
            encode_value(o, kv.value);
        }
        for (const TensorInfo& t : tensors_) {
            put_str_bytes(o, t.name);
            put_u32(o, 2);
            put_u64(o, t.ne[0]);
            put_u64(o, t.ne[1]);
            put_u32(o, static_cast<std::uint32_t>(t.type));
            put_u64(o, t.offset);
        }
        while (o.size() % alignment_ != 0) o.push_back(0);
        o.insert(o.end(), tensor_data_.begin(), tensor_data_.end());
        return o;
    }

private:
    static Value simple(Value::Kind k, std::uint64_t v) {
        Value val;
        val.kind = k;
        val.u = v;
        return val;
    }
    static void encode_value(std::vector<std::uint8_t>& o, const Value& v) {
        switch (v.kind) {
            case Value::UINT32: put_u32(o, 4); put_u32(o, static_cast<std::uint32_t>(v.u)); break;
            case Value::UINT64: put_u32(o, 10); put_u64(o, v.u); break;
            case Value::INT64: put_u32(o, 11); put_u64(o, static_cast<std::uint64_t>(v.i)); break;
            case Value::FLOAT32: put_u32(o, 6); put_f32(o, v.f); break;
            case Value::FLOAT64: put_u32(o, 12); put_f64(o, v.d); break;
            case Value::BOOL: put_u32(o, 7); o.push_back(v.b ? 1 : 0); break;
            case Value::STRING: put_u32(o, 8); put_str_bytes(o, v.s); break;
            case Value::ARRAY_U32:
                put_u32(o, 9); put_u32(o, 4); put_u64(o, v.au32.size());
                for (std::uint32_t x : v.au32) put_u32(o, x);
                break;
            case Value::ARRAY_I64:
                put_u32(o, 9); put_u32(o, 11); put_u64(o, v.ai64.size());
                for (std::int64_t x : v.ai64) put_u64(o, static_cast<std::uint64_t>(x));
                break;
            case Value::ARRAY_F32:
                put_u32(o, 9); put_u32(o, 6); put_u64(o, v.af32.size());
                for (float x : v.af32) put_f32(o, x);
                break;
            case Value::ARRAY_F64:
                put_u32(o, 9); put_u32(o, 12); put_u64(o, v.af64.size());
                for (double x : v.af64) put_f64(o, x);
                break;
            case Value::ARRAY_STR:
                put_u32(o, 9); put_u32(o, 8); put_u64(o, v.astr.size());
                for (const std::string& x : v.astr) put_str_bytes(o, x);
                break;
        }
    }
    std::uint32_t alignment_;
    std::vector<KVPair> kvs_;
    std::vector<TensorInfo> tensors_;
    std::vector<std::uint8_t> tensor_data_;
    std::uint64_t data_at_ = 0;
    bool has_quantization_version_ = false;
};

//! Read a GGUF file image: metadata by key, tensors by name.
class Reader {
public:
    Reader(const std::uint8_t* data, std::size_t size) : c_(data, size) { parse(); }
    explicit Reader(const std::vector<std::uint8_t>& v) : Reader(v.data(), v.size()) {}
    explicit Reader(const std::string& s)
        : Reader(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()) {}

    std::uint32_t version() const { return version_; }
    std::uint32_t alignment() const { return alignment_; }
    std::size_t n_tensors() const { return tensors_.size(); }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    const TensorInfo* find_tensor(const std::string& name) const {
        for (const TensorInfo& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }
    //! A copy of the tensor's raw (still encoded) bytes.
    std::vector<std::uint8_t> tensor_bytes(const TensorInfo& t) const {
        const std::size_t off = data_start_ + t.offset;
        if (off + t.n_bytes() > c_.n) throw std::runtime_error("gguf: tensor data out of range");
        return std::vector<std::uint8_t>(c_.p + off, c_.p + off + t.n_bytes());
    }

    bool has(const std::string& key) const {
        for (const KVPair& kv : kvs_)
            if (kv.key == key) return true;
        return false;
    }
    const Value* find(const std::string& key) const {
        for (const KVPair& kv : kvs_)
            if (kv.key == key) return &kv.value;
        return nullptr;
    }
    std::string get_str(const std::string& key, const std::string& fallback = "") const {
        const Value* v = find(key);
        return v && v->kind == Value::STRING ? v->s : fallback;
    }
    std::uint64_t get_u64(const std::string& key, std::uint64_t fallback = 0) const {
        const Value* v = find(key);
        if (!v) return fallback;
        if (v->kind == Value::UINT64 || v->kind == Value::UINT32) return v->u;
        if (v->kind == Value::INT64) return static_cast<std::uint64_t>(v->i);
        return fallback;
    }
    double get_f64(const std::string& key, double fallback = 0.0) const {
        const Value* v = find(key);
        if (!v) return fallback;
        if (v->kind == Value::FLOAT64) return v->d;
        if (v->kind == Value::FLOAT32) return v->f;
        return fallback;
    }
    std::vector<double> get_array_f64(const std::string& key) const {
        const Value* v = find(key);
        if (!v) return {};
        if (v->kind == Value::ARRAY_F64) return v->af64;
        if (v->kind == Value::ARRAY_F32) return std::vector<double>(v->af32.begin(), v->af32.end());
        return {};
    }
    std::vector<std::string> get_array_str(const std::string& key) const {
        const Value* v = find(key);
        return v && v->kind == Value::ARRAY_STR ? v->astr : std::vector<std::string>();
    }

private:
    Value read_value(std::uint32_t type) {
        Value v;
        switch (type) {
            case 0: v.kind = Value::UINT32; v.u = c_.u8(); break;
            case 1: v.kind = Value::UINT32; v.u = static_cast<std::int8_t>(c_.u8()); break;
            case 2: v.kind = Value::UINT32; v.u = c_.u16(); break;
            case 3: v.kind = Value::UINT32; v.u = static_cast<std::uint16_t>(static_cast<std::int16_t>(c_.u16())); break;
            case 4: v.kind = Value::UINT32; v.u = c_.u32(); break;
            case 5: v.kind = Value::UINT32; v.u = static_cast<std::uint32_t>(static_cast<std::int32_t>(c_.u32())); break;
            case 6: v.kind = Value::FLOAT32; v.f = c_.f32(); break;
            case 7: v.kind = Value::BOOL; v.b = c_.u8() != 0; break;
            case 8: v.kind = Value::STRING; v.s = read_str(); break;
            case 9: {
                const std::uint32_t et = c_.u32();
                const std::uint64_t n = c_.u64();
                for (std::uint64_t i = 0; i < n; ++i) {
                    const Value e = read_value(et);
                    if (e.kind == Value::UINT32) v.au32.push_back(static_cast<std::uint32_t>(e.u));
                    else if (e.kind == Value::INT64) v.ai64.push_back(e.i);
                    else if (e.kind == Value::FLOAT32) v.af32.push_back(e.f);
                    else if (e.kind == Value::FLOAT64) v.af64.push_back(e.d);
                    else if (e.kind == Value::STRING) v.astr.push_back(e.s);
                    else if (e.kind == Value::BOOL) { v.ai64.push_back(e.b ? 1 : 0); }
                    else throw std::runtime_error("gguf: unsupported array element type");
                }
                // pick the array kind from the first element seen
                if (et == 4 || et == 5) v.kind = Value::ARRAY_U32;
                else if (et == 10 || et == 11 || et == 7) v.kind = Value::ARRAY_I64;
                else if (et == 6) v.kind = Value::ARRAY_F32;
                else if (et == 12) v.kind = Value::ARRAY_F64;
                else if (et == 8) v.kind = Value::ARRAY_STR;
                break;
            }
            case 10: v.kind = Value::UINT64; v.u = c_.u64(); break;
            case 11: v.kind = Value::INT64; v.i = static_cast<std::int64_t>(c_.u64()); break;
            case 12: v.kind = Value::FLOAT64; v.d = c_.f64(); break;
            default: throw std::runtime_error("gguf: unsupported metadata type " + std::to_string(type));
        }
        return v;
    }
    std::string read_str() {
        const std::uint64_t n = c_.u64();
        const std::uint8_t* p;
        c_.raw(p, static_cast<std::size_t>(n));
        return std::string(reinterpret_cast<const char*>(p), static_cast<std::size_t>(n));
    }
    void parse() {
        if (c_.u32() != 0x46554747u) throw std::runtime_error("gguf: not a GGUF file (bad magic)");
        version_ = c_.u32();
        if (version_ < 1 || version_ > 3)
            throw std::runtime_error("gguf: unsupported GGUF version " + std::to_string(version_));
        const std::uint64_t nt = c_.u64();
        const std::uint64_t nk = c_.u64();
        for (std::uint64_t i = 0; i < nk; ++i) {
            KVPair kv;
            kv.key = read_str();
            const std::uint32_t t = c_.u32();
            kv.value = read_value(t);
            if (kv.key == "general.alignment" && kv.value.kind == Value::UINT32)
                alignment_ = static_cast<std::uint32_t>(kv.value.u);
            kvs_.push_back(std::move(kv));
        }
        if (alignment_ < 8 || alignment_ % 8 != 0)
            throw std::runtime_error("gguf: invalid alignment");
        tensors_.resize(static_cast<std::size_t>(nt));
        for (std::uint64_t i = 0; i < nt; ++i) {
            TensorInfo& t = tensors_[static_cast<std::size_t>(i)];
            t.name = read_str();
            const std::uint32_t nd = c_.u32();
            if (nd != 2) throw std::runtime_error("gguf: tensor '" + t.name + "' has " + std::to_string(nd) + " dimensions, bff reads 2-D tensors");
            t.ne[0] = c_.u64();
            t.ne[1] = c_.u64();
            t.type = static_cast<Type>(c_.u32());
            switch (t.type) {
                case Type::F32: case Type::F16: case Type::Q4_0: case Type::Q4_1:
                case Type::Q5_0: case Type::Q5_1: case Type::Q8_0: case Type::Q8_1:
                case Type::Q2_K: case Type::Q3_K: case Type::Q4_K: case Type::Q5_K: case Type::Q6_K: case Type::Q8_K: case Type::IQ2_XXS: case Type::IQ2_XS: case Type::IQ3_XXS: case Type::IQ1_S: case Type::IQ4_NL: case Type::IQ3_S: case Type::IQ2_S: case Type::IQ4_XS: case Type::Q2_0:
                case Type::F64: case Type::BF16: case Type::TQ1_0: case Type::TQ2_0:
                case Type::MXFP4: case Type::NVFP4: case Type::Q1_0:
                case Type::I8: case Type::I16: case Type::I32: case Type::I64: case Type::IQ1_M: break;
                default:
                    throw std::runtime_error("gguf: tensor '" + t.name + "' has unsupported type id " +
                                             std::to_string(static_cast<std::uint32_t>(t.type)));
            }
            t.offset = c_.u64();
        }
        while (c_.at % alignment_ != 0) {
            c_.need(1);
            ++c_.at;
        }
        data_start_ = c_.at;
    }
    Cursor c_;
    std::uint32_t version_ = 3;
    std::uint32_t alignment_ = 32;
    std::vector<KVPair> kvs_;
    std::vector<TensorInfo> tensors_;
    std::size_t data_start_ = 0;
};

}  // namespace gguf
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_GGUFIO_H

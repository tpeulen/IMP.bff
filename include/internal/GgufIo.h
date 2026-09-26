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

namespace IMP {
namespace bff {
namespace internal {
namespace gguf {

//! The tensor types bff maps, by their GGUF v3 numbers.
enum class Type : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    I8 = 24,
    F64 = 28,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41
};

//! Elements a block covers and bytes a block occupies, per type.
inline void type_shape(Type t, int& elements, int& bytes) {
    switch (t) {
        case Type::F32: elements = 1; bytes = 4; return;
        case Type::F16: elements = 1; bytes = 2; return;
        case Type::BF16: elements = 1; bytes = 2; return;
        case Type::F64: elements = 1; bytes = 8; return;
        case Type::I8: elements = 1; bytes = 1; return;
        case Type::Q4_0: elements = 32; bytes = 18; return;
        case Type::Q8_0: elements = 32; bytes = 34; return;
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
    if (s == "q8_0") return Type::Q8_0;
    if (s == "q4_0") return Type::Q4_0;
    if (s == "q1_0") return Type::Q1_0;
    if (s == "mxfp4") return Type::MXFP4;
    if (s == "nvfp4") return Type::NVFP4;
    if (s == "tq1_0") return Type::TQ1_0;
    if (s == "tq2_0") return Type::TQ2_0;
    throw std::runtime_error("gguf: unknown tensor type '" + name +
                             "' (F32, F16, BF16, F64, I8, Q8_0, Q4_0, Q1_0, "
                             "MXFP4, NVFP4, TQ1_0, TQ2_0)");
}

//! The canonical ggml spelling of a type.
inline std::string type_to_string(Type t) {
    switch (t) {
        case Type::F32: return "F32";
        case Type::F16: return "F16";
        case Type::BF16: return "BF16";
        case Type::F64: return "F64";
        case Type::I8: return "I8";
        case Type::Q4_0: return "Q4_0";
        case Type::Q8_0: return "Q8_0";
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
           t == Type::I8;
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
        case Type::Q8_0: detail::q8_0_encode(x, n, out); break;
        case Type::Q4_0: detail::q4_0_encode(x, n, out); break;
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
        case Type::Q8_0: detail::q8_0_decode(in, n, x); break;
        case Type::Q4_0: detail::q4_0_decode(in, n, x); break;
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
                case Type::F32: case Type::F16: case Type::Q4_0: case Type::Q8_0:
                case Type::F64: case Type::BF16: case Type::TQ1_0: case Type::TQ2_0:
                case Type::MXFP4: case Type::NVFP4: case Type::Q1_0: case Type::I8: break;
                default:
                    throw std::runtime_error("gguf: tensor '" + t.name + "' has unsupported type " + type_to_string(Type(0)) + " id " + std::to_string(static_cast<std::uint32_t>(t.type)));
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

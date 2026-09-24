/**
 *  \file IMP/bff/internal/MlpFp4.h
 *  \brief 4-bit floating-point (FP4 E2M1) weights for a dense network:
 *         per-row "fp4", OCP "mxfp4" and NVIDIA "nvfp4" -- the codecs, the
 *         packed layout, and the float reference path.
 *
 * std-only and header-only, like MlpCore.h. The fast integer-SIMD kernels
 * that run on these codes are `MlpFp4Kernels.h`; training is
 * `MlpFp4Train.h`.
 *
 * **1. Element and scale codecs.**
 *
 * - *E2M1* (FP4; OCP MX v1.0 "FP4", ONNX `FLOAT4E2M1`, PyTorch
 *   `float4_e2m1fn_x2`): 1 sign bit, 2 exponent bits (bias 1), 1 mantissa
 *   bit; no infinity, no NaN. Code `s ee m`: `ee = 0` is subnormal
 *   (`0.5 m`), otherwise `2^(ee-1) (1 + m/2)`. The eight magnitudes are
 *   0, 0.5, 1, 1.5, 2, 3, 4, 6 at codes 0..7; codes 8..15 are their
 *   negatives (8 is -0). Encoding rounds to nearest, ties to even (to the
 *   code with m = 0: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2,
 *   3.5 -> 4, 5 -> 4), and saturates to +-6 (so 100 -> 6); the sign is kept
 *   (-0.1 -> -0, code 8). NaN encodes as +0 -- E2M1 has no NaN; weights are
 *   validated finite before they get here. `e2m1_encode_sr` is the
 *   stochastic rounding of the training recipe (unbiased inside +-6).
 * - *Packing*: two codes a byte, **element 2i in the low nibble, element
 *   2i+1 in the high nibble** -- `byte = c[2i] | c[2i+1] << 4`, the order of
 *   ONNX FLOAT4E2M1 ("the first element in the 4 least significant bits"),
 *   PyTorch's `float4_e2m1fn_x2` / `pack_uint4`, and NVIDIA's packed NVFP4
 *   tensors. (llama.cpp's blocks use a different in-block order, element j
 *   and j + 16 in one byte; the kernels here keep the standard order and
 *   permute the int8 activations instead -- see MlpFp4Kernels.h.)
 * - *E4M3* (FP8, the OCP/NVIDIA "e4m3fn" variant, PyTorch
 *   `float8_e4m3fn`): bias 7, 3 mantissa bits, no infinities; `S.1111.111`
 *   (0x7F / 0xFF) is NaN, so the largest finite magnitude is
 *   `S.1111.110` = 448. Exponent field 0 is subnormal, `m 2^-9` (smallest
 *   2^-9, largest subnormal 7 2^-9). Encoding is round-to-nearest-even with
 *   saturation to +-448 (the "satfinite" conversion of NVIDIA's
 *   `cvt.rn.satfinite.e4m3x2.f32`, and what PyTorch's cast does), NaN ->
 *   0x7F.
 * - *E8M0* (the OCP MX shared scale, PyTorch `float8_e8m0fnu`): an unsigned
 *   biased exponent, `2^(code - 127)`, codes 0..254 = 2^-127 .. 2^127, 255
 *   is NaN. Only exact powers of two are ever encoded here.
 *
 * **2. Weight formats** (per layer; weights row-major `n_out x n_in`; blocks
 * run along `n_in` -- the K dimension of the product -- within one output
 * row). Every row is padded with zero codes to a multiple of 32 elements
 * (16 bytes), so the kernels only ever see whole 16-byte groups; a
 * trailing partial block is thereby padded with zeros, as the specs say.
 *
 * - `fp4`: one block per row, E2M1 codes with one **float32 scale per output
 *   row**, `s = float(absmax_row / 6)` (1 for an all-zero row); code =
 *   `e2m1(w / s)`, value = `e2m1 * s`. `4 + 32 / n_in` bits a weight.
 * - `mxfp4`: OCP Microscaling Formats (MX) v1.0, MXFP4: blocks of 32, one
 *   shared E8M0 scale a block. The spec's conversion (section 6.3):
 *   `X = 2^(floor(log2(absmax_block)) - emax_elem)` with `emax_elem = 2`
 *   for E2M1 (its largest normal is 1.5 * 2^2), then each element is
 *   `e2m1(w / X)`, rounded to nearest even and **clamped** to +-6 (values
 *   in [6, 8) X saturate -- the spec's rule accepts that). `floor(log2)` is
 *   taken exactly (`std::ilogb`), not through a rounded logarithm; the
 *   exponent is clamped to [-127, 127]; an all-zero block gets code 0
 *   (2^-127) and zero elements. 4.25 bits a weight. PyTorch/torchao default
 *   to the "RCEIL" variant `2^ceil(log2(absmax / 6))`, which never
 *   saturates; it equals the OCP scale when `frac(log2 absmax) <= log2 1.5`
 *   and is twice it otherwise. bff follows the OCP text.
 * - `nvfp4`: NVIDIA NVFP4 (Blackwell; the recipe of TensorRT Model
 *   Optimizer's `NVFP4QTensor`, Transformer Engine, and arXiv 2509.25149
 *   appendix B): blocks of 16, one FP8 E4M3 scale a block, plus one float32
 *   scale a tensor: `g = float(absmax_tensor / (6 * 448))` (the paper's
 *   `1 / s_enc`; 1 if the tensor is zero); block scale
 *   `S = e4m3(absmax_block / (6 g))` (the paper's `e4m3(s_dec,b * s_enc)`;
 *   an all-zero block gets `e4m3(1.0)`, as Model Optimizer sets zero block
 *   scales to 1 before the cast); element `e2m1(w / (S g))`; value
 *   `e2m1 * S * g`. The largest block maps to S = 448 and its largest
 *   element to 6. If S underflows to 0 (a block below ~2^-9 / 448 of the
 *   tensor absmax) its elements are stored as 0 rather than dividing by
 *   zero. 4.5 bits a weight + 32 bits a tensor.
 *
 * All scale arithmetic that decides a code is done in double (Model
 * Optimizer divides in float32; the two can differ only for a quotient
 * within float32 rounding of an E2M1 tie). Decoding is exact in double: an
 * E2M1 value times an E4M3 (or E8M0) scale times a float32 has at most 29
 * significant bits.
 *
 * **3. The float reference** (`Fp4Model`, `predict_reference`). Decodes a
 * layer's weights into a double matrix and multiplies in double
 * (optionally after fake-quantising each layer's input per row in the same
 * format). QuantizedNeuralNet does not use it -- its predict runs the
 * integer kernels of MlpFp4Kernels.h on the packed codes -- it is kept for
 * the tests, which pin the kernels against it.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPFP4_H
#define IMPBFF_INTERNAL_MLPFP4_H

#include <IMP/bff/internal/MlpCore.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace IMP {
namespace bff {
namespace internal {
namespace mlpfp4 {

// ------------------------------------------------------------------ codecs

//! The sixteen E2M1 values, by code (8 is -0).
constexpr double kE2M1[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
//! Largest E2M1 magnitude.
constexpr double kE2M1Max = 6.0;
//! Largest finite E4M3 (e4m3fn) magnitude.
constexpr double kE4M3Max = 448.0;

//! The value of an E2M1 code (low 4 bits of `code`).
inline double e2m1_decode(std::uint8_t code) { return kE2M1[code & 0x0F]; }

//! Round-to-nearest-even E2M1 code of `x`, saturating to +-6.
/*! Ties go to the even code (mantissa bit 0): 0.25 -> 0, 0.75 -> 1,
    1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4. NaN -> +0. */
inline std::uint8_t e2m1_encode(double x) {
    if (std::isnan(x)) return 0;
    const std::uint8_t sign = std::signbit(x) ? 0x08 : 0x00;
    const double m = std::abs(x);
    // branch-free count of the thresholds passed; `<=` at the ties that go
    // down (to the even code), `<` at those that go up
    const int i = (m > 0.25) + (m >= 0.75) + (m > 1.25) + (m >= 1.75) + (m > 2.5) + (m >= 3.5) + (m > 5.0);
    return static_cast<std::uint8_t>(sign | i);
}

//! Stochastically rounded E2M1 code of `x` for a uniform draw `u` in [0, 1).
/*! |x| between neighbours lo < hi rounds up with probability
    (|x| - lo) / (hi - lo), so the expected decoded value is x for |x| <= 6;
    beyond, it saturates to +-6. */
inline std::uint8_t e2m1_encode_sr(double x, double u) {
    if (std::isnan(x)) return 0;
    const std::uint8_t sign = std::signbit(x) ? 0x08 : 0x00;
    const double m = std::abs(x);
    if (m >= kE2M1Max) return static_cast<std::uint8_t>(sign | 7);
    std::uint8_t i = 0;
    while (i < 7 && kE2M1[i + 1] <= m) ++i;  // kE2M1[i] <= m < kE2M1[i + 1]
    const double lo = kE2M1[i], hi = kE2M1[i + 1];
    if (u < (m - lo) / (hi - lo)) ++i;
    return static_cast<std::uint8_t>(sign | i);
}

//! Round half to even of a finite non-negative double (independent of fenv).
inline double round_half_even(double v) {
    const double r = std::floor(v);
    const double d = v - r;
    if (d > 0.5) return r + 1.0;
    if (d < 0.5) return r;
    return std::fmod(r, 2.0) == 0.0 ? r : r + 1.0;
}

//! The value of an E4M3 (e4m3fn) code; 0x7F / 0xFF are NaN.
inline double e4m3_decode(std::uint8_t code) {
    const int e = (code >> 3) & 0x0F;
    const int m = code & 0x07;
    const double sign = (code & 0x80) ? -1.0 : 1.0;
    if (e == 0x0F && m == 0x07) return std::numeric_limits<double>::quiet_NaN();
    if (e == 0) return sign * std::ldexp(static_cast<double>(m), -9);
    return sign * std::ldexp(1.0 + m / 8.0, e - 7);
}

//! Round-to-nearest-even E4M3 code of `x`, saturating to +-448; NaN -> 0x7F.
inline std::uint8_t e4m3_encode(double x) {
    if (std::isnan(x)) return 0x7F;
    const std::uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
    const double m = std::abs(x);
    if (m >= kE4M3Max) return static_cast<std::uint8_t>(sign | 0x7E);
    if (m == 0.0) return sign;
    const int e = std::ilogb(m);  // floor(log2 m), exact
    int code;
    if (e < -6) {
        // subnormal: quantum 2^-9, q in [0, 8]; q == 8 is 2^-6, code 0x08
        code = static_cast<int>(round_half_even(std::ldexp(m, 9)));
    } else {
        // normal: quantum 2^(e-3), q in [8, 16]; q == 16 carries into the
        // exponent field, which the code arithmetic does by itself
        const int q = static_cast<int>(round_half_even(std::ldexp(m, 3 - e)));
        code = ((e + 7) << 3) + (q - 8);
    }
    if (code > 0x7E) code = 0x7E;  // cannot happen for m < 448; kept as a guard
    return static_cast<std::uint8_t>(sign | code);
}

//! The value of an E8M0 code: 2^(code - 127); 255 is NaN.
inline double e8m0_decode(std::uint8_t code) {
    if (code == 0xFF) return std::numeric_limits<double>::quiet_NaN();
    return std::ldexp(1.0, static_cast<int>(code) - 127);
}

//! The E8M0 code of 2^e, e clamped to [-127, 127].
inline std::uint8_t e8m0_from_exponent(int e) {
    return static_cast<std::uint8_t>(std::max(-127, std::min(127, e)) + 127);
}

//! Pack `n` 4-bit codes, element 2i in the low nibble; `(n + 1) / 2` bytes.
inline void pack_nibbles(const std::uint8_t* codes, std::size_t n, std::uint8_t* out) {
    for (std::size_t i = 0; i < n / 2; ++i)
        out[i] = static_cast<std::uint8_t>((codes[2 * i] & 0x0F) | ((codes[2 * i + 1] & 0x0F) << 4));
    if (n % 2) out[n / 2] = static_cast<std::uint8_t>(codes[n - 1] & 0x0F);
}

//! Code `i` of a packed nibble array.
inline std::uint8_t nibble(const std::uint8_t* packed, std::size_t i) {
    return static_cast<std::uint8_t>((packed[i / 2] >> ((i % 2) * 4)) & 0x0F);
}

//! A small deterministic generator (splitmix64) for stochastic rounding.
struct SplitMix64 {
    std::uint64_t s;
    explicit SplitMix64(std::uint64_t seed = 0) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    //! Uniform in [0, 1), 53 bits.
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

// ------------------------------------------------------------------ formats

enum class Format { FP4, MXFP4, NVFP4 };

//! How a value is rounded to E2M1.
enum class Rounding { NearestEven, Stochastic };

//! "fp4", "mxfp4" or "nvfp4" (exact, lower case); throws otherwise.
inline Format format_from_string(const std::string& name) {
    if (name == "fp4") return Format::FP4;
    if (name == "mxfp4") return Format::MXFP4;
    if (name == "nvfp4") return Format::NVFP4;
    throw std::runtime_error("unknown FP4 format '" + name + "'");
}

inline std::string format_to_string(Format f) {
    switch (f) {
        case Format::FP4: return "fp4";
        case Format::MXFP4: return "mxfp4";
        case Format::NVFP4: return "nvfp4";
    }
    return "fp4";
}

//! Elements of a row after padding: `cols` rounded up to a multiple of 32.
inline int padded_cols(int cols) { return cols <= 0 ? 0 : (cols + 31) / 32 * 32; }

//! Elements a block holds: the whole padded row for fp4, 32 for mxfp4,
//! 16 for nvfp4.
inline int block_size(Format f, int n_cols) {
    switch (f) {
        case Format::FP4: return std::max(32, padded_cols(n_cols));
        case Format::MXFP4: return 32;
        case Format::NVFP4: return 16;
    }
    return 32;
}

//! A row-major `rows x cols` matrix in packed FP4 with its scales.
/*! Rows are padded with zero codes to `padded_cols()` (a multiple of 32).
    Row r's codes start at byte `r * row_bytes()`; `scales` holds, per row:
    fp4 -- one float32 (4 little-endian bytes); mxfp4 -- one E8M0 byte a
    block; nvfp4 -- one E4M3 byte a block (blocks of the padded row).
    `tensor_scale` is nvfp4's float32 global scale (1 otherwise). A 2-D
    (16 x 16 / 32 x 32) scaled tensor is the same layout with each tile's
    scale repeated in its rows. */
struct Fp4Tensor {
    Format format = Format::FP4;
    int rows = 0;
    int cols = 0;
    int block = 32;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> scales;
    float tensor_scale = 1.0f;

    std::size_t padded() const { return static_cast<std::size_t>(padded_cols(cols)); }
    int blocks_per_row() const { return block <= 0 ? 0 : static_cast<int>(padded()) / block; }
    std::size_t row_bytes() const { return padded() / 2; }
    //! Bytes of scale data a row carries.
    std::size_t row_scale_bytes() const {
        return format == Format::FP4 ? 4 : static_cast<std::size_t>(blocks_per_row());
    }
    //! Codes + scales (+ nvfp4's 4-byte tensor scale).
    std::size_t bytes() const {
        return codes.size() + scales.size() + (format == Format::NVFP4 ? 4 : 0);
    }
    //! Whether the buffers have the sizes the shape implies.
    bool consistent() const {
        return rows >= 0 && cols >= 0 && block == block_size(format, cols) &&
               codes.size() == static_cast<std::size_t>(rows) * row_bytes() &&
               scales.size() == static_cast<std::size_t>(rows) * row_scale_bytes();
    }
};

namespace detail {
inline void put_f32(float v, std::uint8_t* out) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    for (int i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(u >> (8 * i));
}
inline float get_f32(const std::uint8_t* in) {
    std::uint32_t u = 0;
    for (int i = 0; i < 4; ++i) u |= static_cast<std::uint32_t>(in[i]) << (8 * i);
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}
inline double absmax(const double* x, std::size_t n) {
    double a = 0.0;
    for (std::size_t i = 0; i < n; ++i) a = std::max(a, std::abs(x[i]));
    return a;
}
}  // namespace detail

//! NVFP4's float32 global scale for a tensor of absmax `amax`.
inline float nvfp4_tensor_scale(double amax) {
    const float g = static_cast<float>(amax / (kE2M1Max * kE4M3Max));
    return (g > 0.0f && std::isfinite(g)) ? g : 1.0f;
}

//! The scale code of a block of absmax `amax` (mxfp4: E8M0; nvfp4: E4M3).
inline std::uint8_t block_scale_code(Format f, double amax, float g) {
    if (f == Format::MXFP4) return e8m0_from_exponent(amax > 0.0 ? std::ilogb(amax) - 2 : -127);
    return amax > 0.0 ? e4m3_encode(amax / (kE2M1Max * static_cast<double>(g))) : e4m3_encode(1.0);
}

//! The decoded multiplier of a block scale code (without the fp4 row case).
inline double block_scale_value(Format f, std::uint8_t code, float g) {
    return f == Format::MXFP4 ? e8m0_decode(code) : e4m3_decode(code) * static_cast<double>(g);
}

//! Encode `n` values with decoded scale `d` into (unpacked) codes.
inline void encode_elements(const double* x, int n, double d, std::uint8_t* out,
                            Rounding rnd = Rounding::NearestEven, SplitMix64* rng = nullptr) {
    if (!(d > 0.0)) {
        std::fill(out, out + n, std::uint8_t(0));
        return;
    }
    if (rnd == Rounding::Stochastic && rng != nullptr)
        for (int k = 0; k < n; ++k) out[k] = e2m1_encode_sr(x[k] / d, rng->uniform());
    else
        for (int k = 0; k < n; ++k) out[k] = e2m1_encode(x[k] / d);
}

//! Encode one row (`cols` values, zero padded to padded_cols(cols)) into
//! `codes` (padded / 2 bytes) and `scales` (fp4: 4 bytes; else one a
//! block); `g` is nvfp4's global scale. `tmp` is scratch.
inline void encode_row(Format f, const double* x, int cols, int block, float g,
                       std::uint8_t* codes, std::uint8_t* scales,
                       std::vector<std::uint8_t>& tmp,
                       Rounding rnd = Rounding::NearestEven, SplitMix64* rng = nullptr) {
    const int padded = padded_cols(cols);
    tmp.assign(static_cast<std::size_t>(padded), 0);
    if (f == Format::FP4) {
        const double amax = detail::absmax(x, static_cast<std::size_t>(cols));
        float s = static_cast<float>(amax / kE2M1Max);
        if (!(s > 0.0f) || !std::isfinite(s)) s = 1.0f;
        detail::put_f32(s, scales);
        encode_elements(x, cols, static_cast<double>(s), tmp.data(), rnd, rng);
    } else {
        const int nb = padded / block;
        for (int b = 0; b < nb; ++b) {
            const int k0 = std::min(cols, b * block);
            const int k1 = std::min(cols, k0 + block);
            scales[b] = block_scale_code(f, detail::absmax(x + k0, static_cast<std::size_t>(k1 - k0)), g);
            encode_elements(x + k0, k1 - k0, block_scale_value(f, scales[b], g), tmp.data() + k0, rnd, rng);
        }
    }
    pack_nibbles(tmp.data(), static_cast<std::size_t>(padded), codes);
}

//! Decode one row into `out` (`cols` values).
inline void decode_row(Format f, const std::uint8_t* codes, const std::uint8_t* scales,
                       int cols, int block, float g, double* out) {
    if (f == Format::FP4) {
        const double s = static_cast<double>(detail::get_f32(scales));
        for (int k = 0; k < cols; ++k) out[k] = e2m1_decode(nibble(codes, static_cast<std::size_t>(k))) * s;
        return;
    }
    for (int k = 0; k < cols; ++k)
        out[k] = e2m1_decode(nibble(codes, static_cast<std::size_t>(k))) *
                 block_scale_value(f, scales[k / block], g);
}

//! A zeroed tensor of the given shape.
inline Fp4Tensor make_tensor(int rows, int cols, Format f) {
    Fp4Tensor t;
    t.format = f;
    t.rows = rows;
    t.cols = cols;
    t.block = block_size(f, cols);
    t.codes.assign(static_cast<std::size_t>(rows) * t.row_bytes(), 0);
    t.scales.assign(static_cast<std::size_t>(rows) * t.row_scale_bytes(), 0);
    return t;
}

//! Quantise a row-major `rows x cols` matrix, blocks along each row.
inline Fp4Tensor quantize(const double* W, int rows, int cols, Format f,
                          Rounding rnd = Rounding::NearestEven, SplitMix64* rng = nullptr) {
    Fp4Tensor t = make_tensor(rows, cols, f);
    if (f == Format::NVFP4)
        t.tensor_scale = nvfp4_tensor_scale(
                detail::absmax(W, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)));
    std::vector<std::uint8_t> tmp;
    for (int r = 0; r < rows; ++r)
        encode_row(f, W + static_cast<std::size_t>(r) * cols, cols, t.block, t.tensor_scale,
                   t.codes.data() + static_cast<std::size_t>(r) * t.row_bytes(),
                   t.scales.data() + static_cast<std::size_t>(r) * t.row_scale_bytes(), tmp, rnd, rng);
    return t;
}

//! Quantise with 2-D square tiles (`block x block`: 16 x 16 for nvfp4,
//! 32 x 32 for mxfp4), one scale a tile repeated in the tile's rows, so the
//! transpose quantises to the same values (NVIDIA's NVFP4 training recipe
//! for weights, arXiv 2509.25149 section 4.3). Round-to-nearest-even.
inline Fp4Tensor quantize_2d(const double* W, int rows, int cols, Format f) {
    if (f == Format::FP4) throw std::runtime_error("FP4: 2-D scaling needs mxfp4 or nvfp4");
    Fp4Tensor t = make_tensor(rows, cols, f);
    if (f == Format::NVFP4)
        t.tensor_scale = nvfp4_tensor_scale(
                detail::absmax(W, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)));
    const int b = t.block, nb = t.blocks_per_row();
    for (int r0 = 0; r0 < rows; r0 += b) {
        const int r1 = std::min(rows, r0 + b);
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = std::min(cols, kb * b), k1 = std::min(cols, k0 + b);
            double amax = 0.0;
            for (int r = r0; r < r1; ++r)
                amax = std::max(amax, detail::absmax(W + static_cast<std::size_t>(r) * cols + k0,
                                                     static_cast<std::size_t>(k1 - k0)));
            const std::uint8_t code = block_scale_code(f, amax, t.tensor_scale);
            for (int r = r0; r < r1; ++r) t.scales[static_cast<std::size_t>(r) * nb + kb] = code;
        }
    }
    std::vector<std::uint8_t> tmp(t.padded());
    for (int r = 0; r < rows; ++r) {
        std::fill(tmp.begin(), tmp.end(), std::uint8_t(0));
        const double* x = W + static_cast<std::size_t>(r) * cols;
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = std::min(cols, kb * b), k1 = std::min(cols, k0 + b);
            encode_elements(x + k0, k1 - k0,
                            block_scale_value(f, t.scales[static_cast<std::size_t>(r) * nb + kb],
                                              t.tensor_scale),
                            tmp.data() + k0);
        }
        pack_nibbles(tmp.data(), tmp.size(), t.codes.data() + static_cast<std::size_t>(r) * t.row_bytes());
    }
    return t;
}

//! The exact transpose of a 2-D scaled tensor (quantize_2d): the same codes,
//! each tile keeping its scale.
inline Fp4Tensor transpose_2d(const Fp4Tensor& t) {
    Fp4Tensor u = make_tensor(t.cols, t.rows, t.format);
    u.tensor_scale = t.tensor_scale;
    const int b = t.block;
    if (u.block != b) throw std::runtime_error("FP4: transpose_2d needs a 2-D scaled tensor");
    const int nbt = t.blocks_per_row(), nbu = u.blocks_per_row();
    std::vector<std::uint8_t> tmp(u.padded());
    for (int k = 0; k < t.cols; ++k) {
        std::fill(tmp.begin(), tmp.end(), std::uint8_t(0));
        for (int r = 0; r < t.rows; ++r)
            tmp[static_cast<std::size_t>(r)] = nibble(
                    t.codes.data() + static_cast<std::size_t>(r) * t.row_bytes(), static_cast<std::size_t>(k));
        pack_nibbles(tmp.data(), tmp.size(), u.codes.data() + static_cast<std::size_t>(k) * u.row_bytes());
        for (int rb = 0; rb < nbu; ++rb)
            u.scales[static_cast<std::size_t>(k) * nbu + rb] =
                    rb * b < t.rows ? t.scales[static_cast<std::size_t>(rb * b) * nbt + k / b]
                                    : block_scale_code(t.format, 0.0, t.tensor_scale);
    }
    return u;
}

//! Decode the whole matrix into `out` (`rows * cols` doubles, row-major).
inline void dequantize(const Fp4Tensor& t, double* out) {
    for (int r = 0; r < t.rows; ++r)
        decode_row(t.format, t.codes.data() + static_cast<std::size_t>(r) * t.row_bytes(),
                   t.scales.data() + static_cast<std::size_t>(r) * t.row_scale_bytes(), t.cols,
                   t.block, t.tensor_scale, out + static_cast<std::size_t>(r) * t.cols);
}

//! Encode and decode each row of `A` (`rows x cols`) in place: the
//! activation fake-quantisation (nvfp4 with a per-row global scale).
inline void fake_quantize_rows(double* A, int rows, int cols, Format f) {
    const int block = block_size(f, cols);
    const int padded = padded_cols(cols);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(padded / 2)),
            scales(f == Format::FP4 ? 4 : static_cast<std::size_t>(padded / block)), tmp;
    for (int r = 0; r < rows; ++r) {
        double* row = A + static_cast<std::size_t>(r) * cols;
        const float g = f == Format::NVFP4
                                ? nvfp4_tensor_scale(detail::absmax(row, static_cast<std::size_t>(cols)))
                                : 1.0f;
        encode_row(f, row, cols, block, g, codes.data(), scales.data(), tmp);
        decode_row(f, codes.data(), scales.data(), cols, block, g, row);
    }
}

// ------------------------------------------------------------------ model

//! One dense layer with packed FP4 weights -- or, for a layer the FP4
//! training recipe keeps in high precision, float64 weights.
struct Fp4Layer {
    int n_in = 0;
    int n_out = 0;
    Activation activation = Activation::Identity;
    Fp4Tensor weight;                 //!< n_out x n_in (unless full_precision)
    std::vector<double> bias;         //!< n_out, double
    bool full_precision = false;      //!< weights in `weight_f64` instead
    std::vector<double> weight_f64;   //!< n_out x n_in, when full_precision
};

//! The FP4 layers plus the (double) scalers of the source model.
struct Fp4Model {
    Format format = Format::FP4;
    bool quantize_activations = false;
    std::vector<Fp4Layer> layers;
    StandardScaler x_scaler;
    StandardScaler y_scaler;

    int n_inputs() const { return layers.empty() ? 0 : layers.front().n_in; }
    int n_outputs() const { return layers.empty() ? 0 : layers.back().n_out; }
    //! Packed codes + scales of every layer.
    std::size_t weight_bytes() const {
        std::size_t n = 0;
        for (const auto& l : layers)
            n += l.full_precision ? l.weight_f64.size() * sizeof(double) : l.weight.bytes();
        return n;
    }
    //! Weights (before packing).
    std::size_t n_weights() const {
        std::size_t n = 0;
        for (const auto& l : layers) n += static_cast<std::size_t>(l.n_in) * l.n_out;
        return n;
    }
};

//! Quantise a model's weights once (blocks along each row).
inline Fp4Model quantize(const MlpModel& m, Format f, bool quantize_activations) {
    m.validate();
    Fp4Model out;
    out.format = f;
    out.quantize_activations = quantize_activations;
    out.x_scaler = m.x_scaler;
    out.y_scaler = m.y_scaler;
    for (const DenseLayer& l : m.layers) {
        for (double w : l.weight)
            if (!std::isfinite(w)) throw std::runtime_error("FP4: the weights must be finite");
        Fp4Layer q;
        q.n_in = l.n_in;
        q.n_out = l.n_out;
        q.activation = l.activation;
        q.bias = l.bias;
        q.weight = quantize(l.weight.data(), l.n_out, l.n_in, f);
        out.layers.push_back(std::move(q));
    }
    return out;
}

//! The float reference forward pass, in the model's physical units.
/*! Each layer's weights are decoded into a double matrix and multiplied in
    double; with `quantize_activations` the layer input is first
    fake-quantised per row. For tests; QuantizedNeuralNet runs the kernels. */
template <class Gemm = mlpcore::PortableGemm>
inline void predict_reference(const Fp4Model& m, const double* X, int n_rows, std::vector<double>& y) {
    y.clear();
    if (n_rows <= 0 || m.layers.empty()) return;
    const std::size_t rows = static_cast<std::size_t>(n_rows);
    std::vector<double> a(X, X + rows * static_cast<std::size_t>(m.n_inputs())), z, W;
    mlpcore::detail::scale_in(a, n_rows, m.n_inputs(), m.x_scaler);
    for (const Fp4Layer& l : m.layers) {
        W.resize(static_cast<std::size_t>(l.n_out) * static_cast<std::size_t>(l.n_in));
        if (l.full_precision) {
            W = l.weight_f64;
        } else {
            dequantize(l.weight, W.data());
            if (m.quantize_activations) fake_quantize_rows(a.data(), n_rows, l.n_in, m.format);
        }
        z.resize(rows * static_cast<std::size_t>(l.n_out));
        Gemm::nt(n_rows, l.n_out, l.n_in, a.data(), W.data(), z.data());
        for (std::size_t r = 0; r < rows; ++r) {
            double* zr = z.data() + r * static_cast<std::size_t>(l.n_out);
            for (int o = 0; o < l.n_out; ++o) zr[o] += l.bias[static_cast<std::size_t>(o)];
        }
        a.resize(z.size());
        mlpcore::act_apply(z.data(), a.data(), z.size(), l.activation);
    }
    y.swap(a);
    mlpcore::detail::unscale_out(y, n_rows, m.n_outputs(), m.y_scaler);
}

}  // namespace mlpfp4
}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPFP4_H

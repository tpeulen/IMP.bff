// FP4 (E2M1) weights for bff's network: the codecs and formats of
// internal/MlpFp4.h, the integer-SIMD kernels of internal/MlpFp4Kernels.h,
// and the pieces of FP4 training in internal/MlpFp4Train.h.
//
// Built as a module test by IMP's CMake, and compiled standalone -- four ways
// -- by test/test_neural_net_fp4.py against bff's headers only:
//
//   c++ -std=c++17 -O2 -I <dir holding IMP/bff/internal> \
//       test/cpp_snippets/test_fp4_kernels.cpp -o /tmp/test_fp4_kernels
//
// with -march=armv8.2-a+dotprod (NEON + dotprod), -march=armv8-a (NEON
// without dotprod), -arch x86_64 -mavx2 -mfma (AVX2, run under Rosetta 2 on
// Apple Silicon) and -DIMPBFF_FP4_NO_SIMD (the generic scalar code). The
// first output line names the compiled variant. Every variant must give the
// generic kernel's result exactly: the integer sums are exact and the float
// scaling is shared code.
//
// Lines starting "  time" are measurements, reported and never asserted.
// Exit status is the number of failed checks.

#include "IMP/bff/internal/MlpCore.h"
#include "IMP/bff/internal/MlpGemm.h"
#include "IMP/bff/internal/MlpFp4.h"
#include "IMP/bff/internal/MlpFp4Kernels.h"
#include "IMP/bff/internal/MlpFp4Train.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using IMP::bff::internal::Activation;
using IMP::bff::internal::DenseLayer;
using IMP::bff::internal::MatGemm;
using IMP::bff::internal::MlpModel;
namespace mc = IMP::bff::internal::mlpcore;
namespace f4 = IMP::bff::internal::mlpfp4;
namespace kn = IMP::bff::internal::mlpfp4::kern;
namespace tr = IMP::bff::internal::mlpfp4::train;

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL  %s\n", what.c_str());
        ++g_failures;
    }
}
static void report(bool ok, const char* what, double value, double tol) {
    if (!ok) {
        std::printf("  FAIL  %s (%.3g > %.3g)\n", what, value, tol);
        ++g_failures;
    } else {
        std::printf("  ok    %s (%.3g)\n", what, value);
    }
}

struct Lcg {
    std::uint64_t s;
    explicit Lcg(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s >> 11;
    }
    double uniform() {  // in (-1, 1)
        return static_cast<double>(next() & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 52) - 1.0;
    }
};

// --------------------------------------------------------------------------
// 1. E2M1
// --------------------------------------------------------------------------
static void test_e2m1() {
    std::printf("E2M1 codec\n");
    const double want[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
    int bad = 0;
    for (int c = 0; c < 16; ++c) {
        // independent decode from the bit fields: s ee m, bias 1
        const int s = c >> 3, e = (c >> 1) & 3, m = c & 1;
        const double v = (s ? -1.0 : 1.0) * (e == 0 ? 0.5 * m : std::ldexp(1.0 + 0.5 * m, e - 1));
        if (f4::e2m1_decode(static_cast<std::uint8_t>(c)) != want[c] || v != want[c]) ++bad;
        if (f4::e2m1_encode(want[c]) != c) ++bad;  // round trip, -0 -> 8 included
        if (std::signbit(f4::e2m1_decode(static_cast<std::uint8_t>(c))) != (s == 1)) ++bad;
    }
    report(bad == 0, "16 codes decode (bit fields) and round-trip", bad, 0);
    struct T { double x; int code; };
    const T ties[] = {{0.25, 0}, {0.75, 2}, {1.25, 2}, {1.75, 4}, {2.5, 4}, {3.5, 6}, {5.0, 6},
                      {-0.25, 8}, {-0.75, 10}, {-5.0, 14}, {100.0, 7}, {-100.0, 15}, {6.0, 7},
                      {-0.1, 8}, {0.2500001, 1}, {0.7499999, 1}, {4.9999, 6}, {5.0001, 7},
                      {std::numeric_limits<double>::infinity(), 7}};
    bad = 0;
    for (const T& t : ties)
        if (f4::e2m1_encode(t.x) != t.code) {
            std::printf("    e2m1(%g) = %d, want %d\n", t.x, f4::e2m1_encode(t.x), t.code);
            ++bad;
        }
    report(bad == 0, "ties to even (0.25->0, 0.75->1, 5->4, 2.5->2), saturation, sign", bad, 0);
    // brute force: nearest, and at a tie the even mantissa
    Lcg rng(1);
    bad = 0;
    for (int i = 0; i < 200000; ++i) {
        const double x = 7.0 * rng.uniform();
        const int c = f4::e2m1_encode(x);
        const double d = std::abs(f4::e2m1_decode(static_cast<std::uint8_t>(c)) - std::min(6.0, std::max(-6.0, x)));
        for (int k = 0; k < 16; ++k) {
            if (std::signbit(want[k]) != std::signbit(x)) continue;
            const double dk = std::abs(want[k] - std::min(6.0, std::max(-6.0, x)));
            if (dk < d || (dk == d && (k & 1) == 0 && (c & 1) == 1)) { ++bad; break; }
        }
    }
    report(bad == 0, "random values encode to the nearest code (200000)", bad, 0);
    std::uint8_t packed[2];
    const std::uint8_t codes[3] = {1, 2, 3};
    f4::pack_nibbles(codes, 3, packed);
    check(packed[0] == 0x21 && packed[1] == 0x03, "packing: element 0 in the low nibble");
    check(f4::nibble(packed, 0) == 1 && f4::nibble(packed, 1) == 2 && f4::nibble(packed, 2) == 3,
          "nibble() reads the packing back");
    std::printf("  ok    packing (element 2i low nibble, 2i+1 high)\n");
}

// --------------------------------------------------------------------------
// 2. E4M3 (e4m3fn) and E8M0
// --------------------------------------------------------------------------
static void test_e4m3_e8m0() {
    std::printf("E4M3 and E8M0 codecs\n");
    int bad = 0, n_finite = 0;
    double prev = -1.0;
    for (int c = 0; c < 256; ++c) {
        const double v = f4::e4m3_decode(static_cast<std::uint8_t>(c));
        const bool nan = (c & 0x7F) == 0x7F;
        if (nan != std::isnan(v)) ++bad;
        if (nan) continue;
        ++n_finite;
        if (f4::e4m3_encode(v) != c) ++bad;  // round trip, -0 -> 0x80 included
        if (c < 0x80) {  // positive codes are increasing
            if (!(v > prev) && c > 0) ++bad;
            prev = v;
        }
    }
    report(bad == 0 && n_finite == 254, "256 codes: 254 finite round-trip, NaN = S.1111.111, monotone", bad, 0);
    check(f4::e4m3_decode(0x7E) == 448.0, "max finite 448 (0x7E)");
    check(f4::e4m3_decode(0x01) == std::ldexp(1.0, -9), "smallest subnormal 2^-9");
    check(f4::e4m3_decode(0x07) == 7 * std::ldexp(1.0, -9), "largest subnormal 7 * 2^-9");
    check(f4::e4m3_decode(0x08) == std::ldexp(1.0, -6), "smallest normal 2^-6");
    check(f4::e4m3_decode(0x38) == 1.0, "1.0 is 0x38");
    check(std::signbit(f4::e4m3_decode(0x80)) && f4::e4m3_decode(0x80) == 0.0, "0x80 is -0");
    check(f4::e4m3_encode(1000.0) == 0x7E && f4::e4m3_encode(-1e9) == 0xFE, "saturates to +-448");
    check(f4::e4m3_encode(464.0) == 0x7E, "464 (tie 448/480) -> 448");
    check(f4::e4m3_encode(std::nan("")) == 0x7F, "NaN -> 0x7F");
    check(f4::e4m3_encode(std::ldexp(1.0, -10)) == 0x00, "2^-10 (tie 0 / 2^-9) -> 0");
    check(f4::e4m3_encode(std::ldexp(1.01, -10)) == 0x01, "just above 2^-10 -> 2^-9");
    // midpoints of neighbours go to the even mantissa
    bad = 0;
    for (int c = 0; c + 1 < 0x7F; ++c) {
        const double mid = 0.5 * (f4::e4m3_decode(static_cast<std::uint8_t>(c)) +
                                  f4::e4m3_decode(static_cast<std::uint8_t>(c + 1)));
        const int want = (c % 2 == 0) ? c : c + 1;
        if (f4::e4m3_encode(mid) != want) ++bad;
        if (f4::e4m3_encode(std::nextafter(mid, 0.0)) != c) ++bad;
        if (f4::e4m3_encode(std::nextafter(mid, 1e9)) != c + 1) ++bad;
    }
    report(bad == 0, "E4M3 midpoints tie to even, neighbours round to nearest", bad, 0);
    bad = 0;
    for (int c = 0; c < 255; ++c) {
        if (f4::e8m0_decode(static_cast<std::uint8_t>(c)) != std::ldexp(1.0, c - 127)) ++bad;
        if (f4::e8m0_from_exponent(c - 127) != c) ++bad;
    }
    check(std::isnan(f4::e8m0_decode(0xFF)), "E8M0 255 is NaN");
    check(f4::e8m0_from_exponent(-500) == 0 && f4::e8m0_from_exponent(500) == 254, "E8M0 clamps");
    report(bad == 0, "E8M0: 2^(code - 127), 255 codes", bad, 0);
}

// --------------------------------------------------------------------------
// 3. The block recipes
// --------------------------------------------------------------------------
static void test_formats() {
    std::printf("weight formats\n");
    // mxfp4: X = 2^(floor(log2 amax) - 2)
    std::vector<double> row(32, 0.0);
    row[0] = 5.0; row[1] = -1.3; row[2] = 0.3;
    f4::Fp4Tensor t = f4::quantize(row.data(), 1, 32, f4::Format::MXFP4);
    check(t.scales[0] == 127, "mxfp4 amax 5 -> X = 2^0");
    row[0] = 7.9;  // floor(log2 7.9) = 2 -> X = 1, 7.9 saturates to 6
    t = f4::quantize(row.data(), 1, 32, f4::Format::MXFP4);
    check(t.scales[0] == 127 && f4::nibble(t.codes.data(), 0) == 7, "mxfp4 amax 7.9 -> X = 1, clamped to 6");
    row[0] = 8.0;
    t = f4::quantize(row.data(), 1, 32, f4::Format::MXFP4);
    check(t.scales[0] == 128, "mxfp4 amax 8 -> X = 2");
    std::fill(row.begin(), row.end(), 0.0);
    t = f4::quantize(row.data(), 1, 32, f4::Format::MXFP4);
    check(t.scales[0] == 0, "mxfp4 zero block -> code 0");
    // nvfp4: g = amax / 2688, S = e4m3(amax_b / (6 g))
    std::vector<double> w(2 * 20);
    Lcg rng(3);
    for (auto& v : w) v = rng.uniform();
    w[5] = 2688.0 * 0.001;  // absmax
    t = f4::quantize(w.data(), 2, 20, f4::Format::NVFP4);
    check(t.tensor_scale == static_cast<float>(2.688 / 2688.0), "nvfp4 tensor scale absmax / (6 * 448)");
    check(t.blocks_per_row() == 2 && t.row_bytes() == 16 && t.scales[0] == 0x7E,
          "nvfp4 layout: rows padded to 32, the absmax block scale is 448");
    check(t.consistent(), "nvfp4 tensor consistent");
    std::vector<double> back(w.size());
    f4::dequantize(t, back.data());
    report(std::abs(back[5] - w[5]) <= 1e-7 * w[5], "nvfp4 absmax element = 6 * 448 * g (float32 g)",
           std::abs(back[5] - w[5]) / w[5], 1e-7);
    // error bounds per format
    const int R = 64, C = 96;
    std::vector<double> W(static_cast<std::size_t>(R) * C), D(W.size());
    for (auto& v : W) v = rng.uniform() * (1.0 + 3.0 * std::abs(rng.uniform()));
    for (f4::Format f : {f4::Format::FP4, f4::Format::MXFP4, f4::Format::NVFP4}) {
        t = f4::quantize(W.data(), R, C, f);
        f4::dequantize(t, D.data());
        double err = 0.0, amax = 0.0;
        for (std::size_t i = 0; i < W.size(); ++i) {
            err = std::max(err, std::abs(D[i] - W[i]));
            amax = std::max(amax, std::abs(W[i]));
        }
        const double bits = 8.0 * static_cast<double>(t.bytes()) / static_cast<double>(W.size());
        std::printf("  ok    %s: max error %.3g of absmax, %.3f bits/weight\n",
                    f4::format_to_string(f).c_str(), err / amax, bits);
        // half the top step: 1/6 of the absmax (fp4, nvfp4); mxfp4 can also
        // clamp [6, 8) X to 6, up to 1/4
        check(err / amax <= (f == f4::Format::MXFP4 ? 0.25 : 1.0 / 6.0) + 1e-12, "max error bound");
    }
    // 2-D (16 x 16) scaling: the transpose quantises to the same values
    for (f4::Format f : {f4::Format::NVFP4, f4::Format::MXFP4}) {
        const f4::Fp4Tensor q = f4::quantize_2d(W.data(), R, C, f);
        const f4::Fp4Tensor qt = f4::transpose_2d(q);
        std::vector<double> a(W.size()), b(W.size());
        f4::dequantize(q, a.data());
        f4::dequantize(qt, b.data());
        int bad = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                if (a[static_cast<std::size_t>(r) * C + c] != b[static_cast<std::size_t>(c) * R + r]) ++bad;
        report(bad == 0, (f4::format_to_string(f) + " 2-D tiles: W and W^T quantise identically").c_str(), bad, 0);
    }
    // stochastic rounding is unbiased
    f4::SplitMix64 sm(7);
    for (double x : {0.3, 1.1, -2.6, 4.7}) {
        double mean = 0.0;
        const int n = 200000;
        for (int i = 0; i < n; ++i) mean += f4::e2m1_decode(f4::e2m1_encode_sr(x, sm.uniform()));
        mean /= n;
        report(std::abs(mean - x) < 0.01, "stochastic rounding: mean of 2e5 roundings", std::abs(mean - x), 0.01);
    }
}

// --------------------------------------------------------------------------
// 4. Kernels: every variant equals the generic one exactly
// --------------------------------------------------------------------------
static void test_variants() {
    std::printf("kernel variant '%s' vs generic\n", kn::kernel_name());
    Lcg rng(11);
    int bad = 0;
    for (int ng = 1; ng <= 9; ++ng) {
        std::vector<std::uint8_t> w(16 * ng), x(16 * ng);
        std::vector<std::int8_t> a(32 * ng);
        for (auto& v : w) v = static_cast<std::uint8_t>(rng.next());
        for (auto& v : x) v = static_cast<std::uint8_t>(rng.next());
        for (auto& v : a) v = static_cast<std::int8_t>(static_cast<int>(rng.next() % 255) - 127);
        std::vector<std::int32_t> s1(2 * ng), s2(2 * ng);
        kn::isums_q8_generic(w.data(), a.data(), ng, s1.data());
        kn::isums_q8_simd(w.data(), a.data(), ng, s2.data());
        if (s1 != s2) ++bad;
        // the generic sum against a plain element loop
        for (int s = 0; s < 2 * ng; ++s) {
            std::int32_t ref = 0;
            for (int k = 16 * s; k < 16 * s + 16; ++k)
                ref += static_cast<int>(2.0 * f4::e2m1_decode(f4::nibble(w.data(), k))) * a[kn::permuted(k)];
            if (ref != s1[s]) ++bad;
        }
        kn::isums_fp4_generic(w.data(), x.data(), ng, s1.data());
        kn::isums_fp4_simd(w.data(), x.data(), ng, s2.data());
        if (s1 != s2) ++bad;
        for (int s = 0; s < 2 * ng; ++s) {
            std::int32_t ref = 0;
            for (int k = 16 * s; k < 16 * s + 16; ++k)
                ref += static_cast<int>(4 * f4::e2m1_decode(f4::nibble(w.data(), k)) *
                                        f4::e2m1_decode(f4::nibble(x.data(), k)));
            if (ref != s1[s]) ++bad;
        }
    }
    report(bad == 0, "integer sums: SIMD == generic == element loop (FP4 x int8, FP4 x FP4)", bad, 0);
    // GEMMs: bit-identical to the generic, and equal to double on the same operands
    const int M = 37, N = 29, K = 200;
    std::vector<double> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(N) * K);
    for (auto& v : A) v = rng.uniform() * 2.0;
    for (auto& v : B) v = rng.uniform() * 0.5;
    for (f4::Format f : {f4::Format::FP4, f4::Format::MXFP4, f4::Format::NVFP4}) {
        const std::string fn = f4::format_to_string(f);
        const f4::Fp4Tensor wt = f4::quantize(B.data(), N, K, f);
        const kn::Fp4Rows W = kn::rows_of(wt);
        kn::Q8Rows q8;
        kn::quantize_q8(A.data(), M, K, kn::q8_block(f), q8);
        const kn::Fp4Rows X = kn::quantize_rows(A.data(), M, K, f);
        std::vector<double> c1(static_cast<std::size_t>(M) * N), c2(c1.size()), c3(c1.size()), c4(c1.size());
        kn::gemm_q8<true>(q8, W, c1.data());
        kn::gemm_q8<false>(q8, W, c2.data());
        kn::gemm_fp4<true>(X, W, c3.data());
        kn::gemm_fp4<false>(X, W, c4.data());
        check(std::memcmp(c1.data(), c2.data(), c1.size() * sizeof(double)) == 0, fn + " gemm_q8 SIMD == generic");
        check(std::memcmp(c3.data(), c4.data(), c3.size() * sizeof(double)) == 0, fn + " gemm_fp4 SIMD == generic");
        // double reference on the same (decoded) operands
        std::vector<double> Wd(B.size()), Xd(A.size()), Ad(A.size()), ref(c1.size()), ref2(c1.size());
        f4::dequantize(wt, Wd.data());
        for (int r = 0; r < M; ++r)
            for (int k = 0; k < K; ++k) {
                const std::size_t i = static_cast<std::size_t>(r) * K + k;
                Ad[i] = q8.q[static_cast<std::size_t>(r) * q8.kp + kn::permuted(k)] *
                        static_cast<double>(q8.sc[static_cast<std::size_t>(r) * q8.n_sub() + k / 16]);
                Xd[i] = f4::e2m1_decode(f4::nibble(X.codes + r * X.stride, k)) * 2.0 *
                        static_cast<double>(X.sc[static_cast<std::size_t>(r) * X.n_sub() + k / 16]);
            }
        mc::PortableGemm::nt(M, N, K, Ad.data(), Wd.data(), ref.data());
        mc::PortableGemm::nt(M, N, K, Xd.data(), Wd.data(), ref2.data());
        double e1 = 0, e2 = 0, amax = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            e1 = std::max(e1, std::abs(c2[i] - ref[i]));
            e2 = std::max(e2, std::abs(c4[i] - ref2[i]));
            amax = std::max(amax, std::abs(ref[i]));
        }
        report(e1 / amax < 1e-5, (fn + " gemm_q8 == double GEMM of the same operands").c_str(), e1 / amax, 1e-5);
        report(e2 / amax < 1e-5, (fn + " gemm_fp4 == double GEMM of the same operands").c_str(), e2 / amax, 1e-5);
    }
}

// --------------------------------------------------------------------------
// 5. The network: kernels vs the float reference, and speed
// --------------------------------------------------------------------------
static MlpModel make_model(const std::vector<int>& dims, Lcg& rng) {
    MlpModel m;
    for (size_t l = 0; l + 1 < dims.size(); ++l) {
        DenseLayer d;
        d.n_in = dims[l];
        d.n_out = dims[l + 1];
        d.activation = l + 2 < dims.size() ? Activation::Tanh : Activation::Identity;
        d.weight.resize(static_cast<size_t>(d.n_out) * d.n_in);
        d.bias.resize(static_cast<size_t>(d.n_out));
        for (auto& w : d.weight) w = rng.uniform() * std::sqrt(3.0 / d.n_in);
        for (auto& b : d.bias) b = 0.1 * rng.uniform();
        m.layers.push_back(d);
    }
    return m;
}

template <class F>
static double best_seconds(int reps, F f) {
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

static double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        m = std::max(m, std::abs(b[i]));
    }
    return e / m;
}

static void test_network() {
    std::printf("network: kernels vs float reference\n");
    Lcg rng(21);
    const MlpModel m = make_model({24, 256, 256, 128, 8}, rng);
    const int n = 256;
    std::vector<double> X(static_cast<size_t>(n) * 24);
    for (auto& v : X) v = 2.0 * rng.uniform();
    std::vector<double> yd, d1, d2;
    mc::model_predict<MatGemm>(m, X.data(), n, 0, nullptr, yd, d1, d2);
    for (f4::Format f : {f4::Format::FP4, f4::Format::MXFP4, f4::Format::NVFP4}) {
        const std::string fn = f4::format_to_string(f);
        for (bool qa : {false, true}) {
            const f4::Fp4Model q = f4::quantize(m, f, qa);
            std::vector<double> yk, yr;
            kn::predict(q, X.data(), n, yk);
            f4::predict_reference<MatGemm>(q, X.data(), n, yr);
            const double e_ref = max_rel(yk, yr), e_dbl = max_rel(yk, yd), e_rd = max_rel(yr, yd);
            std::printf("  ok    %s %s: vs double %.3g (float reference %.3g), kernels vs reference %.3g\n",
                        fn.c_str(), qa ? "W4A4" : "W4A8", e_dbl, e_rd, e_ref);
            // W4A4: identical operands, only the float accumulation differs;
            // W4A8: the int8 activations add their own (small) error.
            check(qa ? e_ref < 1e-5 : e_ref < 0.03, fn + (qa ? " W4A4" : " W4A8") + " kernels match the reference");
            std::vector<double> y1;
            kn::predict(q, X.data() + 24 * 7, 1, y1);
            check(std::memcmp(y1.data(), yk.data() + 8 * 7, 8 * sizeof(double)) == 0,
                  fn + " a row does not depend on its batch");
        }
    }
    std::printf("  time  kernel variant: %s; net 24-256-256-128-8 (us per call, best of 20)\n", kn::kernel_name());
    for (int b : {1, 32, 256}) {
        std::vector<double> y;
        const double td = best_seconds(20, [&] { mc::model_predict<MatGemm>(m, X.data(), b, 0, nullptr, y, d1, d2); });
        std::printf("  time  batch %3d: double MatGemm %8.1f", b, td * 1e6);
        for (f4::Format f : {f4::Format::FP4, f4::Format::MXFP4, f4::Format::NVFP4}) {
            const f4::Fp4Model q = f4::quantize(m, f, false);
            const f4::Fp4Model qa = f4::quantize(m, f, true);
            const double tr = best_seconds(20, [&] { f4::predict_reference<MatGemm>(q, X.data(), b, y); });
            const double tk = best_seconds(20, [&] { kn::predict(q, X.data(), b, y); });
            const double t4 = best_seconds(20, [&] { kn::predict(qa, X.data(), b, y); });
            std::printf(" | %s dequant %.1f, W4A8 %.1f, W4A4 %.1f", f4::format_to_string(f).c_str(),
                        tr * 1e6, tk * 1e6, t4 * 1e6);
        }
        std::printf("\n");
    }
}

// --------------------------------------------------------------------------
// 6. FP4 training pieces (MlpFp4Train.h)
// --------------------------------------------------------------------------
static std::vector<double> decode_rows(const kn::Fp4Rows& R, int K) {
    std::vector<double> out(static_cast<std::size_t>(R.rows) * K);
    for (int r = 0; r < R.rows; ++r)
        for (int k = 0; k < K; ++k)
            out[static_cast<std::size_t>(r) * K + k] =
                    2.0 * f4::e2m1_decode(f4::nibble(R.codes + r * R.stride, k)) *
                    static_cast<double>(R.sc[static_cast<std::size_t>(r) * R.n_sub() + k / 16]);
    return out;
}

static void test_training_pieces() {
    std::printf("FP4 training pieces\n");
    // Hadamard: orthogonal, so the transform is inverted by its transpose
    const tr::Hadamard16 h;
    double worst = 0.0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            double d = 0.0;
            for (int k = 0; k < 16; ++k) d += h.t[i][k] * h.t[j][k];
            worst = std::max(worst, std::abs(d - (i == j ? 1.0 : 0.0)));
        }
    report(worst < 1e-15, "Hadamard T T^T = I", worst, 1e-15);
    Lcg rng(41);
    double x[16], y[16], back[16];
    for (double& v : x) v = rng.uniform();
    h.apply(x, y);
    for (int i = 0; i < 16; ++i) {
        back[i] = 0.0;
        for (int k = 0; k < 16; ++k) back[i] += h.t[k][i] * y[k];
    }
    worst = 0.0;
    for (int i = 0; i < 16; ++i) worst = std::max(worst, std::abs(back[i] - x[i]));
    report(worst < 1e-15, "Hadamard inverted by its transpose", worst, 1e-15);
    // RHT leaves dZ^T A unchanged (before quantisation)
    const int bs = 40, n_out = 24, n_in = 20;
    std::vector<double> dZ(static_cast<std::size_t>(bs) * n_out), A(static_cast<std::size_t>(bs) * n_in);
    for (auto& v : dZ) v = 0.01 * rng.uniform();
    for (auto& v : A) v = rng.uniform();
    std::vector<double> ref(static_cast<std::size_t>(n_out) * n_in);
    mc::PortableGemm::tn(n_out, n_in, bs, dZ.data(), A.data(), ref.data());
    {
        std::vector<double> dzt(static_cast<std::size_t>(n_out) * bs), at(static_cast<std::size_t>(n_in) * bs), t1, t2;
        for (int r = 0; r < bs; ++r) {
            for (int o = 0; o < n_out; ++o) dzt[static_cast<std::size_t>(o) * bs + r] = dZ[static_cast<std::size_t>(r) * n_out + o];
            for (int k = 0; k < n_in; ++k) at[static_cast<std::size_t>(k) * bs + r] = A[static_cast<std::size_t>(r) * n_in + k];
        }
        int K16 = 0;
        tr::rht_rows(dzt.data(), n_out, bs, h, t1, K16);
        tr::rht_rows(at.data(), n_in, bs, h, t2, K16);
        std::vector<double> got(ref.size());
        mc::PortableGemm::nt(n_out, n_in, K16, t1.data(), t2.data(), got.data());
        report(max_rel(got, ref) < 1e-13, "(T dZ)^T (T A) == dZ^T A", max_rel(got, ref), 1e-13);
    }
    // wgrad on the kernels == double product of the same quantised operands
    for (f4::Format f : {f4::Format::NVFP4, f4::Format::MXFP4}) {
        tr::Config c;
        c.format = f;
        f4::SplitMix64 sr(5);
        tr::WgradOperands ops;
        std::vector<double> dW(ref.size());
        tr::wgrad(dZ.data(), A.data(), bs, n_out, n_in, c, sr, h, dW.data(), &ops);
        const int K = ops.x.kp;
        const std::vector<double> X = decode_rows(ops.x, K), Y = decode_rows(ops.y, K);
        std::vector<double> want(ref.size());
        mc::PortableGemm::nt(n_out, n_in, K, X.data(), Y.data(), want.data());
        const std::string fn = f4::format_to_string(f);
        report(max_rel(dW, want) < 1e-5, (fn + " wgrad == float GEMM of the same quantised operands").c_str(),
               max_rel(dW, want), 1e-5);
        std::printf("  ok    %s wgrad vs exact dZ^T A: %.3g relative\n", fn.c_str(), max_rel(dW, ref));
        // dgrad: Q(dZ) Q(W) through the exact transpose
        std::vector<double> W(static_cast<std::size_t>(n_out) * n_in);
        for (auto& v : W) v = 0.3 * rng.uniform();
        const f4::Fp4Tensor wq = f4::quantize_2d(W.data(), n_out, n_in, f);
        const f4::Fp4Tensor wqt = f4::transpose_2d(wq);
        f4::SplitMix64 s1(9), s2(9);
        const kn::Fp4Rows q = kn::quantize_rows(dZ.data(), bs, n_out, f, f4::Rounding::Stochastic, &s1);
        std::vector<double> dA(static_cast<std::size_t>(bs) * n_in), dAref(dA.size()), Wd(W.size());
        kn::gemm_fp4(q, kn::rows_of(wqt), dA.data());
        const kn::Fp4Rows q2 = kn::quantize_rows(dZ.data(), bs, n_out, f, f4::Rounding::Stochastic, &s2);
        check(q2.own == q.own, fn + " stochastic rounding is deterministic per seed");
        f4::dequantize(wq, Wd.data());
        const std::vector<double> dzq = decode_rows(q, n_out);
        mc::PortableGemm::nn(bs, n_in, n_out, dzq.data(), Wd.data(), dAref.data());
        report(max_rel(dA, dAref) < 1e-5, (fn + " dgrad == float GEMM of Q(dZ) and fprop's Q(W)").c_str(),
               max_rel(dA, dAref), 1e-5);
    }
    // one training step twice with the same seed: identical gradients
    Lcg r2(3);
    MlpModel m = make_model({6, 64, 48, 32, 2}, r2);
    std::vector<double> X(static_cast<std::size_t>(32) * 6), dY(static_cast<std::size_t>(32) * 2);
    for (auto& v : X) v = r2.uniform();
    for (auto& v : dY) v = 0.1 * r2.uniform();
    std::vector<double> g1(mc::n_parameters(m.layers)), g2(g1.size());
    for (std::vector<double>* g : {&g1, &g2}) {
        tr::Config c;
        tr::Workspace ws;
        f4::SplitMix64 sr(77);
        tr::quantize_weights(m.layers, c, ws);
        tr::forward(m.layers, c, X.data(), 32, ws);
        tr::backward(m.layers, c, ws, dY.data(), 32, sr, h, g->data());
    }
    check(std::memcmp(g1.data(), g2.data(), g1.size() * sizeof(double)) == 0, "an FP4 step is deterministic per seed");
    // and the FP4 gradient is close to the float64 one (reported)
    {
        std::vector<double> gd(g1.size());
        mc::Workspace ws;
        mc::forward(m.layers, X.data(), 32, ws, 0);
        mc::backward(m.layers, ws, dY.data(), nullptr, nullptr, gd.data());
        double num = 0, den = 0;
        for (std::size_t i = 0; i < gd.size(); ++i) {
            num += (g1[i] - gd[i]) * (g1[i] - gd[i]);
            den += gd[i] * gd[i];
        }
        const double cosang = [&] {
            double d = 0, a = 0, b = 0;
            for (std::size_t i = 0; i < gd.size(); ++i) { d += g1[i] * gd[i]; a += g1[i] * g1[i]; b += gd[i] * gd[i]; }
            return d / std::sqrt(a * b);
        }();
        report(cosang > 0.9, "nvfp4 gradient vs float64 gradient, cosine", cosang, 0.9);
        std::printf("  ok    nvfp4 gradient relative L2 difference %.3g\n", std::sqrt(num / den));
    }
    // the inference model of a trained net reproduces training's fprop
    {
        tr::Config c;
        const f4::Fp4Model q = tr::to_model(m, c);
        tr::Workspace ws;
        tr::quantize_weights(m.layers, c, ws);
        tr::forward(m.layers, c, X.data(), 32, ws);
        std::vector<double> y;
        kn::predict(q, X.data(), 32, y);
        check(std::memcmp(y.data(), ws.output().data(), y.size() * sizeof(double)) == 0,
              "to_model() predicts exactly training's forward pass");
        std::printf("  ok    to_model() == training forward (bit-identical)\n");
    }
}

int main() {
    std::printf("variant %s\n", kn::kernel_name());
    test_e2m1();
    test_e4m3_e8m0();
    test_formats();
    test_variants();
    test_network();
    test_training_pieces();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures;
}

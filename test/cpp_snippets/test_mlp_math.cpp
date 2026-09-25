// MlpMath.h: the vectorised tanh / sigmoid / SiLU against libm, and every
// compiled SIMD variant against the generic one (one committed fingerprint).
//
// Built by test/test_neural_net_fp4.py the same ways as test_fp4_kernels.cpp
// (NEON, AVX2, AVX-512, generic). Prints "variant <name>" first, the measured
// ulp errors, and "<n> failure(s)" last.
#include <IMP/bff/internal/MlpMath.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace mm = IMP::bff::internal::mlpmath;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

static std::uint64_t bits(double x) { std::uint64_t u; std::memcpy(&u, &x, 8); return u; }
//! Distance in units in the last place (0 for equal values, +0 == -0).
static std::int64_t ulp_dist(double a, double b) {
    if (a == b) return 0;
    auto ord = [](double x) {
        const std::int64_t i = static_cast<std::int64_t>(bits(x));
        return i < 0 ? std::numeric_limits<std::int64_t>::min() - i : i;
    };
    const std::int64_t d = ord(a) - ord(b);
    return d < 0 ? -d : d;
}

// The committed fingerprint: FNV-1a over the bits of tanh / sigmoid / SiLU on
// the probe set below; every variant must print the generic build's.
constexpr std::uint64_t kMathFingerprint = 0x3f4f1d48f024ef7fULL;

static std::uint64_t fnv(std::uint64_t h, double v) {
    const std::uint64_t u = bits(v);
    for (int i = 0; i < 8; ++i) { h ^= (u >> (8 * i)) & 0xFF; h *= 0x100000001b3ULL; }
    return h;
}

struct Lcg {
    std::uint64_t s;
    std::uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
    double uniform() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

//! Inputs: dense grid, log-spaced magnitudes both signs, consecutive doubles
//! around the branch point and the saturation, random, and the edge cases.
static std::vector<double> probes() {
    std::vector<double> x;
    for (int i = -2000000; i <= 2000000; ++i) x.push_back(i * 1e-5);        // [-20, 20]
    for (int e = -1074; e <= 110; ++e)                                        // 2^-1074 .. 2^110,
        for (int j = 0; j < 8; ++j) {                                         // exact (no libm in
            const double m = std::ldexp(1.0 + j / 8.0, e);                    // the inputs)
            x.push_back(m); x.push_back(-m);
        }
    for (double c : {0.625, 0.3, 1.0, 19.0, 20.0, 354.0, 709.0, 710.0, 745.0}) {
        double v = c;
        for (int i = 0; i < 20000; ++i) v = std::nextafter(v, 0.0);
        for (int i = 0; i < 40000; ++i) { x.push_back(v); x.push_back(-v); v = std::nextafter(v, 1e300); }
    }
    Lcg r{12345};
    for (int i = 0; i < 1000000; ++i) x.push_back((r.uniform() * 2 - 1) * 30.0);
    for (int i = 0; i < 200000; ++i) {  // random bit patterns: every exponent
        double v; const std::uint64_t u = r.next(); std::memcpy(&v, &u, 8);
        if (v == v) x.push_back(v);
    }
    const double inf = std::numeric_limits<double>::infinity();
    for (double v : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::denorm_min(),
                     -std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
                     std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()})
        x.push_back(v);
    return x;
}

template <class F, class R>
static std::int64_t max_ulp(const char* name, const std::vector<double>& x, const std::vector<double>& y, R ref, F ok) {
    std::int64_t worst = 0; double at = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double want = ref(x[i]);
        if (!ok(x[i])) continue;
        const std::int64_t d = ulp_dist(y[i], want);
        if (d > worst) { worst = d; at = x[i]; }
    }
    std::printf("  %s: max %lld ulp (at x = %.17g)\n", name, static_cast<long long>(worst), at);
    return worst;
}

template <class F>
static double time_ns(F f, std::size_t n) {
    double best = 1e300;
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best * 1e9 / static_cast<double>(n);
}

int main() {
    std::printf("variant %s\n", mm::math_variant());
    const std::vector<double> x = probes();
    const std::size_t n = x.size();
    std::vector<double> yt(n), ys(n), yl(n);
    mm::tanh_n(x.data(), yt.data(), n);
    mm::sigmoid_n(x.data(), ys.data(), n);
    mm::silu_n(x.data(), yl.data(), n);

    // The vector lanes and the scalar code agree to the bit (NaN: both NaN).
    std::size_t diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = mm::tanh(x[i]), b = mm::sigmoid(x[i]), c = mm::silu(x[i]);
        if (bits(a) != bits(yt[i]) || bits(b) != bits(ys[i]) || (bits(c) != bits(yl[i]) && !(c != c && yl[i] != yl[i])))
            ++diff;
    }
    check(diff == 0, "vector lanes == scalar code");

    const auto all = [](double) { return true; };
    const auto normal_out = [](double v) {  // results in the normal range
        return v == v && !(v < -708.0) && std::abs(v) < 1e300;
    };
    check(max_ulp("tanh vs libm", x, yt, [](double v) { return std::tanh(v); }, all) <= 2, "tanh <= 2 ulp vs libm");
    check(max_ulp("sigmoid vs libm exp", x, ys, [](double v) { return 1.0 / (1.0 + std::exp(-v)); }, normal_out) <= 2,
          "sigmoid <= 2 ulp vs libm (normal results)");
    check(max_ulp("silu vs libm exp", x, yl, [](double v) { return v / (1.0 + std::exp(-v)); },
                  [&](double v) { return normal_out(v) && std::abs(v) > 1e-300; }) <= 2,
          "silu <= 2 ulp vs libm (normal results)");

    // Against the true value where long double is wider (x86: 64-bit
    // mantissa tanhl, rounded to double): the error of this tanh itself.
    if (sizeof(long double) > 8)
        check(max_ulp("tanh vs tanhl (true value)", x, yt,
                      [](double v) { return static_cast<double>(std::tanh(static_cast<long double>(v))); }, all) <= 2,
              "tanh <= 2 ulp vs tanhl");

    // Edge cases and symmetry.
    const double inf = std::numeric_limits<double>::infinity();
    check(bits(mm::tanh(-0.0)) == bits(-0.0) && bits(mm::tanh(0.0)) == 0, "tanh(+-0) = +-0");
    check(mm::tanh(inf) == 1.0 && mm::tanh(-inf) == -1.0, "tanh(+-inf) = +-1");
    check(std::isnan(mm::tanh(std::nan(""))) && std::isnan(yt.back() * 0 + mm::tanh(-std::nan(""))), "tanh(NaN) = NaN");
    check(mm::sigmoid(inf) == 1.0 && mm::sigmoid(-inf) == 0.0 && std::isnan(mm::sigmoid(std::nan(""))), "sigmoid edges");
    check(mm::silu(inf) == inf && std::isnan(mm::silu(-inf)) && std::isnan(mm::silu(std::nan(""))), "silu edges (as libm's formula)");
    std::size_t odd = 0, tiny = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (bits(mm::tanh(-x[i])) != (bits(yt[i]) ^ 0x8000000000000000ULL) && x[i] == x[i]) ++odd;
        if (std::abs(x[i]) < 1e-8 && yt[i] != x[i]) ++tiny;
    }
    check(odd == 0, "tanh exactly odd");
    check(tiny == 0, "tanh(x) == x for |x| < 1e-8");
    // Monotone: the dense grid (first 4e6 + 1 probes, increasing) and
    // consecutive doubles across the branch point and saturation.
    std::size_t nonmono = 0;
    for (std::size_t i = 1; i <= 4000000; ++i) nonmono += yt[i] < yt[i - 1];
    for (double c : {0.625, 0.3, 1.0, 19.0}) {
        double v = c, prev = -2;
        for (int i = 0; i < 20000; ++i) v = std::nextafter(v, 0.0);
        for (int i = 0; i < 40000; ++i) { const double t = mm::tanh(v); nonmono += t < prev; prev = t; v = std::nextafter(v, 1e300); }
    }
    std::printf("  tanh non-monotone steps: %zu\n", nonmono);
    check(nonmono == 0, "tanh monotone (grid and consecutive doubles)");
    for (std::size_t i = 1; i <= 4000000; ++i) nonmono += ys[i] < ys[i - 1];
    check(nonmono == 0, "sigmoid monotone on the grid");

    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < n; ++i) { h = fnv(h, yt[i]); h = fnv(h, ys[i]); h = fnv(h, yl[i] == yl[i] ? yl[i] : 0.0); }
    std::printf("  fingerprint %016llx\n", static_cast<unsigned long long>(h));
    check(h == kMathFingerprint, "fingerprint == the generic build's");

    // Speed (reported, not asserted): 1e6 values in [-4, 4].
    std::vector<double> xs(1000000), out(xs.size());
    Lcg r{7};
    for (double& v : xs) v = (r.uniform() * 2 - 1) * 4.0;
    volatile double sink = 0;
    const double t_fast = time_ns([&] { mm::tanh_n(xs.data(), out.data(), xs.size()); sink = sink + out[7]; }, xs.size());
    const double t_libm = time_ns([&] { for (std::size_t i = 0; i < xs.size(); ++i) out[i] = std::tanh(xs[i]); sink = sink + out[7]; }, xs.size());
    const double t_sig = time_ns([&] { mm::sigmoid_n(xs.data(), out.data(), xs.size()); sink = sink + out[7]; }, xs.size());
    const double t_sigl = time_ns([&] { for (std::size_t i = 0; i < xs.size(); ++i) out[i] = 1.0 / (1.0 + std::exp(-xs[i])); sink = sink + out[7]; }, xs.size());
    std::printf("  time tanh %.2f ns (libm %.2f), sigmoid %.2f ns (libm exp %.2f)\n", t_fast, t_libm, t_sig, t_sigl);
    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}

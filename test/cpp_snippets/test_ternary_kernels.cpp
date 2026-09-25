// MlpTernary.h / MlpTernaryTrain.h: BitNet b1.58's quantisers, the TQ2 /
// TQ1 packings, the ternary x int8 kernel against the element-by-element
// reference, predict against float64 arithmetic on the dequantised
// operands, ternary training's pieces, and one fingerprint that every SIMD
// variant must reproduce.
//
// Built by test/test_neural_net_ternary.py the same ways as
// test_fp4_kernels.cpp. Prints "variant <name>" first and "<n> failure(s)"
// last.
#include <IMP/bff/internal/MlpTernary.h>
#include <IMP/bff/internal/MlpTernaryTrain.h>
#include <IMP/bff/internal/MlpGemm.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace IMP::bff::internal;
namespace mt = IMP::bff::internal::mlpternary;
namespace mc = IMP::bff::internal::mlpcore;

static int g_failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %s\n", what);
    }
}

struct Lcg {
    std::uint64_t s;
    explicit Lcg(std::uint64_t x) : s(x) {}
    std::uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
    double uniform() { return static_cast<double>(next() >> 11) / 4503599627370496.0 - 1.0; }  // [-1, 1)
};

static std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ULL; }
    return h;
}

// ---------------------------------------------------------------- 1. quantisers
static void test_quantisers() {
    std::printf("quantisers\n");
    // mean |W| = 1 exactly, so s = 1: w s = +-0.5 ties to 0, +-1.5 to +-2 -> +-1
    const double W[8] = {0.5, -0.5, 1.5, -1.5, 0.25, 0.75, -2.0, 0.5};
    const mt::TernaryTensor t = mt::quantize(W, 2, 4);
    check(t.gamma.size() == 1 && t.gamma[0] == 7.5 / 8.0, "absmean over the tensor");
    // s = 8 / 7.5: 0.5 s = 0.533 -> 1, 0.25 s = 0.267 -> 0, 0.75 s = 0.8 -> 1
    const std::int8_t want[8] = {1, -1, 1, -1, 0, 1, -1, 1};
    check(std::memcmp(t.q.data(), want, 8) == 0, "per-tensor trits");
    const mt::TernaryTensor u = mt::quantize(W, 1, 4);  // first row only: mean 1
    check(u.gamma[0] == 1.0, "absmean exact");
    const std::int8_t want1[4] = {0, 0, 1, -1};
    check(std::memcmp(u.q.data(), want1, 4) == 0, "ties to even: 0.5 -> 0, 1.5 -> 2 -> 1");
    check(mt::trit(0.5000000001, 1.0) == 1 && mt::trit(-0.4999999999, 1.0) == 0 && mt::trit(1e300, 1.0) == 1 &&
              mt::trit(-1e300, 1.0) == -1 && mt::trit(std::nan(""), 1.0) == 0,
          "trit: rounding, clamp, NaN -> 0");
    const double Z[4] = {0, 0, 0, 0};
    const mt::TernaryTensor z = mt::quantize(Z, 2, 2);
    check(z.gamma[0] == mt::kEps && z.q[0] == 0, "all-zero tensor: gamma = 1e-5, trits 0");
    const mt::TernaryTensor r = mt::quantize(W, 2, 4, mt::Scale::Row);
    check(r.gamma.size() == 2 && r.gamma[0] == 1.0 && r.gamma[1] == 3.5 / 4.0, "per-row absmean");
    // fixed-order absmean equals the plain sum to rounding
    Lcg rng(3);
    std::vector<double> big(10007);
    double plain = 0.0;
    for (double& v : big) { v = rng.uniform(); plain += std::abs(v); }
    check(std::abs(mt::absmean(big.data(), big.size()) - plain / big.size()) < 1e-13, "absmean ~ sequential sum");
    // activations: the SIMD row quantiser == the scalar one, bit for bit,
    // including ties (x s = k + 0.5), the clamp and NaN
    int bad = 0;
    for (int n : {1, 7, 8, 16, 33, 100, 257}) {
        std::vector<double> x(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) x[static_cast<std::size_t>(k)] = rng.uniform() * 3.0;
        if (n > 8) { x[3] = 0.5; x[4] = -2.5; x[5] = std::nan(""); x[6] = 1e300; }
        std::vector<std::int8_t> a(static_cast<std::size_t>(n)), b(static_cast<std::size_t>(n));
        const std::int32_t sa = mt::va::round_row(x.data(), n, 1.0, a.data());
        const std::int32_t sb = mt::va::round_row_scalar(x.data(), n, 1.0, b.data());
        bad += sa != sb || std::memcmp(a.data(), b.data(), a.size()) != 0;
        if (n > 8) bad += a[3] != 0 || a[4] != -2 || a[5] != 0 || a[6] != 127;
    }
    check(bad == 0, "activation rounding: SIMD == scalar, ties to even, clamp, NaN -> 0");
    std::printf("  ok    quantisers\n");
}

// ---------------------------------------------------------------- 2. packings
static void test_packing() {
    std::printf("packing\n");
    Lcg rng(5);
    int bad = 0;
    for (int cols : {1, 3, 4, 5, 9, 16, 31, 64, 100}) {
        mt::TernaryTensor t;
        t.rows = 3;
        t.cols = cols;
        t.gamma = {0.5};
        for (int i = 0; i < 3 * cols; ++i) t.q.push_back(static_cast<std::int8_t>(static_cast<int>(rng.next() % 3) - 1));
        for (mt::Storage st : {mt::Storage::TQ2, mt::Storage::TQ1}) {
            const std::vector<std::uint8_t> b = mt::pack(t, st);
            bad += b.size() != 3 * mt::row_bytes(cols, st);
            mt::TernaryTensor back;
            back.rows = 3;
            back.cols = cols;
            mt::unpack(b.data(), b.size(), st, back);
            bad += back.q != t.q;
        }
    }
    check(bad == 0, "TQ2 and TQ1 round trips (every row length)");
    // TQ1: all 243 five-trit strings, and the ggml decode of every byte
    bad = 0;
    for (int v = 0; v < 243; ++v) {
        mt::TernaryTensor t;
        t.rows = 1;
        t.cols = 5;
        for (int n = 0, x = v; n < 5; ++n, x /= 3) t.q.insert(t.q.begin(), static_cast<std::int8_t>(x % 3 - 1));
        const std::vector<std::uint8_t> b = mt::pack(t, mt::Storage::TQ1);
        mt::TernaryTensor back;
        back.rows = 1;
        back.cols = 5;
        mt::unpack(b.data(), 1, mt::Storage::TQ1, back);
        bad += back.q != t.q;
    }
    check(bad == 0, "TQ1: every 5-trit string round-trips");
    std::uint8_t three = 0xFF;  // 2-bit code 3
    mt::TernaryTensor tt;
    tt.rows = 1;
    tt.cols = 4;
    bool threw = false;
    try { mt::unpack(&three, 1, mt::Storage::TQ2, tt); } catch (const std::exception&) { threw = true; }
    check(threw, "TQ2 code 3 refused");
    std::printf("  ok    packing: TQ2 2.0, TQ1 1.6 bits a weight (+ row padding)\n");
}

// ---------------------------------------------------------------- 3. kernel
static mt::TernaryTensor random_trits(int rows, int cols, Lcg& rng, mt::Scale sc = mt::Scale::Tensor) {
    std::vector<double> W(static_cast<std::size_t>(rows) * cols);
    for (double& v : W) v = rng.uniform() * 0.3;
    return mt::quantize(W.data(), rows, cols, sc);
}

static std::uint64_t test_kernel() {
    std::printf("ternary x int8 GEMM\n");
    Lcg rng(7);
    std::uint64_t h = 1469598103934665603ULL;
    int bad = 0;
    for (int M : {1, 3, 4, 5, 17}) {
        for (int N : {1, 4, 7, 16, 33}) {
            for (int K : {1, 15, 16, 17, 64, 100}) {
                const mt::TernaryTensor t = random_trits(N, K, rng, (M + N) % 2 ? mt::Scale::Row : mt::Scale::Tensor);
                std::vector<double> A(static_cast<std::size_t>(M) * K);
                for (double& v : A) v = rng.uniform() * 2.0;
                mt::A8Rows q;
                mt::quantize_rows(A.data(), M, K, static_cast<std::size_t>(K), q);
                std::vector<double> C(static_cast<std::size_t>(M) * N), R(C.size());
                mt::gemm(q, mt::pack_kernel(t), C.data());
                mt::gemm_reference(q, t, R.data());
                bad += std::memcmp(C.data(), R.data(), C.size() * sizeof(double)) != 0;
                h = fnv(C.data(), C.size() * sizeof(double), fnv(q.q.data(), q.q.size(), h));
            }
        }
    }
    check(bad == 0, "packed kernel == element-by-element reference (bit for bit, 150 shapes)");
    // extreme operands: every weight +-1, every activation -128 / 127
    {
        const int M = 4, N = 16, K = 4096;
        mt::TernaryTensor t;
        t.rows = N;
        t.cols = K;
        t.gamma = {1.0};
        for (int i = 0; i < N * K; ++i) t.q.push_back(static_cast<std::int8_t>(i % 2 ? 1 : -1));
        mt::A8Rows q;
        q.rows = M;
        q.kp = K;
        q.q.assign(static_cast<std::size_t>(M) * K, 0);
        for (int i = 0; i < M * K; ++i) q.q[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(i % 2 ? 127 : -128);
        q.sum.assign(M, 0);
        for (int r = 0; r < M; ++r)
            for (int k = 0; k < K; ++k) q.sum[static_cast<std::size_t>(r)] += q.q[static_cast<std::size_t>(r) * K + k];
        q.step.assign(M, 1.0);
        std::vector<double> C(static_cast<std::size_t>(M) * N), R(C.size());
        mt::gemm(q, mt::pack_kernel(t), C.data());
        mt::gemm_reference(q, t, R.data());
        check(std::memcmp(C.data(), R.data(), C.size() * sizeof(double)) == 0 && C[0] == 255.0 * 2048,
              "saturating operands (no int16 overflow)");
    }
    std::printf("  ok    kernel == reference\n");
    return h;
}

// ---------------------------------------------------------------- 4. predict
static MlpModel make_model(const std::vector<int>& d, Lcg& rng, Activation act = Activation::Tanh) {
    MlpModel m;
    for (std::size_t l = 0; l + 1 < d.size(); ++l) {
        DenseLayer x;
        x.n_in = d[l];
        x.n_out = d[l + 1];
        x.activation = l + 2 < d.size() ? act : Activation::Identity;
        x.weight.resize(static_cast<std::size_t>(x.n_out) * x.n_in);
        x.bias.resize(static_cast<std::size_t>(x.n_out));
        for (double& w : x.weight) w = rng.uniform() * std::sqrt(3.0 / x.n_in);
        for (double& b : x.bias) b = 0.1 * rng.uniform();
        m.layers.push_back(x);
    }
    return m;
}

static std::uint64_t test_predict(std::uint64_t h) {
    std::printf("predict\n");
    Lcg rng(11);
    const MlpModel m = make_model({6, 48, 40, 3}, rng);
    const int n = 37;
    std::vector<double> X(static_cast<std::size_t>(n) * 6);
    for (double& v : X) v = rng.uniform() * 2.0;
    for (mt::Scale sc : {mt::Scale::Tensor, mt::Scale::Row}) {
        const mt::TModel t = mt::quantize(m, sc);
        std::vector<double> y;
        mt::predict(t, X.data(), n, y);
        // float64 arithmetic on the dequantised operands, layer by layer
        std::vector<double> a = X, z;
        double worst = 0.0, amax = 0.0;
        for (std::size_t l = 0; l < t.layers.size(); ++l) {
            const mt::TLayer& L = t.layers[l];
            std::vector<double> Wd(static_cast<std::size_t>(L.n_out) * L.n_in);
            mt::dequantize(L.weight, Wd.data());
            mt::A8Rows q;
            mt::quantize_rows(a.data(), n, L.n_in, static_cast<std::size_t>(L.n_in), q);
            z.assign(static_cast<std::size_t>(n) * L.n_out, 0.0);
            for (int r = 0; r < n; ++r)
                for (int o = 0; o < L.n_out; ++o) {
                    double s = 0.0;
                    for (int k = 0; k < L.n_in; ++k)
                        s += static_cast<double>(q.q[static_cast<std::size_t>(r) * q.kp + k]) *
                             q.step[static_cast<std::size_t>(r)] * Wd[static_cast<std::size_t>(o) * L.n_in + k];
                    z[static_cast<std::size_t>(r) * L.n_out + o] = s + L.bias[static_cast<std::size_t>(o)];
                }
            a.resize(z.size());
            mc::act_apply(z.data(), a.data(), z.size(), L.activation);
        }
        for (std::size_t i = 0; i < y.size(); ++i) {
            worst = std::max(worst, std::abs(y[i] - a[i]));
            amax = std::max(amax, std::abs(a[i]));
        }
        std::printf("  ok    predict vs float64 on the dequantised operands: %.2e of the absmax\n", worst / amax);
        check(worst <= 1e-12 * amax, "predict == float64 arithmetic on the dequantised operands (1e-12)");
        h = fnv(y.data(), y.size() * sizeof(double), h);
    }
    return h;
}

// ---------------------------------------------------------------- 5. training
// A float64 GEMM whose products are rounded before they are added (no FMA
// on any build), so the backward's float64 GEMMs hash alike everywhere;
// the library uses MatGemm, which may fuse.
struct NoFmaGemm {
    static double mul(double a, double b) { double p = a * b; IMPBFF_FP4_KEEP(p); return p; }
    static void nn(int M, int N, int K, const double* A, const double* B, double* C) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) s += mul(A[static_cast<std::size_t>(i) * K + k], B[static_cast<std::size_t>(k) * N + j]);
                C[static_cast<std::size_t>(i) * N + j] = s;
            }
    }
    static void nt(int M, int N, int K, const double* A, const double* B, double* C) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) s += mul(A[static_cast<std::size_t>(i) * K + k], B[static_cast<std::size_t>(j) * K + k]);
                C[static_cast<std::size_t>(i) * N + j] = s;
            }
    }
    static void tn(int M, int N, int K, const double* A, const double* B, double* C) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) s += mul(A[static_cast<std::size_t>(k) * M + i], B[static_cast<std::size_t>(k) * N + j]);
                C[static_cast<std::size_t>(i) * N + j] = s;
            }
    }
};

// The generic build's value: kernel outputs, predict, and three ternary
// training steps (all layers ternary, tanh; MlpMath's tanh, NoFmaGemm).
constexpr std::uint64_t kTernaryFingerprint = 0x4f6b5c542ab55f36ULL;

static std::uint64_t test_training(std::uint64_t h) {
    std::printf("ternary training\n");
    namespace tr = mt::train;
    Lcg rng(13);
    MlpModel m = make_model({5, 48, 40, 3}, rng);
    const int bs = 37;
    std::vector<double> X(static_cast<std::size_t>(bs) * 5), dY(static_cast<std::size_t>(bs) * 3);
    for (double& v : X) v = rng.uniform();
    for (double& v : dY) v = 0.1 * rng.uniform();
    // to_model() predicts training's forward pass bit for bit -- with the
    // portable GEMM and with the library's (MatGemm: train_neural_net's
    // TrainGemm and QuantizedNeuralNet's Gemm for the kept float64 layers)
    auto same_forward = [&](auto gemm_tag, bool keep) {
        using G = decltype(gemm_tag);
        tr::Config c;
        c.keep_first = c.keep_last = keep;
        tr::Workspace ws;
        tr::quantize_weights(m.layers, c, ws);
        tr::forward<G>(m.layers, c, X.data(), bs, ws);
        const mt::TModel q = tr::to_model(m, c);
        std::vector<double> y;
        mt::predict<G>(q, X.data(), bs, y);
        return y.size() == ws.output().size() &&
               std::memcmp(y.data(), ws.output().data(), y.size() * sizeof(double)) == 0;
    };
    for (bool keep : {true, false}) {
        check(same_forward(mc::PortableGemm(), keep), "to_model() predicts exactly training's forward pass");
        check(same_forward(MatGemm(), keep), "to_model() == training's forward pass (MatGemm)");
    }
    // the STE gradient: deterministic, and close in direction to float64's
    {
        tr::Config c;
        c.keep_first = c.keep_last = false;
        std::vector<double> g1(mc::n_parameters(m.layers)), g2(g1.size()), g64(g1.size());
        for (std::vector<double>* g : {&g1, &g2}) {
            tr::Workspace ws;
            tr::quantize_weights(m.layers, c, ws);
            tr::forward(m.layers, c, X.data(), bs, ws);
            tr::backward(m.layers, c, ws, dY.data(), bs, g->data());
        }
        check(g1 == g2, "ternary backward deterministic");
        mc::Workspace w;
        mc::forward(m.layers, X.data(), bs, w, 0);
        mc::backward(m.layers, w, dY.data(), nullptr, nullptr, g64.data());
        double dot = 0, n1 = 0, n2 = 0;
        for (std::size_t i = 0; i < g1.size(); ++i) { dot += g1[i] * g64[i]; n1 += g1[i] * g1[i]; n2 += g64[i] * g64[i]; }
        const double cosv = dot / std::sqrt(n1 * n2);
        std::printf("  ok    STE gradient cosine to float64's: %.3f\n", cosv);
        check(cosv > 0.8, "STE gradient points like the float64 one (cosine > 0.8)");
    }
    // three steps of an all-ternary tanh net into the fingerprint
    tr::Config c;
    c.keep_first = c.keep_last = false;
    tr::Workspace ws;
    std::vector<double> g(mc::n_parameters(m.layers));
    for (int step = 0; step < 3; ++step) {
        tr::quantize_weights(m.layers, c, ws);
        tr::forward<NoFmaGemm>(m.layers, c, X.data(), bs, ws);
        tr::backward<NoFmaGemm>(m.layers, c, ws, dY.data(), bs, g.data());
        h = fnv(g.data(), g.size() * sizeof(double), fnv(ws.output().data(), ws.output().size() * sizeof(double), h));
        std::size_t p = 0;  // flatten() layout: weight then bias, a layer
        for (DenseLayer& d : m.layers) {
            for (double& w : d.weight) w -= 0.125 * g[p++];
            for (double& b : d.bias) b -= 0.125 * g[p++];
        }
    }
    return h;
}

// ---------------------------------------------------------------- 6. timing
static void time_gemm() {
    Lcg rng(17);
    for (int n : {64, 256}) {
        const int M = 256;
        const mt::TernaryTensor t = random_trits(n, n, rng);
        const mt::PackedT p = mt::pack_kernel(t);
        std::vector<double> A(static_cast<std::size_t>(M) * n), C(static_cast<std::size_t>(M) * n);
        for (double& v : A) v = rng.uniform();
        mt::A8Rows q;
        mt::quantize_rows(A.data(), M, n, static_cast<std::size_t>(n), q);
        auto best = [&](auto f) {
            double b = 1e300;
            for (int rep = 0; rep < 20; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                f();
                b = std::min(b, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            return b * 1e6;
        };
        const double tg = best([&] { mt::gemm(q, p, C.data()); });
        const double tq = best([&] { mt::quantize_rows(A.data(), M, n, static_cast<std::size_t>(n), q); });
        std::vector<double> W(static_cast<std::size_t>(n) * n);
        mt::dequantize(t, W.data());
        const double tf = best([&] { mc::PortableGemm::nt(M, n, n, A.data(), W.data(), C.data()); });
        std::printf("  time  %d x %d x %d: ternary gemm %.1f us, activation quantisation %.1f us, portable double gemm %.1f us\n",
                    M, n, n, tg, tq, tf);
    }
}

int main() {
    std::printf("variant %s\n", mt::kern::kernel_name());
    test_quantisers();
    test_packing();
    std::uint64_t h = test_kernel();
    h = test_predict(h);
    h = test_training(h);
    std::printf("  ok    fingerprint %016llx\n", static_cast<unsigned long long>(h));
    check(h == kTernaryFingerprint, "fingerprint == the generic build's");
    time_gemm();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures;
}

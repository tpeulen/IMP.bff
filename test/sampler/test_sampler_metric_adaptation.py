"""PRD-147 A3: metric adaptation -- CovarianceEstimator (SamplerWarmup.h) and NutsKernel's adapt_metric. Written 2026-09-16.

1. Against Stan's own code: `stan::mcmc::covar_adaptation` and `var_adaptation` (CmdStan 2.39.0 headers, compiled by the
   generator kept at the end of this file) on a fixed 5-d stream with 1000 warm-up iterations and the default buffers
   update the metric at iterations 99, 149, 249, 449, 949 to the values in STAN; `warmup_windows` + `CovarianceEstimator`
   driven the way NutsKernel drives them must give the same iterations and the same metrics to 1e-12 relative.
2. NutsKernel with adapt_metric "dense" from the identity on a 20-d correlated Gaussian (condition 1e4): the adapted
   inverse metric's eigenvalues are within 30 % of the true covariance's (median ratio) and the adapted step size is
   more than 10x the unit-metric one; "diag" recovers the marginal variances within 30 %; the draws after warm-up have
   means and variances within 4 MCSE of the truth.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "decay"))
from bayesian_cxx import run_driver  # noqa: E402

STAN = {"dense": [{"t": 99, "m": [0.4040526155812537, -0.29071061250762426, -0.24189790233284322, 2.034220830839586, 0.03790040897628607, -0.29071061250762426, 0.7725699457320723, -0.055736557327568347, -1.0802241428091472, 0.05178173772470905, -0.24189790233284322, -0.055736557327568374, 0.26023078193538474, -1.3846542501336716, -0.05395838825432101, 2.0342208308395864, -1.0802241428091472, -1.3846542501336716, 10.641218373992235, 0.2765495348934939, 0.03790040897628607, 0.05178173772470906, -0.05395838825432101, 0.2765495348934939, 0.023523670712946712]}, {"t": 149, "m": [0.4678209614829069, -0.09075804385446934, -0.010712584353542604, 2.3914169122598135, -0.035477880929454926, -0.09075804385446934, 2.1726538580995594, -0.005103645875562404, -0.4005307947690191, -0.673382076915513, -0.010712584353542573, -0.005103645875562385, 0.27296413440870887, -0.1543991310745589, 0.022752303939588953, 2.3914169122598143, -0.4005307947690191, -0.15439913107455902, 12.830717908280441, -0.1324875445556343, -0.035477880929454954, -0.6733820769155132, 0.02275230393958898, -0.13248754455563444, 0.24321255058491506]}, {"t": 249, "m": [0.47552151785368363, -0.1587933505431509, 0.07310403955008685, 2.2482249836772707, -0.014235563806109238, -0.15879335054315089, 1.8628633933169378, 0.10500669024401779, -2.740742492000574, 0.1413410235278217, 0.07310403955008685, 0.10500669024401775, 0.251722422714325, -0.02386645623273165, 0.032104494453909456, 2.2482249836772707, -2.7407424920005745, -0.023866456232731707, 15.46459059447305, -0.2253136635552816, -0.014235563806109245, 0.14134102352782174, 0.032104494453909456, -0.2253136635552815, 0.1470120314532393]}, {"t": 449, "m": [0.5019526227693334, -0.11632893840144701, -0.01879221876398288, 2.4843080074946053, -0.009925214354883983, -0.11632893840144695, 2.0459281342775735, 0.13029975746835443, -1.9859666918745171, -0.045412606733101685, -0.018792218763982876, 0.13029975746835443, 0.2871625790537837, -0.5396741742674437, -0.09571403543366257, 2.4843080074946053, -1.9859666918745165, -0.5396741742674437, 17.200656873238998, 0.02384182371704452, -0.00992521435488398, -0.045412606733101664, -0.09571403543366257, 0.023841823717044523, 0.21506165162097376]}, {"t": 949, "m": [0.5008061001147214, -0.15638144284826047, -0.021383936141076046, 2.6338096390208983, -0.01212031768178904, -0.15638144284826047, 2.1431742374599545, -0.0031310804161338744, -1.3194077429315758, -0.08701758023314704, -0.021383936141076056, -0.0031310804161338783, 0.29149612714331985, -0.528822873068938, -0.02348370535370762, 2.633809639020898, -1.3194077429315751, -0.528822873068938, 18.40239635296236, -0.03196595319339444, -0.012120317681789037, -0.08701758023314704, -0.023483705353707626, -0.031965953193394375, 0.2536039500877456]}], "diag": [{"t": 99, "v": [0.4040526155812537, 0.7725699457320723, 0.26023078193538474, 10.641218373992235, 0.023523670712946712]}, {"t": 149, "v": [0.4678209614829069, 2.1726538580995594, 0.27296413440870887, 12.830717908280441, 0.24321255058491506]}, {"t": 249, "v": [0.47552151785368363, 1.8628633933169378, 0.251722422714325, 15.46459059447305, 0.1470120314532393]}, {"t": 449, "v": [0.5019526227693334, 2.0459281342775735, 0.2871625790537837, 17.200656873238998, 0.21506165162097376]}, {"t": 949, "v": [0.5008061001147214, 2.1431742374599545, 0.29149612714331985, 18.40239635296236, 0.2536039500877456]}]}

ESTIMATOR = r"""
#include <IMP/bff/SamplerWarmup.h>
#include <cmath>
#include <cstdio>
using namespace IMP::bff;
static std::vector<double> q_of(int t) {
  const double a = std::sin(0.37 * t), b = std::cos(0.11 * t + 0.4), c = std::sin(0.023 * t * t * 0.01 + 1.3);
  return {a + 0.1 * b, 2.0 * b - 0.5 * a, 0.3 * c + a * b, 5.0 * a - 3.0 * c, 0.01 * t * 0.001 + b * c};
}
int main() {
  const int n = 1000;
  const WarmupWindows w = warmup_windows(n);
  for (int mode = 0; mode < 2; ++mode) {
    CovarianceEstimator est(5, mode == 0);
    std::printf("%s", mode == 0 ? "{\"dense\": [" : "], \"diag\": [");
    bool first = true;
    for (int i = 0; i < n; ++i) {
      if (i >= w.init_buffer && i < n - w.term_buffer) est.add(q_of(i));
      if (!w.closes(i + 1)) continue;
      const std::vector<double> m = est.regularised();
      std::printf("%s{\"t\": %d, \"m\": [", first ? "" : ",", i); first = false;
      for (std::size_t k = 0; k < m.size(); ++k) std::printf("%s%.17g", k ? "," : "", m[k]);
      std::printf("]}");
      est.restart();
    }
  }
  std::printf("]}\n");
  return 0;
}
"""

NUTS = r"""
#include <IMP/bff/NutsKernel.h>
#include <IMP/bff/SamplerDiagnostics.h>
#include <cmath>
#include <cstdio>
#include <random>
using namespace IMP::bff;
using Vec = std::vector<double>;
int main() {
  const std::size_t n = 20;
  std::mt19937_64 rng(5); std::normal_distribution<double> N(0.0, 1.0);
  Vec Q(n * n); for (double& v : Q) v = N(rng);
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t p = 0; p < c; ++p) { double d = 0; for (std::size_t r = 0; r < n; ++r) d += Q[r*n+c]*Q[r*n+p]; for (std::size_t r = 0; r < n; ++r) Q[r*n+c] -= d*Q[r*n+p]; }
    double nn = 0; for (std::size_t r = 0; r < n; ++r) nn += Q[r*n+c]*Q[r*n+c]; nn = std::sqrt(nn); for (std::size_t r = 0; r < n; ++r) Q[r*n+c] /= nn;
  }
  Vec lam(n), C(n * n, 0.0), P(n * n, 0.0), mu(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) lam[k] = std::pow(10.0, -4.0 * double(k) / double(n - 1));
  for (std::size_t i = 0; i < n; ++i) { mu[i] = 0.3 * N(rng); for (std::size_t j = 0; j < n; ++j) for (std::size_t k = 0; k < n; ++k) {
    C[i*n+j] += Q[i*n+k]*lam[k]*Q[j*n+k]; P[i*n+j] += Q[i*n+k]/lam[k]*Q[j*n+k]; } }
  const SamplingTarget g = sampling_target(n, std::function<double(const Vec&, Vec&)>([&](const Vec& x, Vec& gr) {
    gr.assign(n, 0.0); double lp = 0;
    for (std::size_t i = 0; i < n; ++i) { double s = 0; for (std::size_t j = 0; j < n; ++j) s += P[i*n+j]*(x[j]-mu[j]); gr[i] = -s; lp -= 0.5*(x[i]-mu[i])*s; }
    return lp; }));
  std::mt19937_64 sr(1); std::uniform_real_distribution<double> U(-2.0, 2.0);
  std::vector<std::vector<Vec>> starts(4, std::vector<Vec>(1, Vec(n)));
  for (auto& s : starts) for (double& v : s[0]) v = U(sr);
  SamplerOptions o; o.warmup = 1000; o.draws = 1000; o.seed = 3;
  auto run = [&](const std::string& adapt) { NutsOptions no; no.adapt_metric = adapt; return run_sampler(g, NutsKernel(no), starts, o); };
  const SampleResult unit = run("none"), dense = run("dense"), diag = run("diag");
  auto step_of = [](const SampleResult& r) { std::size_t k = 0; while (r.stat_names[k] != "step_size") ++k; double s = 0; for (auto& row : r.stats) s += row.back()[k]; return s / r.stats.size(); };
  // eigenvalues of the symmetric matrix by Jacobi
  auto eig = [n](Vec a) {
    for (int sweep = 0; sweep < 200; ++sweep) {
      double off = 0; for (std::size_t p = 0; p < n; ++p) for (std::size_t q = p + 1; q < n; ++q) off += a[p*n+q]*a[p*n+q];
      if (off < 1e-30) break;
      for (std::size_t p = 0; p < n; ++p) for (std::size_t q = p + 1; q < n; ++q) {
        if (std::fabs(a[p*n+q]) < 1e-300) continue;
        const double th = (a[q*n+q]-a[p*n+p]) / (2*a[p*n+q]); const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th*th+1));
        const double c = 1/std::sqrt(t*t+1), s = t*c;
        for (std::size_t k = 0; k < n; ++k) { const double akp = a[k*n+p], akq = a[k*n+q]; a[k*n+p] = c*akp - s*akq; a[k*n+q] = s*akp + c*akq; }
        for (std::size_t k = 0; k < n; ++k) { const double apk = a[p*n+k], aqk = a[q*n+k]; a[p*n+k] = c*apk - s*aqk; a[q*n+k] = s*apk + c*aqk; }
      }
    }
    Vec e(n); for (std::size_t i = 0; i < n; ++i) e[i] = a[i*n+i]; std::sort(e.begin(), e.end()); return e; };
  auto parse_metric = [n](const std::string& js) {
    Vec m; const auto p = js.find('[', js.find("inverse_metric")); std::size_t i = p + 1;
    while (i < js.size() && js[i] != ']') { char* end; m.push_back(std::strtod(js.c_str() + i, &end)); i = end - js.c_str(); if (js[i] == ',') ++i; }
    return m; };
  const Vec etrue = eig(C);
  double worst_dense_eig = 0, worst_diag_var = 0;
  std::vector<double> ratios;
  for (const auto& tj : dense.tuning) { const Vec e = eig(parse_metric(tj)); for (std::size_t i = 0; i < n; ++i) ratios.push_back(e[i] / etrue[i]); }
  std::sort(ratios.begin(), ratios.end()); const double median_eig_ratio = ratios[ratios.size() / 2];
  for (double r : ratios) worst_dense_eig = std::max(worst_dense_eig, std::fabs(std::log(r)));
  for (const auto& tj : diag.tuning) { const Vec m = parse_metric(tj); for (std::size_t i = 0; i < n; ++i) worst_diag_var = std::max(worst_diag_var, std::fabs(m[i*n+i] / C[i*n+i] - 1.0)); }
  double worst_z = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto ch = dense.independent_chains([i](const Vec& x) { return x[i]; });
    double m = 0, N2 = 0; for (auto& c : ch) for (double v : c) { m += v; N2 += 1; } m /= N2;
    worst_z = std::max(worst_z, std::fabs(m - mu[i]) / mcse_mean(ch));
    auto sq = ch; for (auto& c : sq) for (double& v : c) v = (v - mu[i]) * (v - mu[i]);
    double vb = 0; for (auto& c : sq) for (double v : c) vb += v; vb /= N2;
    worst_z = std::max(worst_z, std::fabs(vb - C[i*n+i]) / mcse_mean(sq));
  }
  std::printf("{\"step_unit\": %.6g, \"step_dense\": %.6g, \"step_diag\": %.6g, \"median_eig_ratio\": %.4f, \"worst_log_eig_ratio\": %.4f, \"worst_diag_var\": %.4f, \"worst_z\": %.3f}\n",
              step_of(unit), step_of(dense), step_of(diag), median_eig_ratio, worst_dense_eig, worst_diag_var, worst_z);
  return 0;
}
"""


def test_estimator_and_windows_match_stan():
    out = run_driver(ESTIMATOR)
    for mode, key in (("dense", "m"), ("diag", "v")):
        ours, theirs = out[mode], STAN[mode]
        assert [u["t"] for u in ours] == [u["t"] for u in theirs], mode
        for u, v in zip(ours, theirs):
            for a, b in zip(u["m"], v[key]):
                assert abs(a - b) <= 1e-12 * max(abs(b), 1e-300), (mode, u["t"], a, b)


def test_nuts_dense_and_diag_adaptation_recover_the_geometry():
    r = run_driver(NUTS, timeout=1800)
    assert 0.7 < r["median_eig_ratio"] < 1.3, r
    assert r["step_dense"] > 10 * r["step_unit"], r
    assert r["worst_diag_var"] < 0.3, r
    assert r["worst_z"] < 4.0, r


GENERATOR = r"""#include <stan/mcmc/covar_adaptation.hpp>
#include <stan/mcmc/var_adaptation.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <cmath>
#include <cstdio>
#include <sstream>
// Stan's covar_adaptation and var_adaptation on a fixed 5-d stream q_t (t = 0..999), num_warmup 1000 with the default
// buffers (75/50/25); prints every update (the iteration it happened at and the metric).
static Eigen::VectorXd q_of(int t) {
  Eigen::VectorXd q(5);
  const double a = std::sin(0.37 * t), b = std::cos(0.11 * t + 0.4), c = std::sin(0.023 * t * t * 0.01 + 1.3);
  q << a + 0.1 * b, 2.0 * b - 0.5 * a, 0.3 * c + a * b, 5.0 * a - 3.0 * c, 0.01 * t * 0.001 + b * c;
  return q;
}
int main() {
  std::stringstream s1, s2, s3, s4, s5;
  stan::callbacks::stream_logger logger(s1, s2, s3, s4, s5);
  stan::mcmc::covar_adaptation ca(5); ca.set_window_params(1000, 75, 50, 25, logger);
  stan::mcmc::var_adaptation va(5); va.set_window_params(1000, 75, 50, 25, logger);
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(5, 5); Eigen::VectorXd v = Eigen::VectorXd::Ones(5);
  std::printf("{\"dense\": [");
  bool first = true;
  for (int t = 0; t < 1000; ++t) if (ca.learn_covariance(M, q_of(t))) {
    std::printf("%s{\"t\": %d, \"m\": [", first ? "" : ",", t); first = false;
    for (int i = 0; i < 25; ++i) std::printf("%s%.17g", i ? "," : "", M(i / 5, i % 5));
    std::printf("]}");
  }
  std::printf("], \"diag\": [");
  first = true;
  for (int t = 0; t < 1000; ++t) if (va.learn_variance(v, q_of(t))) {
    std::printf("%s{\"t\": %d, \"v\": [", first ? "" : ",", t); first = false;
    for (int i = 0; i < 5; ++i) std::printf("%s%.17g", i ? "," : "", v(i));
    std::printf("]}");
  }
  std::printf("]}\n");
}
"""

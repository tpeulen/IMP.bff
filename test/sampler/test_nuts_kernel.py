"""PRD-147 step 2: the NUTS kernel behind the sampler interface (Sampling.h, NutsKernel.h), and its
registry entry. Written 2026-09-15.

The kernel is the NUTS benchmarked against CmdStan 2.39.0 (PRD-146 B1/B2), moved from tttrlib. Checks,
each of which can fail:

- a 20-d correlated Gaussian (condition 1e4), 4 chains through run_sampler: every mean and variance
  within 4 MCSE of the truth, rank R-hat < 1.01, bulk ESS > 400 (SamplerDiagnostics.h);
- the true covariance as inverse metric: tree depth drops by > 1 and ESS per gradient rises > 5x;
- max_depth 1 (the control): a far smaller ESS per draw;
- Neal's funnel at a fixed, too large step: divergences reported;
- **against Stan**: 20 independent normals (sds 0.1..10), unit metric, fixed step 0.15, 4 x 5000 draws:
  the tree-depth histogram matches CmdStan 2.39.0's (ucfret s89_cpp/nuts_vs_stan/emit_depth_reference.py)
  by chi-square at p > 1e-3, and the mean acceptance within 0.01. This is the check that caught the
  wrong end momentum in the cross-subtree U-turn test; the moments alone did not;
- a target without a gradient is refused, and independent_chains() refuses ensemble rows;
- linked against the library: create_sampler_kernel("nuts", options) builds the registered kernel,
  and refuses an undeclared option, an out-of-range value and an unknown name, naming the option.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "test", "decay"))
from bayesian_cxx import run_driver  # noqa: E402

COMMON = r"""
#include <IMP/bff/NutsKernel.h>
#include <IMP/bff/SamplerDiagnostics.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace IMP::bff;
using Vec = std::vector<double>;
"""

HEADER_ONLY = COMMON + r"""
static SamplingTarget gaussian20(Vec& C, Vec& mu) {
  const std::size_t n = 20;
  std::mt19937_64 rng(5); std::normal_distribution<double> N(0.0, 1.0);
  Vec Q(n * n); for (double& v : Q) v = N(rng);
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t p = 0; p < c; ++p) { double d = 0; for (std::size_t r = 0; r < n; ++r) d += Q[r*n+c]*Q[r*n+p]; for (std::size_t r = 0; r < n; ++r) Q[r*n+c] -= d*Q[r*n+p]; }
    double nn = 0; for (std::size_t r = 0; r < n; ++r) nn += Q[r*n+c]*Q[r*n+c]; nn = std::sqrt(nn); for (std::size_t r = 0; r < n; ++r) Q[r*n+c] /= nn;
  }
  Vec lam(n), P(n * n, 0.0); C.assign(n * n, 0.0); mu.assign(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) lam[k] = std::pow(10.0, -4.0 * double(k) / double(n - 1));
  for (std::size_t i = 0; i < n; ++i) { mu[i] = 0.3 * N(rng); for (std::size_t j = 0; j < n; ++j) for (std::size_t k = 0; k < n; ++k) {
    C[i*n+j] += Q[i*n+k]*lam[k]*Q[j*n+k]; P[i*n+j] += Q[i*n+k]/lam[k]*Q[j*n+k]; } }
  return sampling_target(n, std::function<double(const Vec&, Vec&)>([P, mu, n](const Vec& x, Vec& g) {
    g.assign(n, 0.0); double lp = 0;
    for (std::size_t i = 0; i < n; ++i) { double s = 0; for (std::size_t j = 0; j < n; ++j) s += P[i*n+j]*(x[j]-mu[j]); g[i] = -s; lp -= 0.5*(x[i]-mu[i])*s; }
    return lp; }));
}
static std::vector<std::vector<Vec>> starts(std::size_t n, int chains, double spread, std::uint64_t seed) {
  std::mt19937_64 rng(seed); std::uniform_real_distribution<double> U(-spread, spread);
  std::vector<std::vector<Vec>> s(chains, std::vector<Vec>(1, Vec(n)));
  for (auto& c : s) for (double& v : c[0]) v = U(rng);
  return s;
}
static double stat_mean(const SampleResult& r, const std::string& name) {
  std::size_t k = 0; while (r.stat_names[k] != name) ++k;
  double s = 0; std::size_t m = 0; for (const auto& row : r.stats) for (const auto& d : row) { s += d[k]; ++m; }
  return s / double(m);
}
static double stat_sum(const SampleResult& r, const std::string& name) { return stat_mean(r, name) * double(r.stats.size() * r.stats[0].size()); }
static double min_ess(const SampleResult& r, std::size_t n) {
  double e = 1e300; for (std::size_t i = 0; i < n; ++i) e = std::min(e, ess_bulk(r.independent_chains([i](const Vec& x) { return x[i]; })));
  return e;
}
int main() {
  const std::size_t n = 20;
  Vec C, mu; const SamplingTarget g = gaussian20(C, mu);
  SamplerOptions o; o.warmup = 1000; o.draws = 1000; o.seed = 11;
  const SampleResult id = run_sampler(g, NutsKernel(), starts(n, 4, 0.5, 1), o);
  double worst_mean = 0, worst_var = 0, worst_rhat = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto c = id.independent_chains([i](const Vec& x) { return x[i]; });
    double m = 0; for (auto& ch : c) for (double v : ch) m += v; m /= 4000.0;
    worst_mean = std::max(worst_mean, std::fabs(m - mu[i]) / mcse_mean(c));
    auto sq = c; for (auto& ch : sq) for (double& v : ch) v = (v - mu[i]) * (v - mu[i]);
    double vb = 0; for (auto& ch : sq) for (double v : ch) vb += v; vb /= 4000.0;
    worst_var = std::max(worst_var, std::fabs(vb - C[i*n+i]) / mcse_mean(sq));
    worst_rhat = std::max(worst_rhat, rhat_rank(c));
  }
  const double ess_id = min_ess(id, n);
  NutsOptions dense_opt; dense_opt.inverse_metric = C;
  SamplerOptions od = o; od.warmup = 300; od.seed = 12;
  const SampleResult dense = run_sampler(g, NutsKernel(dense_opt), starts(n, 4, 0.5, 2), od);
  NutsOptions shallow_opt; shallow_opt.max_depth = 1;
  SamplerOptions os = o; os.seed = 13;
  const SampleResult shallow = run_sampler(g, NutsKernel(shallow_opt), starts(n, 4, 0.5, 3), os);
  // Neal's funnel, fixed step 0.8
  const std::size_t nf = 10;
  const SamplingTarget funnel = sampling_target(nf, std::function<double(const Vec&, Vec&)>([nf](const Vec& x, Vec& gr) {
    gr.assign(nf, 0.0); const double v = x[0]; double lp = -v * v / 18.0; gr[0] = -v / 9.0;
    for (std::size_t k = 1; k < nf; ++k) { lp += -0.5*x[k]*x[k]*std::exp(-v) - 0.5*v; gr[0] += 0.5*x[k]*x[k]*std::exp(-v) - 0.5; gr[k] = -x[k]*std::exp(-v); }
    return lp; }));
  NutsOptions fixed; fixed.step_size = 0.8; fixed.find_step_size = false;
  SamplerOptions of; of.warmup = 0; of.draws = 500; of.seed = 14;
  const SampleResult fun = run_sampler(funnel, NutsKernel(fixed), starts(nf, 2, 0.5, 4), of);
  // against Stan: 20 independent normals, fixed step 0.15, 200 warm-up + 5000 draws per chain
  const std::size_t ns = 20;
  const SamplingTarget scales = sampling_target(ns, std::function<double(const Vec&, Vec&)>([ns](const Vec& x, Vec& gr) {
    gr.assign(ns, 0.0); double lp = 0;
    for (std::size_t k = 0; k < ns; ++k) { const double sd = std::pow(10.0, -1.0 + 2.0 * double(k) / 19.0); lp -= 0.5*x[k]*x[k]/(sd*sd); gr[k] = -x[k]/(sd*sd); }
    return lp; }));
  NutsOptions step015; step015.step_size = 0.15; step015.find_step_size = false;
  SamplerOptions ost; ost.warmup = 200; ost.draws = 5000; ost.seed = 1717;
  // warm-up with no adaptation: run_sampler adapts during warm-up, so burn in by recording and discarding
  SamplerOptions burn = ost; burn.warmup = 0; burn.draws = 5200;
  const SampleResult st = run_sampler(scales, NutsKernel(step015), starts(ns, 4, 2.0, 17), burn);
  double counts[5] = {0, 0, 0, 0, 0}, acc = 0; std::size_t kd = 2, ka = 0;
  for (const auto& row : st.stats) for (std::size_t d = 200; d < row.size(); ++d) { counts[std::min(4, std::max(0, int(row[d][kd]) - 4))] += 1; acc += row[d][ka]; }
  acc /= 20000.0;
  const double stan_counts[5] = {2 + 4 + 141, 1410, 7154, 10263, 1026};
  double chi2 = 0; for (int b = 0; b < 5; ++b) { const double e = 0.5 * (counts[b] + stan_counts[b]); chi2 += (counts[b]-e)*(counts[b]-e)/e + (stan_counts[b]-e)*(stan_counts[b]-e)/e; }
  // refusals
  int refused_gradient = 0, refused_ensemble_rows = 0;
  try { run_sampler(sampling_target(3, std::function<double(const Vec&)>([](const Vec&) { return 0.0; })), NutsKernel(), starts(3, 1, 1.0, 5), of); }
  catch (const SamplerConfigurationError&) { refused_gradient = 1; }
  SampleResult fake = id; fake.independent_group.assign(fake.independent_group.size(), 0);
  try { fake.independent_chains([](const Vec& x) { return x[0]; }); } catch (const SamplerConfigurationError&) { refused_ensemble_rows = 1; }
  std::printf("{\"worst_mean\": %.6g, \"worst_var\": %.6g, \"worst_rhat\": %.6g, \"min_ess\": %.6g,"
              " \"depth_id\": %.6g, \"depth_dense\": %.6g, \"epg_id\": %.6g, \"epg_dense\": %.6g, \"ess_shallow\": %.6g,"
              " \"funnel_divergences\": %.6g, \"stan_chi2\": %.6g, \"stan_accept_diff\": %.6g,"
              " \"refused_gradient\": %d, \"refused_ensemble_rows\": %d, \"evaluations_id\": %ld, \"leapfrog_id\": %.6g}\n",
              worst_mean, worst_var, worst_rhat, ess_id, stat_mean(id, "tree_depth"), stat_mean(dense, "tree_depth"),
              ess_id / stat_sum(id, "n_leapfrog"), min_ess(dense, n) / stat_sum(dense, "n_leapfrog"), min_ess(shallow, n),
              stat_sum(fun, "divergent"), chi2, acc - 0.76997065, refused_gradient, refused_ensemble_rows, id.evaluations,
              stat_sum(id, "n_leapfrog"));
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(HEADER_ONLY, timeout=900)


def test_gaussian_moments_and_convergence(result):
    assert result["worst_mean"] < 4.0
    assert result["worst_var"] < 4.0
    assert result["worst_rhat"] < 1.01
    assert result["min_ess"] > 400


def test_dense_metric_is_used(result):
    assert result["depth_dense"] < result["depth_id"] - 1.0
    assert result["epg_dense"] > 5.0 * result["epg_id"]


def test_max_depth_one_control(result):
    assert result["ess_shallow"] < 0.2 * result["min_ess"]


def test_funnel_divergences_reported(result):
    assert result["funnel_divergences"] > 0


def test_tree_depth_histogram_matches_stan(result):
    # chi-square, 4 dof, p > 1e-3
    assert result["stan_chi2"] < 18.467, result["stan_chi2"]
    assert abs(result["stan_accept_diff"]) < 0.01


def test_refusals(result):
    assert result["refused_gradient"] == 1
    assert result["refused_ensemble_rows"] == 1


def test_evaluations_are_counted(result):
    # every leapfrog step is one evaluation; the count covers warm-up (1000 transitions, at first with
    # longer trajectories) as well as the 1000 recorded draws whose leapfrog steps are summed here
    assert result["leapfrog_id"] < result["evaluations_id"] < 4.0 * result["leapfrog_id"]


LINKED = COMMON + r"""
#include <IMP/bff/Registry.h>
#include <string>
int main() {
  auto k = create_sampler_kernel("nuts", "{\"max_depth\": 7, \"target_accept\": 0.9}");
  const auto* nk = dynamic_cast<NutsKernel*>(k.get());
  std::string e1, e2, e3, e4;
  try { create_sampler_kernel("nuts", "{\"max_dept\": 7}"); } catch (const SamplerConfigurationError& e) { e1 = e.what(); }
  try { create_sampler_kernel("nuts", "{\"target_accept\": 1.5}"); } catch (const SamplerConfigurationError& e) { e2 = e.what(); }
  try { create_sampler_kernel("nutz"); } catch (const SamplerConfigurationError& e) { e3 = e.what(); }
  try { create_sampler_kernel("nuts", "{\"max_depth\": \"deep\"}"); } catch (const SamplerConfigurationError& e) { e4 = e.what(); }
  auto esc = [](std::string s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o; };
  std::printf("{\"name\": \"%s\", \"max_depth\": %d, \"target_accept\": %g, \"e1\": \"%s\", \"e2\": \"%s\", \"e3\": \"%s\", \"e4\": \"%s\"}\n",
              k->name().c_str(), nk ? nk->options().max_depth : -1, nk ? nk->options().target_accept : -1.0,
              esc(e1).c_str(), esc(e2).c_str(), esc(e3).c_str(), esc(e4).c_str());
  return 0;
}
"""


def _library_dir():
    for c in (os.environ.get("BFF_LIBRARY_DIR"), os.path.join(ROOT, "build", "claude")) + tuple(glob.glob(os.path.join(ROOT, "build", "cp3*"))):
        if c and glob.glob(os.path.join(c, "libimp_bff.*")):
            return c
    return None


def test_registered_kernel_and_option_checking():
    lib = _library_dir()
    if not lib:
        pytest.skip("no built libimp_bff (set BFF_LIBRARY_DIR)")
    cxx = os.environ.get("CXX") or shutil.which("c++")
    tmp = tempfile.mkdtemp()
    try:
        shim = os.path.join(tmp, "inc", "IMP")
        os.makedirs(shim)
        os.symlink(os.path.join(ROOT, "include"), os.path.join(shim, "bff"))
        src, exe = os.path.join(tmp, "d.cpp"), os.path.join(tmp, "d")
        open(src, "w").write(LINKED)
        r = subprocess.run([cxx, "-std=c++17", "-O1", "-DIMPBFF_STANDALONE", "-I", os.path.join(tmp, "inc"), "-I",
                            os.path.join(ROOT, "standalone", "include"), src, "-o", exe, "-L", lib, "-limp_bff",
                            "-Wl,-rpath," + lib], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-2000:]
        import json
        out = json.loads(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert out["name"] == "nuts" and out["max_depth"] == 7 and abs(out["target_accept"] - 0.9) < 1e-12
    assert "no option 'max_dept'" in out["e1"] and "max_depth" in out["e1"]
    assert "target_accept" in out["e2"] and "maximum" in out["e2"] and "dual-averaging" in out["e2"]
    assert "unknown sampler 'nutz'" in out["e3"] and "nuts" in out["e3"]
    assert "expected integer" in out["e4"]

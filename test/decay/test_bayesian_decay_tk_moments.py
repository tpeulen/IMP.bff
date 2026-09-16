"""PRD-149 step 3: Tierney-Kadane moments of mean R/R0, against a sampled posterior. Written 2026-09-16.

The delta method linearises a summary at the mode and pushes the covariance through it, so its error is
`O(n^-1)`; Tierney & Kadane (*J. Amer. Statist. Assoc.* 81:82, 1986) write the posterior mean as a ratio of
two Laplace integrals -- the numerator tilted by `log f` -- whose leading errors cancel, leaving `O(n^-2)`.
ucfret's prototype (`s88_laplace_posterior.py`, `tk_moments`) applies it to the nodes of the lambda grid that
carry weight, with one rule this ports exactly: the tilted integral's curvature is the mode's own matrix plus
the exact Hessian of the tilt at the tilted mode, because two independent scoring runs do not agree to the
1e-4 nats the ratio of their log-determinants needs (measured there: 0.026 nats moved a mean by three sds).

**The reference is a sampled posterior, not another approximation.** Data are drawn from the model at the
fixture's own parameters, one node is fitted, and NUTS is run from dispersed starts with the Laplace
covariance as metric; mean R/R0 is formed per draw, and its own rank R-hat and bulk ESS are checked before
anything is compared to it (Rule 0c: convergence is checked on the quantity being reported, not on the
coordinates). Then the delta method and TK are scored against it. A comparison against another Laplace
approximation would not have been able to fail.

What is checked, and what would make each check fail:
* the tilt's analytic gradient and Hessian equal central differences of the tilt itself -- any sign or
  chain-rule error in `bayesian_decay_mean_rel_tilt` fails it, and a Hessian perturbed by 1e-3 of its own
  scale is run as the control that must fail;
* on a Gaussian posterior with a linear summary, where the exact mean and sd are known in closed form, the
  library's `bayesian_tierney_kadane` returns them from the two tilted ratios;
* the sampled reference converges (R-hat < 1.01, bulk ESS > 400), and against it TK's sd is within 5 % while
  the delta method's is not, and TK's mean is within half a posterior sd;
* TK is never run below `tk_min_weight`, and `summary_method = "delta"` (the default) leaves the grid as it
  was.
"""
import math
import tempfile

import numpy as np
import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecaySampling.h>
#include <IMP/bff/NutsKernel.h>
#include <cstdio>
#include <random>
using namespace IMP::bff;

static void num(double v) { if (std::isfinite(v)) std::printf("%.17g", v); else std::printf("null"); }
static void vec(const char* name, const std::vector<double>& v) {
  std::printf(", \"%s\": [", name);
  for (std::size_t i = 0; i < v.size(); ++i) { if (i) std::printf(","); num(v[i]); }
  std::printf("]");
}

int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::size_t dim = ex.dim;
  std::vector<double> truth(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(truth.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const double eps = 1e-6;

  // 1. the tilt's derivatives against central differences of the tilt itself, on the spline block
  {
    const double power = 2.0;
    const BayesianDecayTilt T0 = bayesian_decay_mean_rel_tilt(ex, E, truth, power, eps);
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(ex, "c");
    const std::size_t nz = ex["Q_c"].shape[1];
    std::vector<double> g(nz), gfd(nz), H(nz * nz), hfd(nz * nz);
    const double h = 1e-5;
    for (std::size_t b = 0; b < nz; ++b) {
      std::vector<double> tp = truth, tm = truth;
      tp[vc.off + b] += h; tm[vc.off + b] -= h;
      const BayesianDecayTilt Tp = bayesian_decay_mean_rel_tilt(ex, E, tp, power, eps);
      const BayesianDecayTilt Tm = bayesian_decay_mean_rel_tilt(ex, E, tm, power, eps);
      gfd[b] = (Tp.value - Tm.value) / (2.0 * h);
      g[b] = T0.grad[vc.off + b];
      for (std::size_t a = 0; a < nz; ++a) {
        hfd[a * nz + b] = (Tp.grad[vc.off + a] - Tm.grad[vc.off + a]) / (2.0 * h);
        H[a * nz + b] = T0.hess[(vc.off + a) * dim + vc.off + b];
      }
    }
    double off_block = 0.0;
    for (std::size_t i = 0; i < dim; ++i) if (i < vc.off || i >= vc.off + nz) off_block = std::max(off_block, std::fabs(T0.grad[i]));
    std::printf("{\"nz\": %zu, \"grad_off_block\": %.17g", nz, off_block);
    vec("grad", g); vec("grad_fd", gfd); vec("hess", H); vec("hess_fd", hfd);
  }

  // 2. data drawn from the model at the fixture's own parameters, and one fitted node
  {
    const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, truth));
    std::mt19937_64 rng(3);
    for (std::size_t i = 0; i < lam.size(); ++i) ex.arrays["y"].d[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
  }
  const BayesianDecayFit F = bayesian_decay_fit_node(ex, E, truth, 1.0, 400);
  const std::vector<double> Sig = bayesian_decay_covariance(F.pt, dim);
  std::vector<double> p, p_sd;
  double delta_mean = 0.0, delta_sd = 0.0;
  bayesian_decay_distribution_with_sd(ex, E, F.theta, Sig, p, p_sd, &delta_mean, &delta_sd);
  const BayesianDecayTkSummary tk = bayesian_decay_tk_mean_rel(ex, E, F, eps, 100, false);
  const BayesianDecayTkSummary tkx = bayesian_decay_tk_mean_rel(ex, E, F, eps, 100, true);
  std::printf(", \"node\": {\"delta_mean\": %.17g, \"delta_sd\": %.17g, \"tk_mean\": %.17g, \"tk_sd\": %.17g, \"tk_ok\": %d,"
              " \"tkx_mean\": %.17g, \"tkx_sd\": %.17g, \"tkx_ok\": %d, \"converged\": %d, \"shift1\": %.17g, \"shift2\": %.17g,"
              " \"dlogdet1\": %.17g, \"dlogdet2\": %.17g}",
              delta_mean, delta_sd, tk.mean, tk.sd, tk.ok ? 1 : 0, tkx.mean, tkx.sd, tkx.ok ? 1 : 0, F.converged ? 1 : 0,
              tk.shift[0], tk.shift[1], tk.dlogdet[0], tk.dlogdet[1]);

  // 3. the reference: mean R/R0 sampled, with its own convergence
  {
    const SamplingTarget target = bayesian_decay_sampling_target(ex, E);
    NutsOptions nopt; nopt.inverse_metric = Sig;
    std::normal_distribution<double> N(0.0, 1.0);
    std::mt19937_64 r2(11);
    std::vector<std::vector<std::vector<double>>> starts;
    for (int c = 0; c < 4; ++c) {
      std::vector<double> s = F.theta;
      for (std::size_t i = 0; i < dim; ++i) s[i] += 0.5 * std::sqrt(Sig[i * dim + i]) * N(r2);
      starts.push_back({s});
    }
    SamplerOptions so; so.warmup = 1000; so.draws = 3000; so.seed = 77;
    const SampleResult r = run_sampler(target, NutsKernel(nopt), starts, so);
    const std::vector<std::vector<double>> ch = r.independent_chains([&](const std::vector<double>& x) {
      const std::vector<double> q = bayesian_decay_distribution(E, bayesian_decay_unpack(ex, x));
      double m = 0.0;
      for (std::size_t j = 0; j < q.size(); ++j) m += q[j] * ex["rel"].d[j];
      return m;
    });
    const ChainSummary cs = summarize_chains(ch);
    double m = 0.0, v = 0.0; std::size_t n = 0;
    for (const auto& c : ch) for (double x : c) { m += x; ++n; }
    m /= double(n);
    for (const auto& c : ch) for (double x : c) v += (x - m) * (x - m);
    v /= double(n - 1);
    std::printf(", \"mcmc\": {\"mean\": %.17g, \"sd\": %.17g, \"mcse\": %.17g, \"rhat\": %.6f, \"ess\": %.1f, \"n\": %zu}",
                m, std::sqrt(v), cs.mcse, cs.rhat, cs.ess_bulk, n);
  }

  // 4. the grid: TK only where a node carries weight, and the default untouched
  {
    const std::vector<double> nodes = {0.0, 1.0, 2.0};
    BayesianDecayLambdaOptions od; od.refine_step = 0.0; od.sweeps = 0; od.max_iter = 300;
    const BayesianDecayLambdaGrid Gd = bayesian_decay_fit_lambda_grid(ex, E, nodes, F.theta, 1.0, od);
    //: the floor is set between the grid's own largest and smallest weight, so it must exclude some nodes
    //: and keep others whatever the simulated data happen to give
    double wmin = 1.0, wmax = 0.0;
    for (double w : Gd.weights) { wmin = std::min(wmin, w); wmax = std::max(wmax, w); }
    const double floor_ = 0.5 * (wmin + wmax);
    BayesianDecayLambdaOptions ot = od; ot.summary_method = "tk"; ot.tk_min_weight = floor_; ot.tk_eps = eps;
    const BayesianDecayLambdaGrid Gt = bayesian_decay_fit_lambda_grid(ex, E, nodes, F.theta, 1.0, ot);
    std::vector<double> w, dm, tm, above;
    for (std::size_t i = 0; i < Gt.nodes.size(); ++i) {
      w.push_back(Gt.weights[i]);
      dm.push_back(Gd.nodes[i].mean_rel);
      tm.push_back(Gt.nodes[i].tk_ok ? 1.0 : 0.0);
      above.push_back(Gt.weights[i] >= floor_ ? 1.0 : 0.0);
    }
    std::printf(", \"grid\": {\"n_tk\": %zu, \"delta_n_tk\": %zu, \"tk_min_weight\": %.17g", Gt.n_tk, Gd.n_tk, floor_);
    vec("weights", w); vec("delta_mean", dm); vec("tk_ok", tm); vec("above_floor", above);
    std::vector<double> same;
    for (std::size_t i = 0; i < Gt.nodes.size(); ++i) same.push_back(Gt.nodes[i].mean_rel - Gd.nodes[i].mean_rel);
    vec("delta_pair_difference", same);
    std::printf("}");
  }
  std::printf("}\n");
  return 0;
}
"""

TK_COMBINE = r"""
#include <IMP/bff/BayesianLaplace.h>
#include <cstdio>
#include <cstdlib>
using namespace IMP::bff;
int main(int, char** argv) {
  //: the two log ratios are the caller's (closed form for a Gaussian posterior with a linear summary);
  //: this driver exercises only the library's combination of them
  const double r1 = std::atof(argv[1]), r2 = std::atof(argv[2]), eps = std::atof(argv[3]);
  const BayesianTierneyKadane r = bayesian_tierney_kadane(r1, r2, eps);
  std::printf("{\"mean\": %.17g, \"sd\": %.17g, \"ok\": %d}\n", r.mean, r.sd, r.ok ? 1 : 0);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp], timeout=3600)


def test_the_tilts_gradient_and_hessian_are_exact(result):
    """`bayesian_decay_mean_rel_tilt` is why a tilted fit costs no more than an untilted one; a wrong
    derivative would put the tilted mode in the wrong place and bias TK silently. Central differences of
    the tilt itself are the reference, and the tilt must touch only the spline block."""
    nz = result["nz"]
    g, gfd = np.asarray(result["grad"]), np.asarray(result["grad_fd"])
    H, Hfd = np.asarray(result["hess"]).reshape(nz, nz), np.asarray(result["hess_fd"]).reshape(nz, nz)
    assert result["grad_off_block"] == 0.0
    np.testing.assert_allclose(g, gfd, rtol=2e-6, atol=1e-9 * max(1.0, np.abs(gfd).max()))
    np.testing.assert_allclose(H, Hfd, rtol=5e-5, atol=1e-7 * max(1.0, np.abs(Hfd).max()))
    #: the control: the same comparison rejects a Hessian off by 1e-3 of its own scale
    assert not np.allclose(H + 1e-3 * np.abs(H).max(), Hfd, rtol=5e-5, atol=1e-7 * max(1.0, np.abs(Hfd).max()))


def test_tk_reproduces_a_gaussian_posteriors_exact_moments():
    """A Gaussian posterior `N(mu, Sigma)` with a linear summary `f = a'theta + b` has mean `m = a'mu + b`
    and variance `c = a' Sigma a` exactly, and its tilted modes are closed form: for power `k`, with
    `s = f(theta_k) + eps` the positive root of `s^2 - (m + eps) s - k c = 0`, the mode sits at
    `mu + k Sigma a / s`, the tilt's curvature is `k a a' / s^2`, and

        log_ratio_k = -k^2 c / (2 s^2) + k log s - 0.5 log(1 + k c / s^2).

    The combination of those two ratios is the library's, and that is what this pins."""
    m, c, eps = 1.08, 0.031 ** 2, 1e-6           # the CBM56 scale: mean R/R0 and its variance
    ratios = []
    for k in (1.0, 2.0):
        s = 0.5 * ((m + eps) + math.sqrt((m + eps) ** 2 + 4.0 * k * c))
        ratios.append(-0.5 * k * k * c / (s * s) + k * math.log(s) - 0.5 * math.log1p(k * c / (s * s)))
    out = run_driver(TK_COMBINE, args=[repr(ratios[0]), repr(ratios[1]), repr(eps)])
    assert out["ok"] == 1
    assert math.isclose(out["mean"], m, rel_tol=1e-6), (out["mean"], m)
    assert math.isclose(out["sd"], math.sqrt(c), rel_tol=2e-3), (out["sd"], math.sqrt(c))


def test_the_sampled_reference_converged(result):
    """Rule 0c on the reference itself, on the quantity being reported: without this the comparisons below
    would be against noise."""
    mc = result["mcmc"]
    assert mc["rhat"] < 1.01, mc
    assert mc["ess"] > 400, mc


def test_tk_gets_the_posterior_sd_the_delta_method_does_not(result):
    """The declared check of PRD-149 step 3, against the sampled posterior: TK's sd is within 5 %, and the
    delta method's is not -- it linearises a summary that curves over the posterior's width, and on this
    fixture that costs it about half its own value. Both bases of the shared curvature are measured; the
    exact-Hessian one is the better of the two and the Fisher one is the default (a node's exact Hessian
    can be indefinite -- PRD-143 A4.9)."""
    node, mc = result["node"], result["mcmc"]
    assert node["converged"] == 1 and node["tk_ok"] == 1 and node["tkx_ok"] == 1
    for key in ("tk_sd", "tkx_sd"):
        assert abs(node[key] - mc["sd"]) <= 0.05 * mc["sd"], (key, node[key], mc["sd"])
    assert abs(node["delta_sd"] - mc["sd"]) > 0.2 * mc["sd"], (node["delta_sd"], mc["sd"])


def test_tk_mean_is_within_half_a_posterior_sd(result):
    """Declared before the run. TK's mean is an approximation, not an estimator, so it is scored in units
    of the posterior's own width; the delta method's error is recorded beside it."""
    node, mc = result["node"], result["mcmc"]
    assert abs(node["tk_mean"] - mc["mean"]) <= 0.5 * mc["sd"], (node["tk_mean"], mc["mean"], mc["sd"])
    assert abs(node["tkx_mean"] - mc["mean"]) <= 0.5 * mc["sd"], (node["tkx_mean"], mc["mean"], mc["sd"])
    #: and the exact-curvature base is no worse than Fisher's on this fixture
    assert abs(node["tkx_mean"] - mc["mean"]) <= abs(node["tk_mean"] - mc["mean"])


def test_the_tilted_modes_move_by_what_the_theory_says(result):
    """The tilted mode solves `grad log p + power * grad log f = 0`, so it sits about
    `power * sd(f) / f` away from the untilted one in the posterior's own metric. Measuring it is how a
    tilted fit that ran off (or did not move at all) is caught: it is the one number that says the two
    Laplace integrals are being taken at comparable points."""
    node = result["node"]
    expect = node["delta_sd"] / node["delta_mean"]
    for k, key in ((1.0, "shift1"), (2.0, "shift2")):
        assert node[key] == pytest.approx(k * expect, rel=0.35), (key, node[key], k * expect)


def test_tk_runs_only_where_the_node_carries_weight(result):
    g = result["grid"]
    ok = np.asarray(g["tk_ok"], dtype=bool)
    above = np.asarray(g["above_floor"], dtype=bool)
    assert g["n_tk"] == int(ok.sum())
    assert np.all(ok <= above), (g["weights"], g["tk_ok"])
    assert above.sum() < len(above), "the floor excluded nothing: the check cannot fail"
    assert ok.any(), "no node passed the floor at all"


def test_the_default_summary_method_is_unchanged(result):
    """TK is opt-in: with `summary_method = "delta"` no node carries a TK number, and the TK pass leaves
    the delta-method pair of every node exactly as it was."""
    g = result["grid"]
    assert g["delta_n_tk"] == 0
    assert all(v == 0.0 for v in g["delta_pair_difference"]), g["delta_pair_difference"]

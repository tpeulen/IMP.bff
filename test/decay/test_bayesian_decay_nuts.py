"""PRD-146 step 2: NUTS on the Bayesian decay posterior (`BayesianDecaySampling.h`; since PRD-147 step 5 through
run_sampler with NutsKernel and summarize_chains).

On the synthetic experiment with data drawn from the model:

* `bayesian_decay_log_posterior_and_gradient` equals `bayesian_decay_evaluate`'s log posterior and
  gradient (to 1e-12 of the peak);
* two independent runs of 4 chains each (different seeds and dispersed starts around the mode,
  the Laplace covariance as metric) agree on the posterior means of every coordinate within
  4 combined Monte Carlo standard errors;
* they do NOT reach rank R-hat < 1.01 (expected failure, strict): this synthetic experiment clamps
  its shifted response at zero on purpose, the posterior has kinks there, and NUTS reports
  divergent transitions in exactly those coordinates;
* a run whose chains are stopped after 5 draws without warm-up is not declared converged (the
  control: R-hat or ESS must flag it).

Written 2026-09-15 (ucfret prompt 463).
"""

import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecaySampling.h>
#include <cstdio>
#include <random>
using namespace IMP::bff;

int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::size_t dim = ex.dim;
  std::vector<double> truth(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(truth.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, truth));
  std::mt19937_64 rng(3);
  for (std::size_t i = 0; i < lam.size(); ++i) ex.arrays["y"].d[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
  // gradient path
  const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, truth);
  std::vector<double> g;
  const double lp = bayesian_decay_log_posterior_and_gradient(ex, E, truth, g);
  double gd = 0, gp = 0;
  for (std::size_t i = 0; i < dim; ++i) { gd = std::max(gd, std::fabs(g[i] - pt.grad[i])); gp = std::max(gp, std::fabs(pt.grad[i])); }
  // the mode and its Laplace covariance
  const BayesianDecayFit F = bayesian_decay_fit_node(ex, E, truth, 1.0, 400);
  const std::vector<double> Sig = bayesian_decay_covariance(F.pt, dim);
  const SamplingTarget target = bayesian_decay_sampling_target(ex, E);
  NutsOptions nopt; nopt.inverse_metric = Sig;
  auto run = [&](std::uint64_t seed, int warm, int draws) {
    std::vector<std::vector<std::vector<double>>> per_coord(dim, std::vector<std::vector<double>>(4));
    std::normal_distribution<double> N(0.0, 1.0);
    std::mt19937_64 r2(seed);
    std::vector<std::vector<std::vector<double>>> starts;
    for (int c = 0; c < 4; ++c) {
      std::vector<double> start = F.theta;
      for (std::size_t i = 0; i < dim; ++i) start[i] += 0.5 * std::sqrt(Sig[i * dim + i]) * N(r2);
      starts.push_back({start});
    }
    SamplerOptions so; so.warmup = warm; so.draws = draws; so.seed = seed * 10;
    const SampleResult r = run_sampler(target, NutsKernel(nopt), starts, so);
    for (std::size_t i = 0; i < dim; ++i) per_coord[i] = r.independent_chains([i](const std::vector<double>& x) { return x[i]; });
    return per_coord;
  };
  const auto A = run(1, 300, 400), B = run(2, 300, 400), C = run(3, 0, 5);
  double worst_z = 0, worst_rhat = 0, min_ess = 1e300, ctl_rhat = 0, ctl_ess = 1e300;
  for (std::size_t i = 0; i < dim; ++i) {
    const ChainSummary a = summarize_chains(A[i]), b = summarize_chains(B[i]), c = summarize_chains(C[i]);
    worst_z = std::max(worst_z, std::fabs(a.mean - b.mean) / std::sqrt(a.mcse * a.mcse + b.mcse * b.mcse));
    worst_rhat = std::max({worst_rhat, a.rhat, b.rhat});
    min_ess = std::min({min_ess, a.ess_bulk, b.ess_bulk});
    ctl_rhat = std::max(ctl_rhat, c.rhat);
    ctl_ess = std::min(ctl_ess, c.ess_bulk);
  }
  std::printf("{\"grad_err\": %.3e, \"lp_err\": %.3e, \"worst_mean_z\": %.4f, \"worst_rhat\": %.5f, \"min_ess\": %.1f,"
              " \"control_rhat\": %.4f, \"control_ess\": %.2f}\n", gd / gp, std::fabs(lp - pt.logpost), worst_z, worst_rhat, min_ess, ctl_rhat, ctl_ess);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp], timeout=1800)


def test_gradient_path_equals_evaluate(result):
    assert result["grad_err"] < 1e-12 and result["lp_err"] < 1e-8, result


def test_two_independent_runs_agree_on_the_means(result):
    assert result["worst_mean_z"] < 4.0, result


@pytest.mark.xfail(strict=True, reason="PRD-146: the synthetic posterior is not smooth -- divergent transitions "
                   "(5-25 % of draws) sit in the response-shift and reference coordinates, where the shifted response "
                   "is clamped at zero; R-hat 1.06, ESS 88 at target acceptance 0.8. Strict: passing means it was fixed.")
def test_two_independent_runs_converge(result):
    assert result["worst_rhat"] < 1.01, result
    assert result["min_ess"] > 100, result


def test_a_run_too_short_is_not_declared_converged(result):
    assert result["control_rhat"] > 1.01 or result["control_ess"] < 100, result

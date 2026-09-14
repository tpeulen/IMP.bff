"""Whether a fitted count model describes its histograms (PRD-142 step 2, merged
into `FitStatistics.h` by PRD-143).

`poisson_goodness_of_fit` gives, per histogram and overall, the Poisson deviance
per degree of freedom against a reference measured by drawing Poisson data at
the fitted means, and the runs test on the residual signs. Checked:

* the runs-test p is statsmodels' `runstest_1samp(cutoff='mean', correction=False)`
  on random, alternating and blocked sequences;
* data drawn from the model pass (|z| < 3, runs p not small);
* a model whose means are 5 % off fails the deviance test, and a model with a
  slow 3 % wave in it fails the runs test (at these counts the deviance too).

Written 2026-09-14 (ucfret prompt 432).
"""

import numpy as np
import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/FitStatistics.h>
#include <cstdio>
#include <iostream>
using namespace IMP::bff;

int main(int argc, char** argv) {
  // stdin: n_seq, then each sequence as "len v v v ..."; then the histogram case is built here
  std::size_t n_seq; std::cin >> n_seq;
  std::printf("{\"runs_p\": [");
  for (std::size_t s = 0; s < n_seq; ++s) {
    std::size_t len; std::cin >> len; std::vector<double> x(len);
    for (double& v : x) std::cin >> v;
    double mean = 0; for (double t : x) mean += t; if (!x.empty()) mean /= double(x.size());
    std::printf("%s%.17g", s ? ", " : "", runs_test(x.data(), x.size(), mean).p_value);
  }
  std::printf("], ");
  // three histograms of a decay, 400 bins, counts from ~1e4 down to ~5
  const std::size_t nh = 3, n = 400;
  std::vector<double> m(nh * n), mask(nh * n, 1.0);
  for (std::size_t k = 0; k < nh; ++k) for (std::size_t i = 0; i < n; ++i) m[k * n + i] = 5.0 + 1e4 * std::exp(-double(i) / (60.0 + 20.0 * k));
  std::mt19937_64 rng(42);
  std::vector<double> y(nh * n);
  for (std::size_t j = 0; j < y.size(); ++j) y[j] = double(std::poisson_distribution<long long>(m[j])(rng));
  auto report = [&](const char* name, const std::vector<double>& model) {
    const PoissonGoodnessOfFit G = poisson_goodness_of_fit(y, model, mask, nh, 6.0);
    double zmax = 0.0, pmin = 1.0;
    for (auto& h : G.histograms) { zmax = std::max(zmax, std::fabs(h.z)); pmin = std::min(pmin, h.runs_p); }
    std::printf("\"%s\": {\"z\": %.6g, \"z_max\": %.6g, \"runs_p_min\": %.6g, \"dpd\": %.6g, \"ref\": %.6g}", name, G.z, zmax, pmin, G.deviance_per_dof, G.reference_mean);
  };
  report("true", m); std::printf(", ");
  std::vector<double> off(m); for (double& v : off) v *= 1.05;
  report("scaled", off); std::printf(", ");
  std::vector<double> wave(m);
  for (std::size_t k = 0; k < nh; ++k) for (std::size_t i = 0; i < n; ++i) wave[k * n + i] *= 1.0 + 0.03 * std::sin(2.0 * M_PI * double(i) / 150.0);
  report("wave", wave); std::printf("}\n");
  return 0;
}
"""


def _sequences():
    rng = np.random.default_rng(3)
    seqs = [np.sign(rng.normal(size=500)), rng.normal(size=77), np.tile([1.0, -1.0], 60),
            np.repeat([1.0, -1.0, 1.0, -1.0], 40), np.sign(rng.normal(size=51)) + 0.0 * np.arange(51)]
    seqs[4][[3, 10]] = 0.0          # zeros in a sign sequence, as exact residuals give
    return seqs


@pytest.fixture(scope="module")
def result():
    seqs = _sequences()
    text = f"{len(seqs)}\n" + "\n".join(f"{len(s)} " + " ".join(repr(float(v)) for v in s) for s in seqs) + "\n"
    return run_driver(DRIVER, stdin=text), seqs


def test_runs_test_is_statsmodels(result):
    runs = pytest.importorskip("statsmodels.sandbox.stats.runs")
    res, seqs = result
    for p, s in zip(res["runs_p"], seqs):
        ref = runs.runstest_1samp(np.asarray(s, float), cutoff="mean", correction=False)[1]
        assert abs(p - ref) < 1e-12, (p, ref)


def test_the_true_model_passes(result):
    res, _ = result
    assert abs(res["true"]["z"]) < 3 and res["true"]["z_max"] < 3, res["true"]
    assert res["true"]["runs_p_min"] > 0.001, res["true"]


def test_a_model_five_percent_off_fails_the_deviance(result):
    res, _ = result
    assert res["scaled"]["z"] > 3, res["scaled"]


def test_a_slow_wave_fails_the_runs_test(result):
    res, _ = result
    assert res["wave"]["runs_p_min"] < 1e-3, res["wave"]

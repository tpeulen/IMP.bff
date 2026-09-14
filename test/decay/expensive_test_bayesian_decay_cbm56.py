"""PRD-142 test 5: the CBM56 measurement, fitted through bff's headers, against
the Python prototype's mode.

The fixture is ucfret's (`investigation/pinn_pR_anisotropy/s89_cpp/
emit_cbm56_fixture.py` writes it: the frozen CBM56 model -- donor-only, FRET and
Rh110 samples in two polarised green detectors, both pulse windows -- its data,
its start, and the torch prototype's Newton-polished mode). Named by
`BFF_CBM56_FIXTURE`; skipped without it, since 37 MB of measurement does not
belong in this repository.

From the fixture's start, one penalty node, the fit must land on the Python mode:
evidence within 0.05 nats, p(R/R0) and its delta-method sd within 1e-4 of the
peak, every histogram's deviance per dof within 1e-3. Its wall time is reported
(2.2 s on an M-series Mac with 4 threads, 2026-09-14). The acceptance values were
declared in ucfret's `okf/prd-cpp-real-data.md` before the port.

Written 2026-09-14 (ucfret prompt 432).
"""

import os

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/BayesianDecayPosterior.h>
#include <IMP/bff/BayesianGoodnessOfFit.h>
#include <chrono>
#include <cstdio>
using namespace IMP::bff;

int main(int, char** argv) {
  const auto t0 = std::chrono::steady_clock::now();
  BayesianDecayExperiment ex;
  const nlohmann::json m = bayesian_decay_experiment_load(argv[1], ex);
  const BayesianDecayTensors E(ex);
  const std::size_t dim = ex.dim;
  const BayesianDecayFit F = bayesian_decay_fit_node(ex, E, ex["theta_start"].d, 1.0, 1000);
  const std::vector<double> S = bayesian_decay_covariance(F.pt, dim);
  std::vector<double> p, sd;
  bayesian_decay_distribution_with_sd(ex, E, F.theta, S, p, sd);
  const BayesianGoodnessOfFit G = bayesian_goodness_of_fit(ex["y"].d, F.pt.lam, ex["mask"].d, ex.data_keys.size(), double(dim));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto& pol = m["polished"];
  const auto& pr = ex["p_polished"].d, & sr = ex["p_sd_polished"].d;
  double pk = 0, dp = 0, dsd = 0;
  for (std::size_t j = 0; j < p.size(); ++j) { pk = std::max(pk, pr[j]); dp = std::max(dp, std::fabs(p[j] - pr[j])); dsd = std::max(dsd, std::fabs(sd[j] - sr[j])); }
  double ddpd = 0;
  for (std::size_t k = 0; k < G.histograms.size(); ++k) ddpd = std::max(ddpd, std::fabs(G.histograms[k].deviance_per_dof - pol["dpd"][k].get<double>()));
  std::printf("{\"converged\": %s, \"iterations\": %d, \"evidence_difference\": %.6f, \"p\": %.3e, \"p_sd\": %.3e,"
              " \"dpd\": %.3e, \"z\": %.3f, \"seconds\": %.3f}\n", F.converged ? "true" : "false", F.iterations,
              F.evidence - pol["evidence_fisher"].get<double>(), dp / pk, dsd / pk, ddpd, G.z, secs);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    fx = os.environ.get("BFF_CBM56_FIXTURE")
    if not fx or not os.path.isfile(os.path.join(fx, "manifest.json")):
        pytest.skip("BFF_CBM56_FIXTURE does not name a CBM56 fixture (ucfret s89_cpp/emit_cbm56_fixture.py writes one)")
    return run_driver(DRIVER, args=[fx], timeout=1200)


def test_the_fit_lands_on_the_python_mode(result):
    print(result)
    assert result["converged"], result
    assert abs(result["evidence_difference"]) < 0.05, result
    assert result["p"] < 1e-4 and result["p_sd"] < 1e-4, result
    assert result["dpd"] < 1e-3, result

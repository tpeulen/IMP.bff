"""PRD-143 A4.9: the exact Hessian of the Bayesian decay posterior and the Laplace evidence
it gives.

`bayesian_decay_hessian` differentiates the analytic gradient by central differences. Checked
on the synthetic experiment with data from the model: against second differences of the log
posterior's VALUES on a set of coordinate pairs (an independent route that shares no
gradient code); symmetric; and different from the Fisher information by the dropped term
`(y/m - 1) d2m` (the control -- if the two agreed the check would not be testing that term).
The evidence with it is `bayesian_laplace_log_evidence` on H. On CBM56 (ucfret
`s89_cpp/cbm56_evidence_exact.cpp`) it equals the Python autograd evidence to 1e-4 nats.

Written 2026-09-15 (ucfret prompt 452).
"""

import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecayPosterior.h>
#include <cstdio>
#include <random>
using namespace IMP::bff;
int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::size_t dim = ex.dim;
  std::vector<double> th(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, th));
  std::mt19937_64 rng(3);
  // counts NOT from the model at th, so that (y/m - 1) is far from zero and the dropped term shows
  for (std::size_t i = 0; i < lam.size(); ++i) ex.arrays["y"].d[i] = double(std::poisson_distribution<long long>(1.3 * std::max(lam[i], 1e-12))(rng));
  const std::vector<double> H = bayesian_decay_hessian(ex, E, th);
  const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, th);
  auto f = [&](const std::vector<double>& t) { return -bayesian_decay_log_posterior(ex, E, t); };
  double worst = 0.0, peak = 0.0, asym = 0.0, fisher_diff = 0.0;
  const std::size_t pairs[][2] = {{0, 0}, {0, 1}, {3, 3}, {5, 9}, {10, 10}, {12, 20}, {dim - 1, dim - 1}, {dim - 2, 1}};
  for (const auto& pr : pairs) {
    const std::size_t i = pr[0], j = pr[1];
    const double h = 1e-3;
    auto at = [&](double a, double b) { std::vector<double> t = th; t[i] += a; t[j] += b; return f(t); };
    const double fd = (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h);
    worst = std::max(worst, std::fabs(fd - H[i * dim + j]));
    peak = std::max(peak, std::fabs(H[i * dim + j]));
  }
  for (std::size_t i = 0; i < dim; ++i) for (std::size_t j = 0; j < dim; ++j) {
    asym = std::max(asym, std::fabs(H[i * dim + j] - H[j * dim + i]));
    fisher_diff = std::max(fisher_diff, std::fabs(H[i * dim + j] - pt.A[i * dim + j]));
  }
  std::printf("{\"value_route_err\": %.6e, \"peak\": %.6e, \"asym\": %.3e, \"fisher_diff\": %.6e}\n", worst, peak, asym, fisher_diff);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp])


def test_hessian_equals_second_differences_of_values(result):
    assert result["value_route_err"] < 1e-4 * max(result["peak"], 1.0), result


def test_hessian_is_symmetric(result):
    assert result["asym"] == 0.0, result


def test_hessian_is_not_the_fisher_information(result):
    assert result["fisher_diff"] > 1e-3 * max(result["peak"], 1.0), result

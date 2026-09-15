"""PRD-143 A4.7 and A4.6: declared starts with the best kept, and the P-spline weight
integrated over an evidence-weighted grid.

On the synthetic experiment with data drawn from the model:

* `bayesian_decay_fit_best_of_starts` returns every start's fit and keeps the highest
  log posterior (checked against the maximum computed here); a start placed far away fails
  (a non-finite log posterior), is listed first, and still never wins; an Adam phase runs
  and its fit converges;
* `bayesian_decay_fit_lambda_grid`: weights are the normalised exp(evidence), the mixture
  mean is their weighted mean, and its sd includes the spread between nodes, so it is at
  least the weighted within-node sd (the law of total variance) -- equality only if the
  nodes agree, which the control shows they do not.

Written 2026-09-15 (ucfret prompt 452). On CBM56 (ucfret `s89_cpp/cbm56_best.cpp`): three of
four declared starts reach the polished mode, the plain data start the second mode.
"""

import math
import os
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
  std::vector<double> truth(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(truth.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, truth));
  std::mt19937_64 rng(5);
  for (std::size_t i = 0; i < lam.size(); ++i) ex.arrays["y"].d[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
  std::normal_distribution<double> N(0.0, 1.0);
  std::vector<double> near = truth, far = truth;
  for (std::size_t i = 0; i < dim; ++i) { near[i] += 0.05 * N(rng); far[i] += 2.5 * N(rng); }
  // the failed start first: a selection that compared NaN would keep it
  std::vector<BayesianDecayStart> starts = {{"far", far, 1.0, 0, 0.0}, {"near", near, 1.0, 0, 0.0}, {"near + Adam", near, 1.0, 30, 0.01}};
  const BayesianDecayMultiFit M = bayesian_decay_fit_best_of_starts(ex, E, starts, 400, 3);
  std::size_t argmax = 1;
  for (std::size_t i = 0; i < M.fits.size(); ++i) if (std::isfinite(M.fits[i].pt.logpost) && M.fits[i].pt.logpost > M.fits[argmax].pt.logpost) argmax = i;
  const auto G = bayesian_decay_fit_lambda_grid(ex, E, {-1.0, 0.0, 1.0, 2.0}, M.fits[M.best].theta, 1.0, 400);
  double wsum = 0.0, wmean = 0.0, within = 0.0, emax = -1e300, wmax_err = 0.0;
  for (const auto& n : G.nodes) emax = std::max(emax, n.fit.evidence);
  double zsum = 0.0; for (const auto& n : G.nodes) zsum += std::exp(n.fit.evidence - emax);
  for (std::size_t i = 0; i < G.nodes.size(); ++i) {
    wsum += G.weights[i]; wmean += G.weights[i] * G.nodes[i].mean_rel; within += G.weights[i] * G.nodes[i].mean_rel_sd * G.nodes[i].mean_rel_sd;
    wmax_err = std::max(wmax_err, std::fabs(G.weights[i] - std::exp(G.nodes[i].fit.evidence - emax) / zsum));
  }
  double spread = 0.0;
  for (std::size_t i = 0; i < G.nodes.size(); ++i) spread = std::max(spread, std::fabs(G.nodes[i].mean_rel - wmean));
  auto fin = [](double v) { return std::isfinite(v) ? v : -1e300; };
  std::printf("{\"lp\": [%.10g, %.10g, %.10g], \"best\": %zu, \"argmax\": %zu, \"converged\": [%d, %d, %d],"
              " \"wsum\": %.15g, \"weight_formula_err\": %.3e, \"mix_mean\": %.15g, \"weighted_mean\": %.15g,"
              " \"mix_sd\": %.15g, \"within_sd\": %.15g, \"node_spread\": %.6e, \"sorted\": %d}\n",
              fin(M.fits[0].pt.logpost), fin(M.fits[1].pt.logpost), fin(M.fits[2].pt.logpost), M.best, argmax,
              int(M.fits[0].converged), int(M.fits[1].converged), int(M.fits[2].converged),
              fin(wsum), fin(wmax_err), G.mean_rel, wmean, G.mean_rel_sd, std::sqrt(within), spread,
              int(G.nodes[0].log10_lam < G.nodes.back().log10_lam));
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp])


def test_best_is_the_highest_log_posterior(result):
    assert result["best"] == result["argmax"], result
    assert result["lp"][0] < max(result["lp"][1], result["lp"][2]), result   # the far start ends lower (here: fails)
    assert result["best"] != 0, result
    assert result["converged"][1] == 1 and result["converged"][2] == 1, result


def test_grid_weights_and_mixture(result):
    assert abs(result["wsum"] - 1.0) < 1e-12, result
    assert result["weight_formula_err"] < 1e-12, result
    assert abs(result["mix_mean"] - result["weighted_mean"]) < 1e-12, result
    assert result["sorted"] == 1, result


def test_mixture_sd_includes_the_spread_between_nodes(result):
    assert result["mix_sd"] >= result["within_sd"] - 1e-12, result
    # the control: the nodes disagree, so the between-node term is not zero
    assert result["node_spread"] > 1e-4, result
    assert result["mix_sd"] > result["within_sd"] + 1e-9, result

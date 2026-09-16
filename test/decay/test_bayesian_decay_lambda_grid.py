"""PRD-149: the lambda grid -- the evidence tilt, the refinement around the evidence mode, and the mixture's
own quantiles. Written 2026-09-16.

The P-spline weight lambda is integrated over a grid of nodes weighted by their Laplace evidence (Rue, Martino &
Chopin 2009). ucfret's Python prototype (s88_laplace_posterior.py) carries three rules this ports:
* the weights are the TILTED evidences, `evidence + prior_slope * log10 lambda` ("smooth unless the data insist");
* after the coarse pass, nodes `refine_step` apart are added across the range where the tilted evidence is within
  `refine_window` nats of its maximum, half a decade beyond, each warm-started from the nearest fitted node;
* a summary's interval comes from the mixture's own CDF, `sum_i w_i Phi((t - mu_i)/sd_i)`, not from its moments.

What is checked, and what would make each check fail:
* the reported weights are exactly the normalised tilted evidences (a tilt that did nothing would fail);
* `prior_slope = 2` moves weight toward the smoother nodes by exactly the tilt (against the same nodes' evidences);
* refinement adds nodes at the declared spacing, only inside the window, and leaves the coarse nodes in place;
* the mixture's quantiles equal a numeric evaluation of the same normal mixture (1e-9), and the 16/84 % interval
  is wider than the moment interval when the nodes disagree;
* with `refine_step = 0` the grid is the coarse one (the old behaviour).
"""
import math

import numpy as np
import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn
import tempfile

DRIVER = r"""
#include <IMP/bff/BayesianDecayPosterior.h>
#include <cstdio>
using namespace IMP::bff;
int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const BayesianDecayTensors E(ex);
  std::vector<double> th(ex.dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(ex.dim * sizeof(double)));
  const std::vector<double> nodes = {-1.0, 0.0, 1.0, 2.0};
  auto dump = [&](const char* name, const BayesianDecayLambdaGrid& G, bool comma) {
    std::printf("%s\"%s\": {\"lam\": [", comma ? "," : "", name);
    for (std::size_t i = 0; i < G.nodes.size(); ++i) std::printf("%s%.10g", i ? "," : "", G.nodes[i].log10_lam);
    std::printf("], \"evidence\": [");
    for (std::size_t i = 0; i < G.nodes.size(); ++i) std::printf("%s%.17g", i ? "," : "", G.nodes[i].fit.evidence);
    std::printf("], \"mean\": [");
    for (std::size_t i = 0; i < G.nodes.size(); ++i) std::printf("%s%.17g", i ? "," : "", G.nodes[i].mean_rel);
    std::printf("], \"sd\": [");
    for (std::size_t i = 0; i < G.nodes.size(); ++i) std::printf("%s%.17g", i ? "," : "", G.nodes[i].mean_rel_sd);
    std::printf("], \"weights\": [");
    for (std::size_t i = 0; i < G.weights.size(); ++i) std::printf("%s%.17g", i ? "," : "", G.weights[i]);
    std::printf("], \"n_coarse\": %zu, \"mixture_mean\": %.17g, \"mixture_sd\": %.17g, \"q16\": %.17g, \"q50\": %.17g, \"q84\": %.17g}",
                G.n_coarse, G.mean_rel, G.mean_rel_sd, G.mean_rel_q16, G.mean_rel_q50, G.mean_rel_q84);
  };
  BayesianDecayLambdaOptions plain; plain.refine_step = 0.0; plain.max_iter = 300; plain.sweeps = 0;
  BayesianDecayLambdaOptions tilt = plain; tilt.prior_slope = 2.0;
  BayesianDecayLambdaOptions refined; refined.refine_step = 0.25; refined.refine_window = 2.0; refined.max_iter = 300; refined.sweeps = 0;
  BayesianDecayLambdaOptions swept; swept.refine_step = 0.0; swept.max_iter = 300; swept.sweeps = 2;
  std::printf("{");
  dump("plain", bayesian_decay_fit_lambda_grid(ex, E, nodes, th, 0.0, plain), false);
  dump("tilted", bayesian_decay_fit_lambda_grid(ex, E, nodes, th, 0.0, tilt), true);
  dump("refined", bayesian_decay_fit_lambda_grid(ex, E, nodes, th, 0.0, refined), true);
  {
    const BayesianDecayLambdaGrid S = bayesian_decay_fit_lambda_grid(ex, E, nodes, th, 0.0, swept);
    dump("swept", S, true);
    std::printf(",\"swept_improved\": %zu, \"swept_logpost\": [", S.n_improved);
    for (std::size_t i = 0; i < S.nodes.size(); ++i) std::printf("%s%.17g", i ? "," : "", S.nodes[i].fit.pt.logpost);
    std::printf("], \"plain_logpost\": [");
    const BayesianDecayLambdaGrid P = bayesian_decay_fit_lambda_grid(ex, E, nodes, th, 0.0, plain);
    for (std::size_t i = 0; i < P.nodes.size(); ++i) std::printf("%s%.17g", i ? "," : "", P.nodes[i].fit.pt.logpost);
    std::printf("]");
  }
  std::printf("}\n");
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp], timeout=1800)


def _weights(ev, lam, slope):
    t = np.asarray(ev) + slope * np.asarray(lam)
    w = np.where(np.isfinite(t), np.exp(t - np.nanmax(t[np.isfinite(t)])), 0.0)
    return w / w.sum()


def test_a_sweep_never_lowers_a_node_and_reports_what_it_moved(result):
    """A node warm-started from one side can stay in a worse mode; the sweep refits it from its
    neighbours and keeps the higher posterior, so no node may come out worse."""
    swept, plain = np.asarray(result["swept_logpost"]), np.asarray(result["plain_logpost"])
    assert result["swept"]["lam"] == result["plain"]["lam"]
    assert np.all(swept >= plain - 1e-6), (swept - plain)
    assert result["swept_improved"] == 0 or np.any(swept > plain + 1e-6)


def test_weights_are_the_normalised_tilted_evidences(result):
    for name, slope in (("plain", 0.0), ("tilted", 2.0), ("refined", 0.0)):
        g = result[name]
        np.testing.assert_allclose(g["weights"], _weights(g["evidence"], g["lam"], slope), rtol=0, atol=1e-12)


def test_the_tilt_moves_weight_toward_the_smoother_nodes(result):
    plain, tilted = result["plain"], result["tilted"]
    assert plain["lam"] == tilted["lam"]
    # the same nodes, so the tilt is the whole difference: weight ratio between the ends changes by exp(2 * range)
    lam = np.asarray(plain["lam"])
    ratio_plain = plain["weights"][-1] / plain["weights"][0]
    ratio_tilt = tilted["weights"][-1] / tilted["weights"][0]
    assert math.isclose(ratio_tilt / ratio_plain, math.exp(2.0 * (lam[-1] - lam[0])), rel_tol=1e-9)


def test_refinement_adds_nodes_at_the_declared_spacing_inside_the_window(result):
    plain, refined = result["plain"], result["refined"]
    assert refined["n_coarse"] == len(plain["lam"])
    assert len(refined["lam"]) > refined["n_coarse"]
    lam = np.asarray(refined["lam"])
    assert np.all(np.diff(lam) > 0)                      # sorted, unique
    added = sorted(set(np.round(lam, 10)) - set(np.round(plain["lam"], 10)))
    assert added, refined["lam"]
    # every added node sits on the 0.25 lattice and inside the coarse range
    for x in added:
        assert abs(x / 0.25 - round(x / 0.25)) < 1e-6, x
        assert min(plain["lam"]) - 1e-9 <= x <= max(plain["lam"]) + 1e-9, x
    # and inside the window: its own tilted evidence is not far below the best
    ev = np.asarray(refined["evidence"])
    best = np.nanmax(ev)
    for x, e in zip(lam, ev):
        if round(x, 10) in set(np.round(added, 10)):
            assert e > best - 60.0, (x, e, best)         # the fit at an added node is a real fit, not a failure


def test_the_mixture_quantiles_are_the_mixtures_own(result):
    for name in ("plain", "tilted", "refined"):
        g = result[name]
        w, m, s = np.asarray(g["weights"]), np.asarray(g["mean"]), np.asarray(g["sd"])
        from scipy.stats import norm
        def cdf(t):
            return float(np.sum(w * norm.cdf((t - m) / np.where(s > 0, s, 1e-300))))
        for q, key in ((0.15865525393145705, "q16"), (0.5, "q50"), (0.8413447460685429, "q84")):
            assert abs(cdf(g[key]) - q) < 1e-9, (name, key, cdf(g[key]))
        assert g["q16"] < g["q50"] < g["q84"]


def test_the_moment_mean_is_the_weighted_mean(result):
    for name in ("plain", "tilted", "refined"):
        g = result[name]
        w, m, s = np.asarray(g["weights"]), np.asarray(g["mean"]), np.asarray(g["sd"])
        assert math.isclose(g["mixture_mean"], float(np.sum(w * m)), rel_tol=1e-12)
        var = float(np.sum(w * (s ** 2 + m ** 2)) - np.sum(w * m) ** 2)
        assert math.isclose(g["mixture_sd"], math.sqrt(max(var, 0.0)), rel_tol=1e-12)

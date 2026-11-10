"""PRD-155 step 8: the scored factor graph from Python -- what a caller such as ucfret uses.
Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.

* a Python subclass of `InferenceFactorScore` (a director) makes a toy graph that
  `inference_fitting` fits to its closed-form mode and evidence -- any model can be written this way;
* `bayesian_decay_factor_graph(directory)` gives the decay model as a graph; fitting it from
  Python gives the same fit as the C++ test (`test_bayesian_decay_factor_graph.py`), with the log
  posterior split by kind, the counts' prediction and data for the fit statistic (Poisson deviance
  per degree of freedom with its reference, runs test), p(R) on its grid, and each observation
  met within its errors (the observations in the docking restraint's vocabulary).
"""
import json
import math
import tempfile

import numpy as np
import pytest

bff = pytest.importorskip("IMP.bff")
if not hasattr(bff, "inference_fitting"):
    pytest.skip("this IMP.bff has no inference_fitting", allow_module_level=True)

from test_bayesian_decay_factor_graph import OBSERVATIONS, R0, experiment


class Gauss(bff.InferenceFactorScore):
    def __init__(self, m, s):
        super().__init__("gauss")
        self.m, self.s = m, s

    def eval(self, x, derivatives):
        r = (x[0] - self.m) / self.s
        lp = -0.5 * r * r - math.log(self.s) - 0.5 * math.log(2 * math.pi)
        return np.array([lp, -r / self.s, 1 / self.s ** 2]) if derivatives else np.array([lp])


def test_a_python_scored_graph_is_fitted_exactly():
    g = bff.InferenceFactorGraph()
    g.add_variable("x", "x", 0)
    for key, kind, (m, s) in (("prior", bff.INFERENCE_FACTOR_PRIOR, (1.0, 2.0)),
                              ("y1", bff.INFERENCE_FACTOR_LIKELIHOOD, (3.0, 0.5)),
                              ("y2", bff.INFERENCE_FACTOR_LIKELIHOOD, (2.0, 1.0))):
        g.add_factor(key, kind, ["x"])
        g.set_factor_score(key, Gauss(m, s))
    f = bff.inference_fitting(g, [-5.0])
    P = 1 / 4 + 1 / 0.25 + 1
    mode = (1 / 4 + 3 / 0.25 + 2) / P
    assert f.converged and abs(f.theta[0] - mode) < 1e-10
    assert abs(f.at_mode.log_density - (f.at_mode.log_prior + f.at_mode.log_likelihood)) < 1e-12


def _poisson_reference_z(fit_y, m, mask, n_hist, n_params):
    q = bff.poisson_goodness_of_fit(list(fit_y), list(m), list(mask), n_hist, float(n_params))
    return q


@pytest.mark.parametrize("observed", [False, True], ids=["plain", "observed"])
def test_the_decay_graph_from_python(observed):
    d = tempfile.mkdtemp()
    experiment(d, OBSERVATIONS if observed else None)
    g = bff.bayesian_decay_factor_graph(d)
    theta0 = np.fromfile(f"{d}/theta.bin", dtype="<f8")
    f = bff.inference_fitting(g, theta0.tolist())
    assert f.converged
    s = f.at_mode
    assert abs(s.log_density - (s.log_likelihood + s.log_prior + s.log_hyper)) <= 1e-12 * abs(s.log_density)
    counts = g.get_factor_score("counts")
    m = np.asarray(counts.predict(g.get_factor_values("counts", f.theta)))
    y = np.asarray(counts.get_data())
    manifest = json.load(open(f"{d}/manifest.json"))
    mask = np.fromfile(f"{d}/mask.bin", dtype="<f8")
    n_hist = len(manifest["data_keys"])
    q = bff.poisson_goodness_of_fit(y.tolist(), m.tolist(), mask.tolist(), n_hist, float(len(theta0)))
    z = [h.z for h in q.histograms]
    runs = [h.runs_p for h in q.histograms]
    print({"deviance_per_dof": q.deviance_per_dof, "dof": q.dof, "z": z, "runs_p": runs})
    assert max(abs(v) for v in z) < 3 and min(runs) > 0.001
    p = np.asarray(bff.bayesian_decay_distribution_at(d, list(f.theta)))
    grid = np.asarray(bff.bayesian_decay_distance_grid(d))
    assert p.size == grid.size and abs(p.sum() - 1) < 1e-12
    if observed:
        stats = bff.distance_sample_statistics(list(grid * R0), list(p), R0)   # <R>, R_E, <E>, sd
        expect = {"RDAMean": stats[0], "RDAMeanE": stats[1], "Efficiency": stats[2]}
        for o in OBSERVATIONS:
            key = "observation:" + o["name"]
            m = g.get_factor_score(key).predict(g.get_factor_values(key, f.theta))[0]
            assert abs(m - expect[o["distance_type"]]) < 1e-12
            err = o["error_neg"] if m < o["distance"] else o["error_pos"]
            assert abs(m - o["distance"]) < 3 * err

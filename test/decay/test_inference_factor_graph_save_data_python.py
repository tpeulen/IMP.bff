"""PRD-156 steps 2-3 from Python, on the wheel: a held variable, and a graph saved with its data in
one process and loaded in another through bff's real registry. Written 2026-11-11 by
claude-code/Opus 5.5 for tpeulen.

The C++ drivers (`test_inference_factor_graph_save_data.py`) stand in for the registry; this runs
the registered factory (`registry` capability `factor_score`, type `bayesian_decay_experiment`).

What is checked, and what would make each check fail:
* the score type is in the registry under `factor_score`;
* `save(path)` without data is `to_json()`'s text; `to_msgpack(True)` is `bytes`;
* a decay graph with three FRET observations, saved with data here and loaded by a separate Python
  process, scores the same log density at a point and fits to the same mode (bit for bit, compared
  as 17-digit text);
* a graph with a Python-subclassed score refuses `embed_data=True`, naming the factor;
* `inference_fitting(..., fixed=[...])` from Python holds the variable and records it.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

bff = pytest.importorskip("IMP.bff")
if not hasattr(bff.InferenceFactorGraph, "to_msgpack"):
    pytest.skip("this IMP.bff has no graph save with data (PRD-156)", allow_module_level=True)

from test_bayesian_decay_factor_graph import OBSERVATIONS, experiment

REPORT = r"""
import json, sys
import numpy as np
import IMP.bff as bff
g = bff.InferenceFactorGraph()
g.load(sys.argv[1])
theta = np.fromfile(sys.argv[2], dtype="<f8").tolist()
s = g.eval(theta, False)
f = bff.inference_fitting(g, theta, 300, 1e-6)
print(json.dumps({"log_density": repr(s.log_density), "fit_logpost": repr(f.at_mode.log_density),
                  "fit_evidence": repr(f.laplace_evidence), "fit_iterations": f.iterations,
                  "fit_theta": [repr(v) for v in f.theta], "evaluable": g.get_is_evaluable()}))
"""


class Gauss(bff.InferenceFactorScore):
    def __init__(self, m, s):
        super().__init__("gauss")
        self.m, self.s = m, s

    def eval(self, x, derivatives):
        r = (x[0] - self.m) / self.s
        lp = -0.5 * r * r - math.log(self.s) - 0.5 * math.log(2 * math.pi)
        return np.array([lp, -r / self.s, 1 / self.s ** 2]) if derivatives else np.array([lp])


def test_the_decay_score_type_is_registered():
    assert "bayesian_decay_experiment" in json.loads(bff.registry_category_json("factor_score"))


def test_saved_with_data_and_loaded_in_another_process():
    d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    experiment(d, OBSERVATIONS)
    g = bff.bayesian_decay_factor_graph(d)
    plain, data = os.path.join(out, "plain.json"), os.path.join(out, "data.bff")
    g.save(plain)
    assert open(plain).read() == g.to_json()
    g.save(data, True)
    assert isinstance(g.to_msgpack(True), bytes)
    theta = np.fromfile(os.path.join(d, "theta.bin"), dtype="<f8").tolist()
    s = g.eval(theta, False)
    f = bff.inference_fitting(g, theta, 300, 1e-6)
    here = {"log_density": repr(s.log_density), "fit_logpost": repr(f.at_mode.log_density),
            "fit_evidence": repr(f.laplace_evidence), "fit_iterations": f.iterations,
            "fit_theta": [repr(v) for v in f.theta], "evaluable": True}
    r = subprocess.run([sys.executable, "-c", REPORT, data, os.path.join(d, "theta.bin")],
                       capture_output=True, text=True, check=True)
    there = json.loads(r.stdout.strip().splitlines()[-1])
    assert there == here


def test_a_python_score_cannot_be_embedded_and_says_which():
    g = bff.InferenceFactorGraph()
    g.add_variable("x", "x", 0)
    g.add_factor("python_prior", bff.INFERENCE_FACTOR_PRIOR, ["x"])
    g.set_factor_score("python_prior", Gauss(0.0, 1.0))
    with pytest.raises(Exception, match="python_prior"):
        g.to_msgpack(True)
    g.to_msgpack(False)                          # the structure alone is fine


def test_a_held_variable_from_python():
    g = bff.InferenceFactorGraph()
    g.add_variable("x", "x", 0)
    g.add_variable("y", "y", 1)
    g.add_factor("px", bff.INFERENCE_FACTOR_PRIOR, ["x"])
    g.add_factor("py", bff.INFERENCE_FACTOR_HYPER, ["y"])
    g.set_factor_score("px", Gauss(0.0, 1.0))
    g.set_factor_score("py", Gauss(5.0, 1.0))
    f = bff.inference_fitting(g, [3.0, 0.25], 300, 1e-10, 1.0, ["y"])
    assert f.theta[1] == 0.25 and abs(f.theta[0]) < 1e-8 and list(f.fixed) == ["y"]
    # evidence over x only: the y factor at y = 0.25 plus x's normaliser
    expect = -0.5 * (0.25 - 5.0) ** 2 - 0.5 * math.log(2 * math.pi)
    assert abs(f.laplace_evidence - expect) < 1e-10

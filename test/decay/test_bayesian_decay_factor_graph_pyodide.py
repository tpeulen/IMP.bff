"""PRD-155 step 8: the decay model's fitting in the Pyodide wheel equals the native one.
Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.

Runs only when `IMPBFF_PYODIDE_WHEEL` names a built Pyodide wheel (tools/pyodide/build_wheel.sh)
and Node with the npm `pyodide` package is available (as for tools/pyodide/smoke_test.mjs); skips
and says why otherwise. The same synthetic experiment (counts drawn from the model at its own
parameters, plus external FRET observations in the docking vocabulary) is fitted by the native wheel here
and by the WebAssembly wheel under Node from the same start. They are separate compilations of
one source, so they agree to rounding, not bit for bit: theta to 1e-6, the log posterior and
each factor to 1e-8 relative, p(R) to 1e-8.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

bff = pytest.importorskip("IMP.bff")
WHEEL = os.environ.get("IMPBFF_PYODIDE_WHEEL")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not WHEEL or not os.path.isfile(WHEEL):
    pytest.skip("IMPBFF_PYODIDE_WHEEL is not set to a built Pyodide wheel", allow_module_level=True)
if not shutil.which("node"):
    pytest.skip("node is not installed", allow_module_level=True)
if not hasattr(bff, "inference_fitting"):
    pytest.skip("the native IMP.bff has no inference_fitting", allow_module_level=True)

from test_bayesian_decay_factor_graph import OBSERVATIONS, experiment


def test_the_wasm_fitting_equals_the_native_fitting():
    d = tempfile.mkdtemp()
    experiment(d, OBSERVATIONS)
    g = bff.bayesian_decay_factor_graph(d)
    f = bff.inference_fitting(g, np.fromfile(f"{d}/theta.bin", dtype="<f8").tolist())
    r = subprocess.run(["node", os.path.join(ROOT, "tools", "pyodide", "bayesian_decay_fitting_check.mjs"), WHEEL, d],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-4000:]
    w = json.loads(next(l for l in r.stdout.splitlines() if l.startswith("RESULT "))[len("RESULT "):])
    print(json.dumps({"wasm_version": w["version"], "iterations": [f.iterations, w["iterations"]],
                      "log_density": [f.at_mode.log_density, w["log_density"]],
                      "theta_maxdiff": float(np.max(np.abs(np.asarray(w["theta"]) - np.asarray(f.theta))))}))
    assert w["converged"] and f.converged
    np.testing.assert_allclose(w["theta"], list(f.theta), rtol=0, atol=1e-6)
    assert list(w["factor_keys"]) == list(f.at_mode.factor_keys)
    for a, b in zip(w["factor_log_densities"], f.at_mode.factor_log_densities):
        assert abs(a - b) <= 1e-8 * max(1.0, abs(b))
    p = bff.bayesian_decay_distribution_at(d, list(f.theta))
    np.testing.assert_allclose(w["p"], list(p), rtol=0, atol=1e-8)

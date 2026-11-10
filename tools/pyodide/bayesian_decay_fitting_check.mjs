// The Bayesian decay model as a scored factor graph, fitted in the Pyodide wheel under Node (PRD-155).
// Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.
//
//   node tools/pyodide/bayesian_decay_fitting_check.mjs <imp_bff wheel> <experiment dir>
//
// Mounts the experiment directory (manifest.json + float64 arrays + theta.bin, as
// test/decay/bayesian_decay_synthetic.py writes it), builds the graph with
// bayesian_decay_factor_graph, fits it with inference_fitting from theta.bin, and
// prints the fit as one JSON line prefixed `RESULT `. A caller compares it with the
// native wheel on the same directory (test/decay/test_bayesian_decay_factor_graph_pyodide.py).
//
// Needs the npm `pyodide` package at the version the wheel was built for, resolved
// from PYODIDE_NODE_DIR (default ~/opt/pyodide-node), as smoke_test.mjs does.
import { createRequire } from "node:module";
import { homedir } from "node:os";
import path from "node:path";

const [wheel, experiment] = process.argv.slice(2);
if (!wheel || !experiment) {
  console.error("usage: node bayesian_decay_fitting_check.mjs <imp_bff wheel> <experiment dir>");
  process.exit(2);
}
const nodeDir = process.env.PYODIDE_NODE_DIR || path.join(homedir(), "opt", "pyodide-node");
const require = createRequire(path.join(nodeDir, "package.json"));
const { loadPyodide } = require("pyodide");

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy"]);
await pyodide.loadPackage(path.resolve(wheel));
pyodide.FS.mkdirTree("/exp");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: path.resolve(experiment) }, "/exp");

const script = String.raw`
import json
import numpy as np
import IMP.bff as bff
g = bff.bayesian_decay_factor_graph("/exp")
f = bff.inference_fitting(g, np.fromfile("/exp/theta.bin", dtype="<f8").tolist())
s = f.at_mode
counts = g.get_factor_score("counts")
m = json.load(open("/exp/manifest.json"))
mask = np.fromfile("/exp/" + m["arrays"]["mask"]["file"], dtype="<f8").tolist()
q = bff.poisson_goodness_of_fit(list(counts.get_data()), list(counts.predict(g.get_factor_values("counts", list(f.theta)))),
                                mask, len(m["data_keys"]), float(len(f.theta)))
out = dict(version=bff.get_module_version(), converged=bool(f.converged), iterations=f.iterations,
           theta=list(f.theta), log_density=s.log_density, log_likelihood=s.log_likelihood,
           log_prior=s.log_prior, laplace_evidence=f.laplace_evidence,
           factor_keys=list(s.factor_keys), factor_log_densities=list(s.factor_log_densities),
           p=list(bff.bayesian_decay_distribution_at("/exp", list(f.theta))),
           deviance_per_dof=q.deviance_per_dof, z=[h.z for h in q.histograms], runs_p=[h.runs_p for h in q.histograms])
"RESULT " + json.dumps(out)
`;
console.log(await pyodide.runPythonAsync(script));

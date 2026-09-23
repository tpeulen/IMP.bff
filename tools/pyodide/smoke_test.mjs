// Smoke test for the Pyodide wheel, run under Node.
//
//   node tools/pyodide/smoke_test.mjs <imp_bff wheel> [tttrlib wheel] [chisurf checkout]
//
// Needs the npm `pyodide` package at the version the wheel was built for,
// resolved from PYODIDE_NODE_DIR (default ~/opt/pyodide-node, a directory with
// `npm install pyodide@0.28.0` done in it) so node_modules stays out of the
// checkout. numpy (and chisurf's pure-Python dependencies) come from the Pyodide
// CDN, so the first run needs network.
//
// It loads what the ndXplorer page loads -- numpy, the tttrlib wheel, this
// wheel -- imports IMP and IMP.bff, and runs tools/pyodide/port_checks.py: the
// GraphPort calls chisurf.core.parameter makes. With a chisurf checkout (default
// ~/dev/chisurf, plus ~/dev/mmfdb and ~/dev/chimol beside it, which chisurf's
// core imports) the same round trip also runs through chisurf's
// FittingParameter, and through ndXplorer's constants group and curve fit
// (chisurf's modules/ndxplorer).
// The report is printed as JSON; compare it with
// `python tools/pyodide/port_checks.py` on a native build.
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dev = path.join(homedir(), "dev");
const [
  wheel,
  tttrlibWheel = path.join(dev, "worktrees", "tttrlib-pyodide", "dist", "pyodide",
                           "tttrlib-0.27.0-cp313-cp313-pyodide_2025_0_wasm32.whl"),
  chisurfDir = path.join(dev, "chisurf"),
] = process.argv.slice(2);
if (!wheel) {
  console.error("usage: node smoke_test.mjs <imp_bff wheel> [tttrlib wheel] [chisurf checkout]");
  process.exit(2);
}
const nodeDir = process.env.PYODIDE_NODE_DIR || path.join(homedir(), "opt", "pyodide-node");
const require = createRequire(path.join(nodeDir, "package.json"));
const { loadPyodide } = require("pyodide");

const pyodide = await loadPyodide();
console.log(`pyodide ${pyodide.version}`);
// what the ndXplorer page loads from the distribution (ndxplorer/app/web.py)
await pyodide.loadPackage(["numpy", "scipy", "pyyaml"]);
// chisurf.core.fio imports lzma, which Pyodide ships unvendored from the stdlib
await pyodide.loadPackage("lzma");
// under Node, loadPackage reads a local path straight from disk
await pyodide.loadPackage(path.resolve(tttrlibWheel));
await pyodide.loadPackage(path.resolve(wheel));

pyodide.FS.writeFile("/tmp/port_checks.py", readFileSync(path.join(here, "port_checks.py")));
let withChisurf = false;
const mmfdbDir = path.join(path.dirname(chisurfDir), "mmfdb");
// chisurf.core's structure and trajectory readers import chimol at module
// level, and chisurf.core.fitting reaches them, so the curve fit needs it too
const chimolDir = path.join(path.dirname(chisurfDir), "chimol");
if (existsSync(path.join(chisurfDir, "chisurf", "core", "parameter.py")) && existsSync(mmfdbDir)
    && existsSync(chimolDir)) {
  // read-only in spirit: nothing below writes into the checkouts
  for (const [dir, at] of [[chisurfDir, "/src/chisurf"], [mmfdbDir, "/src/mmfdb"],
                           [chimolDir, "/src/chimol"]]) {
    pyodide.FS.mkdirTree(at);
    pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: path.resolve(dir) }, at);
  }
  withChisurf = true;
}

const script = String.raw`
import json, sys
sys.path[:0] = ["/tmp", "/src/chisurf", "/src/chisurf/modules/ndxplorer", "/src/mmfdb", "/src/chimol"] if ${withChisurf ? "True" : "False"} else ["/tmp"]
import numpy as np, tttrlib
import IMP, IMP.bff
print("numpy", np.__version__, "tttrlib", tttrlib.__version__, "IMP.bff", IMP.bff.get_module_version(), IMP.bff.get_build())
import port_checks
rep = port_checks.run(with_chisurf=${withChisurf ? "True" : "False"})
print(json.dumps(rep, indent=1, sort_keys=True))
for k in ("port_error", "fitting_parameter_error", "ndxplorer_error"):
    if k in rep:
        print(rep[k])
rep["ok"]
`;
const ok = await pyodide.runPythonAsync(script);
process.exit(ok ? 0 : 1);

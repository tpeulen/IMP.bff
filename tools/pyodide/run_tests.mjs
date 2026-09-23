// Run the Python test lane (or part of it) against the Pyodide wheel, under Node.
//
//   node tools/pyodide/run_tests.mjs <imp_bff wheel> [pytest args...]
//   node tools/pyodide/run_tests.mjs dist/pyodide/imp_bff-*.whl test/graph test/session
//
// The checkout is mounted at /src/imp.bff (NODEFS) and pytest runs there, so
// test/conftest.py deselects the connection layer's tests exactly as it does
// for the native standalone build. pytest, numpy and scipy come from the
// Pyodide distribution (network on the first run). Compare the summary with
// `python -m pytest <same args>` against a native standalone build.
import { createRequire } from "node:module";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const [wheel, ...args] = process.argv.slice(2);
if (!wheel) {
  console.error("usage: node run_tests.mjs <imp_bff wheel> [pytest args...]");
  process.exit(2);
}
const nodeDir = process.env.PYODIDE_NODE_DIR || path.join(homedir(), "opt", "pyodide-node");
const require = createRequire(path.join(nodeDir, "package.json"));
const { loadPyodide } = require("pyodide");

const pyodide = await loadPyodide();
console.log(`pyodide ${pyodide.version}`);
await pyodide.loadPackage(["numpy", "scipy", "pytest", "lzma"]);
await pyodide.loadPackage(path.resolve(wheel));

pyodide.FS.mkdirTree("/src/imp.bff");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root }, "/src/imp.bff");
pyodide.globals.set("pytest_args", pyodide.toPy(args.length ? args : ["test"]));

const rc = await pyodide.runPythonAsync(String.raw`
import os, sys
os.chdir("/src/imp.bff")
# the example structures are not in the wheel; read them from the checkout
os.environ.setdefault("IMP_BFF_EXAMPLES", "/src/imp.bff/examples")
import pytest
# no cache directory in the checkout; the native lane's addopts name plugins
# that the Pyodide distribution does not carry
pytest.main(["-q", "-p", "no:cacheprovider", "-o", "addopts=", *pytest_args])
`);
process.exit(Number(rc));

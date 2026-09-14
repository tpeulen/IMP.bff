"""Compile and run a C++ driver against bff's Bayesian decay headers.

PRD-142's tests exercise header-only C++ directly, without the compiled module:
the headers are what the analysis runs on, and a driver compiled here is the
code a caller compiles. Written 2026-09-14.

The include path is the one a caller uses: bff's `include/` under an
`IMP/bff` shim (as `expensive_test_standalone_core_compiles.py` builds it),
`standalone/include/` for the module config, and tttrlib's `thirdparty/` for
pocketfft -- bff reaches its FFT through tttrlib. `TTTRLIB_THIRDPARTY` names that
directory; otherwise the sibling checkout `../tttrlib/thirdparty` is used. A
test skips, and says why, when no compiler or no pocketfft is found.
"""

import json
import os
import shutil
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _thirdparty():
    for c in (os.environ.get("TTTRLIB_THIRDPARTY"), os.path.join(os.path.dirname(ROOT), "tttrlib", "thirdparty")):
        if c and os.path.isfile(os.path.join(c, "pocketfft", "pocketfft_hdronly.h")):
            return c
    return None


def run_driver(source, args=(), timeout=600):
    """Compile `source` (a C++ translation unit) with -O2 and run it; its stdout
    must be one JSON object, which is returned."""
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    third = _thirdparty()
    if not cxx:
        pytest.skip("no C++ compiler")
    if not third:
        pytest.skip("pocketfft not found: set TTTRLIB_THIRDPARTY or check out tttrlib beside imp.bff")
    tmp = tempfile.mkdtemp()
    try:
        shim = os.path.join(tmp, "inc", "IMP")
        os.makedirs(shim)
        os.symlink(os.path.join(ROOT, "include"), os.path.join(shim, "bff"))
        src = os.path.join(tmp, "driver.cpp")
        exe = os.path.join(tmp, "driver")
        with open(src, "w") as fh:
            fh.write(source)
        cmd = [cxx, "-std=c++17", "-O2", "-DIMPBFF_STANDALONE", "-I", os.path.join(tmp, "inc"),
               "-I", os.path.join(ROOT, "standalone", "include"), "-I", third, "-o", exe, src, "-lpthread"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert r.returncode == 0, "\n".join(l for l in r.stderr.splitlines() if "error" in l)[-4000:]
        r = subprocess.run([exe, *map(str, args)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        assert r.returncode == 0, r.stderr[-4000:]
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

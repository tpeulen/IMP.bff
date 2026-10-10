"""New numerical drivers compile without the optional IMP connection layer."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source", ["fit/Minimize.cpp", "util/Odeint.cpp", "util/SparseLinearAlgebra.cpp"])
def test_numerical_driver_compiles_without_imp(source, tmp_path):
    """Compile the real driver with standalone shims and no full IMP headers."""
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    eigen = Path(sys.prefix) / "include" / "eigen3"
    if source == "util/SparseLinearAlgebra.cpp" and not (eigen / "Eigen" / "SparseLU").is_file():
        pytest.skip("Eigen headers are not installed")
    includes = tmp_path / "include" / "IMP"
    includes.mkdir(parents=True)
    shutil.copytree(ROOT / "include", includes / "bff")
    command = [
        compiler, "-std=c++17", "-fsyntax-only", "-DIMPBFF_STANDALONE",
        "-DIMPBFF_COMPILATION", "-I", str(ROOT / "standalone" / "include"),
        "-I", str(tmp_path / "include"),
    ]
    if source == "util/SparseLinearAlgebra.cpp":
        command += ["-isystem", str(eigen)]
    command += [str(ROOT / "src" / source)]
    # A user's injected include path must not accidentally supply IMP and hide
    # the very packaging failure this independent build is meant to catch.
    env = dict(os.environ)
    for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        env.pop(name, None)
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr

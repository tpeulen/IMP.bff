"""The BFF SWIG hook survives regeneration and stays within its directory."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/swig_threads.py"


def _helper():
    spec = importlib.util.spec_from_file_location("swig_threads", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forwarding_preserves_configured_executable_and_arguments():
    real = "/configured path/swig special"
    arguments = ["-python", "-c++", "-I/path with spaces", "input with spaces.i"]
    assert _helper().swig_command(real, arguments) == [real, "-threads", *arguments]


def test_version_mode_does_not_add_generation_flags():
    assert _helper().swig_command("configured-swig", ["-version"]) == [
        "configured-swig", "-version",
    ]


def test_generation_deduplicates_threads_without_mutating_arguments():
    arguments = ["-threads", "-python", "-threads", "input.i"]
    assert _helper().swig_command("configured-swig", arguments) == [
        "configured-swig", "-threads", "-python", "input.i",
    ]
    assert arguments == ["-threads", "-python", "-threads", "input.i"]


def _run(args, cwd):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher execution; Windows metadata tested below")
def test_cmake_regeneration_scopes_hook_and_registers_dependencies(tmp_path):
    cmake = shutil.which("cmake")
    assert cmake, "CMake is required"
    source = tmp_path / "source with spaces"
    source.mkdir()
    build = tmp_path / "build with spaces"
    record = tmp_path / "forwarded.json"
    real = source / "configured swig"
    real.write_text("#!" + sys.executable + "\nimport json, sys\n"
                    "from pathlib import Path\n"
                    "Path(" + repr(str(record)) + ").write_text(json.dumps(sys.argv[1:]))\n"
                    "sys.exit(37 if '--fail' in sys.argv else 0)\n")
    real.chmod(0o755)
    # Exercise BFF's tracked top-level entry, replacing only the parent IMP
    # module build with an isolated command consumer of the same variables.
    module = source / "bff"
    module.mkdir()
    (module / "CMakeLists.txt").write_text((ROOT / "CMakeLists.txt").read_text())
    (module / "cmake").symlink_to(ROOT / "cmake", target_is_directory=True)
    (module / "tools").symlink_to(ROOT / "tools", target_is_directory=True)
    (module / "ModuleBuild.cmake").write_text("# Parent IMP creates pyext later, as a sibling directory.\n")
    (module / "pyext").mkdir()
    other = source / "other/pyext"
    other.mkdir(parents=True)
    (other / "CMakeLists.txt").write_text(
        'file(WRITE "${CMAKE_BINARY_DIR}/other.txt" "${SWIG_EXECUTABLE}")\n'
    )
    generated_module = '''
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/wrapped"
  COMMAND "${SWIG_EXECUTABLE}" -python "-I/include with spaces" "input with spaces.i"
  COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/wrapped"
  DEPENDS ${IMP_bff_PYTHON_EXTRA_DEPENDENCIES} VERBATIM)
add_custom_target(wrapper ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/wrapped")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/observed.txt" "${SWIG_EXECUTABLE}\\n${IMP_bff_PYTHON_EXTRA_DEPENDENCIES}\\n")
'''
    (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.16)
project(swig_hook NONE)
set(SWIG_EXECUTABLE "{real.as_posix()}" CACHE FILEPATH "configured SWIG")
set(PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}")
set(CMAKE_MODULE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}")
add_subdirectory(bff)
add_subdirectory(bff/pyext bff-python)
add_subdirectory(other/pyext other-python)
file(WRITE "${{CMAKE_BINARY_DIR}}/parent.txt" "${{SWIG_EXECUTABLE}}")
''')
    for _ in range(2):
        # Regenerate the ignored module file just as setup_cmake.py does.
        (module / "pyext/CMakeLists.txt").write_text(generated_module)
        _run([cmake, "-S", str(source), "-B", str(build)], source)
        observed = (build / "bff-python/observed.txt").read_text().splitlines()
        launcher = Path(observed[0])
        assert launcher != real, "BFF needs its own launcher after regeneration"
        assert (build / "parent.txt").read_text() == str(real)
        assert (build / "other.txt").read_text() == str(real)
        assert "SWIG_EXECUTABLE:FILEPATH=" + str(real) in (build / "CMakeCache.txt").read_text()
        dependencies = observed[1].split(";")
        assert str(HELPER) in dependencies
        assert str(launcher) in dependencies
        assert any(path.endswith("SwigThreads.cmake") for path in dependencies)
        _run([str(launcher), "-version"], source)
        assert json.loads(record.read_text()) == ["-version"]
        _run([str(launcher), "-python", "-threads", "-I/path with spaces", "input with spaces.i"], source)
        assert json.loads(record.read_text()) == ["-threads", "-python", "-I/path with spaces", "input with spaces.i"]
        failure = subprocess.run([str(launcher), "-python", "--fail"], cwd=source,
                                 capture_output=True, text=True, timeout=60)
        assert failure.returncode == 37
    _run([cmake, "--build", str(build)], source)
    assert json.loads(record.read_text()) == ["-threads", "-python", "-I/include with spaces", "input with spaces.i"]
    # Check the emitted build rule, not only the variable seen by the module.
    rule = (build / "bff-python/CMakeFiles/wrapper.dir/build.make").read_text()
    for dependency in ("swig_threads.py", "swig-threads", "SwigThreads.cmake"):
        assert dependency in rule


def test_windows_launcher_uses_python_and_forwards_argv(tmp_path):
    # Generate Windows launcher metadata on every host. Runtime execution
    # of cmd.exe is left to Windows CI.
    source = tmp_path / "windows source"
    source.mkdir()
    real = source / "configured swig.exe"
    (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.16)
project(windows_launcher NONE)
set(WIN32 TRUE)
set(SWIG_EXECUTABLE "{real.as_posix()}")
set(PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}")
include("{ROOT.as_posix()}/cmake/SwigThreads.cmake")
imp_bff_enable_swig_threads()
file(WRITE "${{CMAKE_BINARY_DIR}}/launcher.txt" "${{SWIG_EXECUTABLE}}")
''')
    build = tmp_path / "windows build"
    _run([shutil.which("cmake"), "-S", str(source), "-B", str(build)], source)
    launcher = Path((build / "launcher.txt").read_text())
    assert launcher.suffix == ".cmd"
    content = launcher.read_text()
    normalized = content.replace("\\", "/")
    assert '"' + Path(sys.executable).as_posix() + '"' in normalized
    assert '"' + HELPER.as_posix() + '"' in normalized
    assert '"' + real.as_posix() + '"' in normalized
    assert "%*" in content
    assert "exit /b %errorlevel%" in content.lower()

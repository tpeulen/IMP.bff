"""Compiler selection and argv coverage, including MSVC on every host."""

from pathlib import Path

import pytest
import test_swig_thread_boundary as boundary


@pytest.mark.parametrize("platform,env_cxx,config_cxx,expected", [
    ("linux", "ccache clang++ -O2", "g++", ["ccache", "clang++", "-O2"]),
    ("linux", None, "clang++ -O2", ["clang++", "-O2"]),
    ("linux", None, None, ["c++"]),
    ("linux", None, "missing-g++", ["c++"]),
    ("win32", None, None, ["cl"]),
    ("win32", None, "missing-cl", ["cl"]),
    ("win32", None, "cl /nologo", ["cl", "/nologo"]),
    ("win32", r'"C:\Program Files\MSVC\cl.exe" /nologo', None,
     [r"C:\Program Files\MSVC\cl.exe", "/nologo"]),
])
def test_compiler_selection(monkeypatch, platform, env_cxx, config_cxx, expected):
    monkeypatch.setattr(boundary.sys, "platform", platform)
    monkeypatch.delenv("CXX", raising=False)
    if env_cxx is not None:
        monkeypatch.setenv("CXX", env_cxx)
    monkeypatch.setattr(boundary.sysconfig, "get_config_var", lambda name: config_cxx)
    monkeypatch.setattr(boundary.shutil, "which", lambda name: None if name.startswith("missing-") else name)
    assert boundary._compiler() == expected


def test_missing_compiler_is_an_explicit_failure(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(boundary.sysconfig, "get_config_var", lambda name: None)
    monkeypatch.setattr(boundary.shutil, "which", lambda name: None)
    with pytest.raises(AssertionError, match="C\\+\\+ compiler required"):
        boundary._compiler()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("standalone", [False, True])
def test_unix_compile_command(monkeypatch, tmp_path, platform, standalone):
    monkeypatch.setattr(boundary.sys, "platform", platform)
    monkeypatch.setattr(boundary.sysconfig, "get_path", lambda name: "/python include")
    cpp, extension = tmp_path / "wrap.cpp", tmp_path / "_boundary.so"
    compiler = ["ccache", "clang++", "-O2"]
    flags = ["-std=c++14", "-shared", "-fPIC", "-I/python include"]
    if platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    if standalone:
        flags += ["-DIMPBFF_STANDALONE=1"]
    assert boundary._compile_command(compiler, cpp, extension, standalone) == [
        *compiler, *flags, str(cpp), "-o", str(extension),
    ]
    assert compiler == ["ccache", "clang++", "-O2"]


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("compiler", [
    ["cl"], [r"C:\Program Files\MSVC\CL.EXE"], ["sccache", "clang-cl.exe"],
])
@pytest.mark.parametrize("configured_library", [None, "python313.lib"])
def test_msvc_compile_command(monkeypatch, tmp_path, standalone, compiler, configured_library):
    monkeypatch.setattr(boundary.sys, "platform", "win32")
    monkeypatch.setattr(boundary.sys, "version_info", (3, 13, 0))
    prefix, base = tmp_path / "test env", tmp_path / "base python"
    monkeypatch.setattr(boundary.sys, "prefix", str(prefix))
    monkeypatch.setattr(boundary.sys, "base_prefix", str(base))
    config = {"LIBRARY": configured_library, "LIBDIR": str(base / "libs")}
    monkeypatch.setattr(boundary.sysconfig, "get_config_var", config.get)
    include = str(base / "include")
    monkeypatch.setattr(boundary.sysconfig, "get_path", lambda name: include)
    cpp, extension = tmp_path / "wrap.cpp", tmp_path / "_boundary.cp313-win_amd64.pyd"
    command = boundary._compile_command(compiler, cpp, extension, standalone)
    flags = ["/nologo", "/std:c++14", "/LD", "/MD", "/EHsc", "/I" + include]
    if standalone:
        flags += ["/DIMPBFF_STANDALONE=1"]
    assert command == [
        *compiler, *flags, str(cpp), "/Fe" + str(extension), "/link",
        "/LIBPATH:" + str(base / "libs"), "/LIBPATH:" + str(prefix / "libs"),
        "python313.lib",
    ]
    assert all(flag not in command for flag in ("-shared", "-fPIC", "-std=c++14", "-o"))


def test_msvc_configured_import_library_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(boundary.sys, "platform", "win32")
    monkeypatch.setattr(boundary.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(boundary.sys, "base_prefix", str(tmp_path))
    config = {"LIBRARY": "python313_custom.lib", "LIBPL": str(tmp_path / "import libs")}
    monkeypatch.setattr(boundary.sysconfig, "get_config_var", config.get)
    monkeypatch.setattr(boundary.sysconfig, "get_path", lambda name: str(tmp_path / "include"))
    command = boundary._compile_command(["cl"], Path("wrap.cpp"), Path("_boundary.pyd"), False)
    assert command[command.index("/link") + 1:] == [
        "/LIBPATH:" + str(tmp_path / "import libs"), "/LIBPATH:" + str(tmp_path / "libs"),
        "python313_custom.lib",
    ]

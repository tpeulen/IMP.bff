"""Compile a tiny director using the shared core's actual thread policy.

The standalone lane takes its flags from the tracked UseSWIG source property.
No full standalone numerical build is needed to exercise the GIL boundary.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _compiler():
    defaults = ["cl", "clang-cl"] if sys.platform == "win32" else ["c++", "clang++", "g++"]
    for configured in [os.environ.get("CXX"), sysconfig.get_config_var("CXX"), *defaults]:
        if not configured:
            continue
        # POSIX shlex consumes backslashes in Windows compiler paths.
        command = shlex.split(configured, posix=sys.platform != "win32")
        if sys.platform == "win32":
            command = [argument.strip('"') for argument in command]
        if command and shutil.which(command[0]):
            return command
    raise AssertionError("C++ compiler required (set CXX or install a native compiler)")


def _compile_command(compiler, cpp, extension, standalone):
    msvc = any(Path(arg.replace("\\", "/")).name.lower() in
               ("cl", "cl.exe", "clang-cl", "clang-cl.exe") for arg in compiler)
    if msvc:
        debug = bool(sysconfig.get_config_var("Py_DEBUG"))
        flags = ["/nologo", "/std:c++14", "/LD", "/MDd" if debug else "/MD", "/EHsc",
                 "/I" + sysconfig.get_path("include")]
        if standalone:
            flags += ["/DIMPBFF_STANDALONE=1"]
        # Virtual environments need the base interpreter's import library too.
        directories = [sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LIBPL"),
                       str(Path(sys.prefix) / "libs"), str(Path(sys.base_prefix) / "libs")]
        library_flags = ["/LIBPATH:" + directory for directory in dict.fromkeys(directories)
                         if directory]
        abi = ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "") + ("_d" if debug else "")
        library = sysconfig.get_config_var("LIBRARY") or (
            f"python{sys.version_info[0]}{sys.version_info[1]}{abi}.lib")
        return [*compiler, *flags, str(cpp), "/Fe" + str(extension), "/link",
                *library_flags, library]
    flags = ["-std=c++14", "-shared", "-fPIC", "-I" + sysconfig.get_path("include")]
    if sys.platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    if standalone:
        flags += ["-DIMPBFF_STANDALONE=1"]
    return [*compiler, *flags, str(cpp), "-o", str(extension)]


def _run(args, directory):
    result = subprocess.run(args, cwd=directory, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("standalone", [False, True], ids=["imp", "standalone"])
def test_shared_directors_and_exception_boundary(tmp_path, standalone):
    swig = shutil.which("swig")
    assert swig, "SWIG required"
    compiler = _compiler()
    core = (ROOT / "pyext/include/IMP_bff.core.i").read_text()
    policy = core[core.index('/* The SWIG launcher'):core.index('/* Does this build')]
    exception = re.search(
        r'%exception IMP::bff::ModelSearch::run \{.*?\n\}', core, re.DOTALL
    ).group()
    macros = (ROOT / "standalone/pyext/IMP_bff_standalone.macros.i").read_text()
    # Use the established standalone exception hierarchy and translator too.
    translation = macros[macros.index('%{\nstatic PyObject'):macros.index('/* ---- show()')]
    translation = translation.replace("_IMP_bff", "_boundary")
    preamble = r'''
%module(directors="1") boundary
%include <std_string.i>
%{
#include <stdexcept>
#include <sstream>
namespace IMP {
struct Exception : std::runtime_error { using std::runtime_error::runtime_error; };
struct ValueException : Exception { using Exception::Exception; };
struct ModelException : Exception { using Exception::Exception; };
struct UsageException : Exception { using Exception::Exception; };
struct IOException : Exception { using Exception::Exception; };
}
%}
'''
    # The IMP lane uses its existing handler; only the small fixture aliases
    # that name to the existing standalone mapping to avoid linking IMP.
    alias = '' if standalone else '%{\nstatic void handle_imp_exception() { imp_bff_handle_exception(); }\n%}\n'
    declarations = r'''
%feature("director") IMP::bff::GraphNode;
%inline %{
namespace IMP { namespace bff {
class GraphNode {
 public:
  virtual ~GraphNode() {}
  virtual int evaluate() { return 1; }
  virtual int do_evaluate() { return 2; }
  virtual std::string get_node_type() const { return "native"; }
  void set_value(int) {}
};
class ModelSearch {
 public:
  int run(GraphNode *node, int mode) {
    if (mode == 1) throw std::domain_error("configuration sentinel");
    if (mode == 2) throw std::runtime_error("native sentinel");
    if (mode == 3) throw 42;
    return node->evaluate() + node->do_evaluate()
        + (node->get_node_type() == "python" ? 4 : 0);
  }
};
}}
%}
'''
    entry = tmp_path / "boundary.i"
    entry.write_text(preamble + translation + alias + policy + exception + declarations)
    if standalone:
        cmake = (ROOT / "standalone/CMakeLists.txt").read_text()
        flags = shlex.split(re.search(
            r'PROPERTY COMPILE_OPTIONS ([^)]+)\)', cmake
        ).group(1))
    else:
        flags = ["-threads"]
    cpp = tmp_path / "wrap.cpp"
    _run([swig, "-python", "-c++", *flags, "-o", str(cpp), str(entry)], tmp_path)
    generated = cpp.read_text()
    for method in ("evaluate", "do_evaluate", "get_node_type"):
        body = re.search(r'SwigDirector_GraphNode::' + method + r'\([^)]*\)[^{]*\{(.*?)\n\}',
                         generated, re.DOTALL).group(1)
        assert "SWIG_PYTHON_THREAD_BEGIN_BLOCK" in body, method
    setter = re.search(r'_wrap_GraphNode_set_value\([^\n]+\{(.*?)\n\}', generated, re.DOTALL).group(1)
    assert "SWIG_PYTHON_THREAD_BEGIN_ALLOW" not in setter
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = tmp_path / ("_boundary" + suffix)
    _run(_compile_command(compiler, cpp, extension, standalone), tmp_path)
    script = r'''
import boundary as b
class SentinelError(LookupError): pass
sentinel = SentinelError('director sentinel')
class Node(b.GraphNode):
    def evaluate(self): return 5
    def do_evaluate(self): return 6
    def get_node_type(self): return 'python'
class Broken(Node):
    def evaluate(self): raise sentinel
search = b.ModelSearch()
node = Node()
for mode, error, message in [(1, b.ValueException, 'configuration sentinel'),
                              (2, RuntimeError, 'native sentinel'),
                              (3, RuntimeError, 'Unknown error')]:
    try: search.run(node, mode)
    except error as caught: assert message in str(caught)
    else: raise AssertionError('native exception missing')
    assert search.run(node, 0) == 15
try: search.run(Broken(), 0)
except SentinelError as caught: assert caught is sentinel
else: raise AssertionError('director exception missing')
assert search.run(node, 0) == 15
del node, search
print('native errors, director identity, return and destruction succeeded')
'''
    result = _run([sys.executable, "-X", "faulthandler", "-c", script], tmp_path)
    assert "destruction succeeded" in result.stdout

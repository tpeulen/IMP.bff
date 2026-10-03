"""Compile a tiny director using the shared core's actual thread policy.

The standalone lane takes its flags from the tracked UseSWIG source property.
No full standalone numerical build is needed to exercise the GIL boundary.
"""

from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(args, directory):
    result = subprocess.run(args, cwd=directory, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("standalone", [False, True], ids=["imp", "standalone"])
def test_shared_directors_and_exception_boundary(tmp_path, standalone):
    swig = shutil.which("swig")
    compiler = shlex.split(sysconfig.get_config_var("CXX"))
    assert swig and shutil.which(compiler[0]), "SWIG and a C++ compiler required"
    core = (ROOT / "pyext/include/IMP_bff.core.i").read_text()
    policy = core[core.index('/* The SWIG launcher'):core.index('/* Does this build')]
    exception = re.search(
        r'%exception IMP::bff::ModelSearch::run \{.*?\n\}', core, re.S
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
                         generated, re.S).group(1)
        assert "SWIG_PYTHON_THREAD_BEGIN_BLOCK" in body, method
    setter = re.search(r'_wrap_GraphNode_set_value\([^\n]+\{(.*?)\n\}', generated, re.S).group(1)
    assert "SWIG_PYTHON_THREAD_BEGIN_ALLOW" not in setter
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    flags = ["-std=c++14", "-shared", "-fPIC", "-I" + sysconfig.get_path("include")]
    if sys.platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    if standalone:
        flags += ["-DIMPBFF_STANDALONE=1"]
    _run([*compiler, *flags, str(cpp), "-o", str(tmp_path / ("_boundary" + suffix))], tmp_path)
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

"""A data *directory* resolves through the registry, fetched on first use.

C++ callers ask for whole directories -- `get_data_path("rotamer_library")`,
`get_data_path("cgprobe")` -- while the registry names files. A standalone
install without the large data (a wheel, a source build) used to fail with
"Unable to find data file rotamer_library" because only the Python wrapper
fetched, and only single files. Both lookups now fetch every registry file
under a directory into the cache. No test here touches the network: the fetch
is replaced by one that writes the files.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import IMP.bff

_REGISTRY = {
    "fakedir/a.txt": "sha256:0",
    "fakedir/sub/b.txt": "sha256:1",
    "otherdir/c.txt": "sha256:2",
}

_FAKE_FETCH = textwrap.dedent('''
    import os
    import IMP.bff as bff
    REGISTRY = %r
    fetched = []
    def fake_fetch(names, progressbar=False):
        for n in names:
            p = os.path.join(bff.get_data_cache_dir(), n)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write("x")
            fetched.append(n)
        return names
    bff.get_data_registry = lambda: REGISTRY
    bff.fetch_data = fake_fetch
''') % (_REGISTRY,)


class TestDirectoryFetch(unittest.TestCase):

    def test_the_python_lookup_fetches_every_file_under_a_directory(self):
        with tempfile.TemporaryDirectory() as cache:
            script = _FAKE_FETCH + textwrap.dedent('''
                def missing(name):
                    raise bff.IOException("not installed: " + name)
                bff._installed_get_data_path = missing
                print(bff.get_data_path("fakedir"))
                print(sorted(fetched))
            ''')
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, env=dict(os.environ, IMP_BFF_CACHE=cache))
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            path, names = out.stdout.strip().splitlines()[-2:]
            self.assertEqual(path, os.path.join(cache, "fakedir"))
            self.assertEqual(names, str(["fakedir/a.txt", "fakedir/sub/b.txt"]))
            self.assertTrue(os.path.isfile(os.path.join(cache, "fakedir", "sub", "b.txt")))

    def test_the_cxx_lookup_asks_the_registry_on_a_miss(self):
        if not hasattr(IMP.bff, "set_data_fetcher"):
            self.skipTest("the IMP module build resolves data through IMP's own lookup")
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as empty:
            # The raw C++ get_data_path, as every C++ caller reaches it.
            script = _FAKE_FETCH + textwrap.dedent('''
                print(bff._installed_get_data_path("fakedir"))
                print(sorted(fetched))
                try:
                    bff._installed_get_data_path("not_in_the_registry")
                except bff.IOException:
                    print("miss stays a miss")
            ''')
            env = dict(os.environ, IMP_BFF_CACHE=cache, IMP_BFF_DATA=empty)
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, env=env)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            lines = out.stdout.strip().splitlines()[-3:]
            self.assertEqual(lines[0], os.path.join(cache, "fakedir"))
            self.assertEqual(lines[1], str(["fakedir/a.txt", "fakedir/sub/b.txt"]))
            self.assertEqual(lines[2], "miss stays a miss")


if __name__ == "__main__":
    unittest.main()

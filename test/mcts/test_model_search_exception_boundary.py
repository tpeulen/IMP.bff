"""A native error must return through Python, including on a second run."""

import os
import subprocess
import sys


def test_unconfigured_search_raises_and_can_be_reused(tmp_path):
    # A subprocess keeps an accidental C++ abort out of the pytest process.
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", """
import IMP.bff as b
search = b.ModelSearch()
for _ in range(2):
    try:
        search.run()
    except b.ValueException as error:
        assert isinstance(error, ValueError)
        assert 'model search has no problem' in str(error)
    else:
        raise AssertionError('missing configuration error')
problem = b.TabularModelSearchProblem()
problem.add_state('root', 0.0, True)
problem.set_initial_state('root')
search.set_problem(problem)
result = search.run()
assert result.get_best_state().get_key() == 'root'
print('translated twice; reuse succeeded')
"""],
        cwd=tmp_path, env={**os.environ, "IMP_BFF_GPU": "off"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "translated twice; reuse succeeded" in result.stdout

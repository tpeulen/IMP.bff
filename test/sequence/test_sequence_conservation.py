"""Per-site rates (Rate4Site's method) and ConSurf grades.

A/B against Rate4Site 3.0.0 run as a black box on the same alignment
(`test/input/sequence/`): scores, intervals and posterior standard
deviations agree to the four significant digits Rate4Site prints, and the
ConSurf grades derived from both are identical.
"""

import os
import re

import numpy as np
import pytest

import IMP.bff as bff

DATA = os.path.join(os.path.dirname(__file__), "..", "input", "sequence")


def _read_res(name):
    rows, alpha, ll = [], None, None
    for line in open(os.path.join(DATA, name)):
        if line.startswith("#The alpha"):
            alpha = float(line.split()[-1])
        if line.startswith("#LL="):
            ll = float(line[4:])
        m = re.match(r"^\s*(\d+)\s+(\w)\s+(\S+)\s+\[\s*(\S+),\s*(\S+)\]\s+(\S+)\s+(\d+)/(\d+)", line)
        if m:
            rows.append([float(m.group(k)) for k in (3, 4, 5, 6)] + [int(m.group(7))])
    return np.array(rows), alpha, ll


@pytest.fixture(scope="module")
def msa():
    return bff.read_sequence_msa(os.path.join(DATA, "rbp_60x120.fasta"), 0, False)


@pytest.mark.parametrize("preset,reference", [("consurf", "rate4site_bn.res"),
                                              ("rate4site", "rate4site_bg.res")])
def test_matches_rate4site(msa, preset, reference):
    options = getattr(bff.SequenceConservationOptions, preset)()
    c = bff.compute_sequence_conservation(msa, options)
    ref, alpha, ll = _read_res(reference)
    assert c.get_n_positions() == len(ref)
    assert c.get_alpha() == pytest.approx(alpha, rel=1e-4)
    assert c.get_log_likelihood() == pytest.approx(ll, abs=0.02)
    tol = dict(rtol=1e-3, atol=3e-4)   # Rate4Site prints four significant digits
    np.testing.assert_allclose(c.get_scores(), ref[:, 0], **tol)
    np.testing.assert_allclose(c.get_lower(), ref[:, 1], **tol)
    np.testing.assert_allclose(c.get_upper(), ref[:, 2], **tol)
    np.testing.assert_allclose(c.get_std(), ref[:, 3], **tol)
    assert list(c.get_n_data()) == [int(x) for x in ref[:, 4]]
    ours = np.array(bff.get_consurf_grades(c)).reshape(-1, 4)
    theirs = np.array(bff.get_consurf_grades(list(ref[:, 0]), list(ref[:, 1]), list(ref[:, 2]),
                                             [int(x) for x in ref[:, 4]])).reshape(-1, 4)
    np.testing.assert_array_equal(ours, theirs)


def test_scores_are_normalised_and_conserved_is_low(msa):
    c = bff.compute_sequence_conservation(msa)
    s = np.asarray(c.get_scores())
    assert abs(s.mean()) < 1e-9 and s.std(ddof=1) == pytest.approx(1.0)
    raw = np.asarray(c.get_raw_rates())
    np.testing.assert_allclose((raw - c.get_raw_mean()) / c.get_raw_sd(), s)
    # a column identical in every sequence is among the most conserved
    cols = [msa.get_sequence(m) for m in range(msa.get_n_sequences())]
    uniform = [k for k, col in enumerate(c.get_columns())
               if len({cols[m][col] for m in range(len(cols))}) == 1]
    assert uniform and max(s[k] for k in uniform) < np.percentile(s, 25)


def test_table_has_rate4site_layout(msa):
    c = bff.compute_sequence_conservation(msa)
    table = c.get_rate4site_table()
    assert table.startswith("#Rates were calculated using the expectation")
    assert "#POS SEQ  SCORE    QQ-INTERVAL     STD      MSA DATA" in table
    assert table.count("/60\n") == c.get_n_positions()


def test_fixed_tree_and_alpha(msa):
    c0 = bff.compute_sequence_conservation(msa)
    o = bff.SequenceConservationOptions.consurf()
    o.tree = c0.get_tree()
    o.alpha = c0.get_alpha()
    c1 = bff.compute_sequence_conservation(msa, o)
    np.testing.assert_allclose(c1.get_scores(), c0.get_scores(), atol=1e-5)


def test_ml_rates_rank_like_the_posterior(msa):
    o = bff.SequenceConservationOptions.consurf()
    o.rate_inference = "ml"
    ml = np.asarray(bff.compute_sequence_conservation(msa, o).get_scores())
    eb = np.asarray(bff.compute_sequence_conservation(msa).get_scores())
    assert np.corrcoef(ml, eb)[0, 1] > 0.8


def test_consurf_grade_bins():
    # lowest score -1.8: bin width 0.4, bins from -1.8 up to +1.8
    scores = [-1.8, -1.41, 0.0, 1.79, 1.8, 3.0]
    g = np.array(bff.get_consurf_grades(scores, scores, scores, [10] * 6)).reshape(-1, 4)
    assert list(g[:, 0]) == [9, 9, 5, 1, 1, 1]
    # a wide interval, or five sequences or fewer, is below the cut-off
    g = np.array(bff.get_consurf_grades([-1.8, 0.0, 0.0], [-1.8, -1.8, 0.0], [-1.8, 1.8, 0.0],
                                        [10, 10, 5])).reshape(-1, 4)
    assert list(g[:, 3]) == [0, 1, 1]


def test_bad_inputs(msa):
    o = bff.SequenceConservationOptions()
    o.branch_lengths = "free"
    with pytest.raises(bff.ValueException):
        bff.compute_sequence_conservation(msa, o)
    with pytest.raises(bff.ValueException):
        bff.compute_sequence_conservation(bff.SequenceMSA(["a", "b"], ["AC", "AC"]))

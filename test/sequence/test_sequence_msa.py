"""SequenceMSA: reading, encoding, reference columns, weights, variety."""

import numpy as np
import pytest

import IMP.bff as bff


def _msa(seqs, **kw):
    return bff.SequenceMSA([f"s{k}" for k in range(len(seqs))], seqs, **kw)


def test_alphabet_and_encoding():
    assert bff.get_sequence_alphabet() == "ACDEFGHIKLMNPQRSTVWY"
    msa = _msa(["AC-Y", "XB.W"], match_columns_only=False)
    assert [msa.get_state(0, k) for k in range(4)] == [1, 2, 0, 20]
    # X and B are not amino acids of the alphabet: they read as gaps
    assert [msa.get_state(1, k) for k in range(4)] == [0, 0, 0, 19]
    assert msa.get_sequence(1) == "---W"


def test_match_columns_and_reference_positions():
    # the reference's lower-case and '.' columns are insert states and are dropped;
    # its gap columns are kept, with no residue number
    msa = _msa(["AC-d.E", "ACGDKE"])
    assert msa.get_n_columns() == 4
    assert list(msa.get_reference_positions()) == [1, 2, -1, 4]
    assert msa.get_sequence(1) == "ACGE"


def test_reader_handles_wrapped_lines(tmp_path):
    p = tmp_path / "aln.fasta"
    p.write_text(">one first\nACDE\nFGHI\n\n>two\nACDEF\nGHI\n")
    msa = bff.read_sequence_msa(str(p))
    assert msa.get_n_sequences() == 2 and msa.get_n_columns() == 8
    assert list(msa.get_names()) == ["one first", "two"]
    bad = tmp_path / "bad.fasta"
    bad.write_text(">a\nACD\n>b\nAC\n")
    with pytest.raises(bff.ValueException):
        bff.read_sequence_msa(str(bad))


def test_weights_share_among_near_identical_sequences():
    seqs = ["ACDEFGHIKL", "ACDEFGHIKL", "ACDEFGHIKM", "WWWWWWWWWW"]
    msa = _msa(seqs)
    w = np.asarray(bff.get_sequence_weights(msa, 0.2))
    # the first three are within 20 % of each other, the last is alone
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3, 1.0])
    np.testing.assert_allclose(bff.get_sequence_weights(msa, 0.0), [1.0] * 4)
    # the threshold is strict: one difference in ten is 0.1, not < 0.1
    np.testing.assert_allclose(bff.get_sequence_weights(msa, 0.1), [1 / 2, 1 / 2, 1.0, 1.0])


def test_variety_and_data_counts():
    msa = _msa(["AC", "WC", "-C", "AC"])
    assert bff.get_residue_variety(msa, 0) == "AW"
    assert bff.get_n_with_data(msa, 0) == 3
    assert bff.get_residue_variety(msa, 1) == "C"
    assert bff.get_n_with_data(msa, 1) == 4

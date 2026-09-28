"""Conservation for the Labelizer: every ConSurf field, from a grade file or
computed natively from an alignment.

The native path runs the Rate4Site method as ConSurf does and grades it with
ConSurf's rules (A/B-tested against both programs in test/sequence); here it
is exercised end to end on MalE with a synthetic alignment built from the
structure's own sequence.
"""

import os

import numpy as np
import pytest

import IMP.bff as bff

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "input", "labelizer")

GRADES = """\t Amino Acid Conservation Scores
\t===============================

- POS: The position of the AA in the SEQRES derived sequence.
- SEQ: The SEQRES derived sequence in one letter code.
- 3LATOM: The ATOM derived sequence in three letter code, including the AA's positions as they appear in the PDB file and the chain identifier.
- SCORE: The normalized conservation scores.
- COLOR: The color scale representing the conservation scores (9 - conserved, 1 - variable).
- CONFIDENCE INTERVAL: When using the bayesian method for calculating rates, a confidence interval is assigned to each of the inferred evolutionary conservation scores.
- CONFIDENCE INTERVAL COLORS: When using the bayesian method for calculating rates. The color scale representing the lower and upper bounds of the confidence interval.
- MSA DATA: The number of aligned sequences having an amino acid (non-gapped) from the overall number of sequences at each position.
- RESIDUE VARIETY: The residues variety at each position of the multiple sequence alignment.

 POS\t SEQ\t    3LATOM\tSCORE\t\tCOLOR\tCONFIDENCE INTERVAL\tCONFIDENCE INTERVAL COLORS\tMSA DATA\tRESIDUE VARIETY
    \t    \t        \t(normalized)\t        \t               
   1\t   M\t         -\t 0.500\t\t  3\t 0.100, 0.900\t\t\t    4,2\t\t\t   10/12\tM,L
   2\t   K\t    LYS2:A\t-0.612\t\t  9\t-0.796,-0.530\t\t\t    9,8\t\t\t   12/12\tK
   3\t   I\t    ILE3:A\t 0.862\t\t  1*\t-0.096, 1.176\t\t\t    6,1\t\t\t    5/12\tS,C,I
"""


def test_grade_file_records(tmp_path):
    p = tmp_path / "chain.grades"
    p.write_text(GRADES)
    r = dict(bff.labelizer_read_consurf_records(str(p)))
    assert sorted(r) == ["A2", "A3"]          # position 1 is not in the structure
    a2, a3 = r["A2"], r["A3"]
    assert a2.score == pytest.approx(-0.612) and a2.grade == 9 and not a2.insufficient
    assert (a2.lower, a2.upper) == (pytest.approx(-0.796), pytest.approx(-0.530))
    assert (a2.lower_grade, a2.upper_grade) == (9, 8)
    assert (a3.grade, a3.insufficient, a3.n_data, a3.n_sequences) == (1, True, 5, 12)
    assert a3.variety == "SCI"
    # the score-only reader agrees on the score
    assert bff.labelizer_read_consurf(str(p))["A2"] == pytest.approx(-0.612)


def _synthetic_alignment(sequence, path, n=40, seed=3):
    """The reference, and mutants: the first third of the positions never
    change, the rest change often."""
    rng = np.random.default_rng(seed)
    alphabet = bff.get_sequence_alphabet()
    rows = [sequence]
    third = len(sequence) // 3
    for _ in range(n - 1):
        s = list(sequence)
        for k in range(third, len(s)):
            if rng.random() < 0.35:
                s[k] = alphabet[rng.integers(0, 20)]
        rows.append("".join(s))
    path.write_text("".join(f">s{k}\n{r}\n" for k, r in enumerate(rows)))
    return third


@pytest.fixture(scope="module")
def male(tmp_path_factory):
    s = bff.labelizer_read_structure(os.path.join(DATA, "1OMP.pdb"))
    seq = "".join(bff.labelizer_one_letter(r.comp_id) for r in s.residues if r.chain == "A")
    path = tmp_path_factory.mktemp("aln") / "male.fasta"
    third = _synthetic_alignment(seq, path)
    return s, str(path), third


def test_native_records_follow_the_alignment(male):
    s, path, third = male
    rec = dict(bff.labelizer_conservation_from_msa(s, path))
    chain_a = [r for r in s.residues if r.chain == "A"]
    assert len(rec) == len(chain_a)
    conserved = [rec[f"A{r.seq_id}"].score for r in chain_a[:third]]
    variable = [rec[f"A{r.seq_id}"].score for r in chain_a[third:]]
    assert max(conserved) < min(variable)
    assert all(rec[f"A{r.seq_id}"].grade >= 7 for r in chain_a[:third])
    first = rec[f"A{chain_a[0].seq_id}"]
    assert first.variety == bff.labelizer_one_letter(chain_a[0].comp_id)
    assert first.n_sequences == 40


def test_alignment_path_scores_every_conservation_table(male):
    s, path, _ = male
    model = bff.LabelizerParameterList()
    for table in ["N_CS2_Score", "N_CS3_Lower_Score", "N_CS4_Upper_Score", "I_CS1_Color",
                  "I_CS5_Variety_Length", "C_CS6_Cys_In_Variety"]:
        model.append(bff.LabelizerParameter("cs", table, 1))
    rows = bff.labelizer_score_structure(os.path.join(DATA, "1OMP.pdb"), model,
                                         bff.LabelizerOptions(), path)
    cs = [x for x in rows if x.score_type == "conservation"]
    assert len(cs) == 6 * sum(1 for r in s.residues if r.chain == "A")
    assert all(x.status == "scored" for x in cs)


def test_score_only_source_refuses_the_other_tables(tmp_path):
    # a B-factor PDB carries the score only
    pdb = os.path.join(DATA, "1DDB-conservationscore-39-A.pdb")
    model = bff.LabelizerParameterList()
    model.append(bff.LabelizerParameter("cs", "I_CS1_Color", 1))
    with pytest.raises(bff.ValueException):
        bff.labelizer_score_structure(os.path.join(DATA, "1DDB-39.pdb"), model,
                                      bff.LabelizerOptions(), pdb)


def test_no_matching_chain_is_an_error(tmp_path):
    s = bff.labelizer_read_structure(os.path.join(DATA, "1OMP.pdb"))
    p = tmp_path / "other.fasta"
    p.write_text(">a\nWWWWWWWWWWWWWWWWWWWW\n>b\nWWWWWWWWWWWWWWWWWWWY\n>c\nWWWWWWWWWWWWWWWWWWYY\n")
    with pytest.raises(bff.ValueException):
        bff.labelizer_conservation_from_msa(s, str(p))

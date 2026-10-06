"""Model confidence follows the existing core parser and selected structure model."""
from pathlib import Path

import numpy as np
import pytest

import IMP.bff as bff

ROOT = Path(__file__).resolve().parents[2]
T4 = ROOT / "ipynb" / "example" / "148L_consurf_grades.pdb"


def _atom(serial, name="CA", residue=1, factor="75.50", altloc=" ", insertion=" "):
    """Return one fixed-column atom record with a controlled B-factor field."""
    line = (
        f"ATOM  {serial:5d} {name:^4s}{altloc}ALA A{residue:4d}{insertion}   "
        f"{float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}"
    )
    if factor is not None:
        line += f"{factor:>6}          C "
    return line + "\n"


@pytest.mark.parametrize("factor,expected", [("73.25", 73.25), ("", 0.0),
                                             (None, 0.0), ("BAD!!!", 0.0)])
def test_bfactor_parsing_does_not_discard_atoms(factor, expected, tmp_path):
    """A missing or unparseable optional factor keeps the atom and defaults to zero."""
    path = tmp_path / "factor.pdb"
    path.write_text(_atom(1, factor=factor), encoding="utf-8")
    records = bff.read_pdb_records(str(path))
    assert len(records) == 1
    assert records[0].bfactor == expected
    structure = bff.labelizer_read_structure(str(path))
    assert list(structure.bfactor) == [expected]
    assert list(structure.xyz) == [1.0, 0.0, 0.0]


def test_record_inventory_and_identifiers_are_unchanged(tmp_path):
    """Reading factors does not introduce new alternate-location or insertion filtering."""
    path = tmp_path / "identifiers.pdb"
    path.write_text(_atom(1, factor="70.00", altloc="A")
                    + _atom(2, factor="80.00", altloc="B")
                    + _atom(3, factor="90.00", insertion="A"), encoding="utf-8")
    records = bff.read_pdb_records(str(path))
    assert [record.serial for record in records] == [1, 2, 3]
    assert [record.atom_name for record in records] == ["CA", "CA", "CA"]
    assert [(record.chain, record.resseq) for record in records] == [("A", 1)] * 3
    assert [record.bfactor for record in records] == [70.0, 80.0, 90.0]


def test_later_models_cannot_replace_the_selected_models_confidence(tmp_path):
    """Scoring model zero uses its own C-alpha factor even when another model differs."""
    path = tmp_path / "models.pdb"
    path.write_text("MODEL        1\n" + _atom(1, factor="12.50")
                    + _atom(2, name="CB", factor="45.00") + "ENDMDL\n"
                    + "MODEL        2\n" + _atom(1, factor="91.50")
                    + _atom(2, name="CB", factor="95.00") + "ENDMDL\n", encoding="utf-8")
    first = bff.labelizer_read_structure(str(path), True, 0)
    second = bff.labelizer_read_structure(str(path), True, 1)
    assert list(first.bfactor) == [12.5, 45.0]
    assert list(second.bfactor) == [91.5, 95.0]
    options = bff.LabelizerOptions()
    options.bfactor_is_confidence = True
    scores = bff.labelizer_score_structure(str(path), [], options)
    assert dict(bff.labelizer_confidence_by_key(scores)) == {"A1": 12.5}


def test_t4_factors_are_reported_without_changing_combined_label_scores():
    """The existing T4 fixture's C-alpha column survives the confidence option unchanged."""
    expected = {}
    for line in T4.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            expected[f"{line[21].strip()}{int(line[22:26])}"] = float(line[60:66].strip() or "0")
    assert expected
    model = bff.labelizer_model_paper()
    plain = bff.labelizer_score_structure(str(T4), model, bff.LabelizerOptions())
    options = bff.LabelizerOptions()
    options.bfactor_is_confidence = True
    confidence = bff.labelizer_score_structure(str(T4), model, options)
    assert dict(bff.labelizer_confidence_by_key(confidence)) == expected
    assert not bff.labelizer_confidence_by_key(plain)
    plain_scores = dict(bff.labelizer_combined_by_key(plain))
    with_scores = dict(bff.labelizer_combined_by_key(confidence))
    assert plain_scores.keys() == with_scores.keys()
    np.testing.assert_allclose(list(plain_scores.values()), list(with_scores.values()), equal_nan=True)

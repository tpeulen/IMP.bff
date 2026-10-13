"""``parse_structure_table``: the core reads a structure the way the IMP build does.

``read_structure_table`` fills the table through IMP's readers and exists only
where IMP is linked; a core-only install (the ``imp-bff`` wheel) had no way to
read an mmCIF at all, and no PDB reader that gives a particle's radius, mass
and residue identity. ``parse_structure_table`` restates IMP's rules without
IMP. The expectations below are what IMP's ``read_pdb`` / ``read_mmcif``
return for the same text (read 2026-10-13 in the IMP build), rule by rule:
the atom set of the default selector, residues sorted by number within a
chain, single-precision coordinates, the element from the column or the atom
name, IMP's masses and radii. In the IMP build, ``test_structure_table.py``
holds the two readers equal on the shipped structures.
"""
from pathlib import Path

import numpy as np
import pytest

import IMP.bff

REPO = Path(__file__).resolve().parents[2]


def _rec(kind, serial, name, alt, res, chain, resseq, icode, xyz, b, elem=None):
    line = "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f" % (
        kind, serial, name, alt, res, chain, resseq, icode, xyz[0], xyz[1], xyz[2], 1.0, b)
    if elem is not None:
        line += "          %2s" % elem
    return line


def _synthetic_pdb():
    rows = [
        ("ATOM", " N  ", " ", "GLY", "A", 3, " ", (1.0, 2.0, 3.0), 10.0, "N"),
        ("ATOM", " CA ", " ", "GLY", "A", 3, " ", (1.5, 2.5, 3.5), 11.0, "C"),
        ("ATOM", " N  ", " ", "ALA", "A", 1, " ", (4.1, 5.2, 6.3), 12.0, "N"),
        ("ATOM", " CA ", " ", "ALA", "A", 1, " ", (4.123, 5.456, 6.789), 13.25, "C"),
        ("ATOM", " CB ", "A", "ALA", "A", 1, " ", (4.5, 5.5, 6.5), 14.0, "C"),
        ("ATOM", " CB ", "B", "ALA", "A", 1, " ", (9.5, 9.5, 9.5), 14.0, "C"),
        ("ATOM", " HA ", " ", "ALA", "A", 1, " ", (4.0, 5.0, 6.0), 15.0, "H"),
        ("ATOM", " N  ", " ", "SER", "A", 2, " ", (7.0, 8.0, 9.0), 16.0, None),
        ("ATOM", " OG ", " ", "SER", "A", 2, " ", (7.7, 8.8, 9.9), 17.0, None),
        ("ATOM", " CA ", " ", "LYS", "A", 10, " ", (1.0, 1.0, 1.0), 18.0, "C"),
        ("ATOM", " CA ", " ", "LYS", "A", 10, "A", (2.0, 1.0, 1.0), 18.0, "C"),
        ("ATOM", " NZ ", " ", "LYS", "A", 10, "A", (2.5, 1.0, 1.0), 18.0, "N"),
        ("ATOM", " CA ", " ", "TRP", "A", 10, "B", (3.0, 1.0, 1.0), 18.0, "C"),
        ("ATOM", " CA ", " ", "ARG", "A", 11, " ", (4.0, 1.0, 1.0), 18.0, "C"),
        "TER",
        ("ATOM", " CA ", " ", "MET", "B", 5, " ", (0.0, 0.0, 9.0), 20.0, "C"),
        ("ATOM", " SD ", " ", "MET", "B", 5, " ", (0.0, 1.0, 9.0), 21.0, "S"),
        ("HETATM", " CA ", " ", "ALA", "B", 6, " ", (0.0, 2.0, 9.0), 22.0, "C"),
        ("HETATM", "SE  ", " ", "MSE", "B", 7, " ", (0.0, 3.0, 9.0), 23.0, "SE"),
        ("HETATM", " CE ", " ", "MSE", "B", 7, " ", (0.0, 4.0, 9.0), 23.0, "C"),
        "TER",
        ("HETATM", "ZN  ", " ", " ZN", "A", 301, " ", (5.0, 5.0, 5.0), 30.0, "ZN"),
        ("HETATM", "CL  ", " ", " CL", "A", 302, " ", (6.0, 5.0, 5.0), 31.0, None),
        ("HETATM", " C1 ", " ", "NAG", "A", 303, " ", (7.0, 5.0, 5.0), 32.0, "C"),
        ("HETATM", " O5 ", " ", "NAG", "A", 303, " ", (7.5, 5.0, 5.0), 32.0, "O"),
        ("HETATM", " O  ", " ", "HOH", "A", 401, " ", (8.0, 5.0, 5.0), 33.0, "O"),
        ("HETATM", " O  ", " ", "DOD", "A", 402, " ", (8.5, 5.0, 5.0), 33.0, "O"),
        ("HETATM", " O  ", " ", "WAT", "A", 403, " ", (9.0, 5.0, 5.0), 33.0, "O"),
        ("HETATM", " XX ", " ", "UNL", "A", 404, " ", (9.5, 5.0, 5.0), 34.0, None),
        ("ATOM", " CA ", " ", "GLY", "A", 0, " ", (3.0, 3.0, 3.0), 35.0, "C"),
    ]
    lines = ["HEADER    SYNTHETIC"]
    serial = 0
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
            continue
        serial += 1
        kind, name, alt, res, chain, resseq, icode, xyz, b, elem = row
        lines.append(_rec(kind, serial, name, alt, res, chain, resseq, icode, xyz, b, elem))
    lines.append("END")
    return "\n".join(lines) + "\n"


#: What IMP's read_pdb returns for _synthetic_pdb(), atom by atom:
#: (chain, res_id, insertion code, residue, atom, element, mass, radius, B, serial)
PDB_EXPECTED = [
    ("A", 1, "", "ALA", "N", "N", 14.0067, 1.85, 12.0, 3),
    ("A", 1, "", "ALA", "CA", "C", 12.0107, 2.275, 13.25, 4),
    ("A", 1, "", "ALA", "CB", "C", 12.0107, 2.06, 14.0, 5),
    ("A", 1, "", "ALA", "HA", "H", 1.007, 1.32, 15.0, 7),
    ("A", 2, "", "SER", "N", "N", 14.0067, 1.85, 16.0, 8),
    ("A", 2, "", "SER", "OG", "O", 15.9994, 1.77, 17.0, 9),
    ("A", 3, "", "GLY", "N", "N", 14.0067, 1.85, 10.0, 1),
    ("A", 3, "", "GLY", "CA", "C", 12.0107, 2.175, 11.0, 2),
    ("A", 10, "", "LYS", "CA", "C", 12.0107, 2.275, 18.0, 10),
    ("A", 10, "A", "LYS", "CA", "C", 12.0107, 2.275, 18.0, 11),
    ("A", 10, "A", "LYS", "NZ", "N", 14.0067, 1.85, 18.0, 12),
    ("A", 10, "B", "TRP", "CA", "C", 12.0107, 2.275, 18.0, 13),
    ("A", 11, "", "ARG", "CA", "C", 12.0107, 2.275, 18.0, 14),
    ("B", 5, "", "MET", "CA", "C", 12.0107, 2.275, 20.0, 15),
    ("B", 5, "", "MET", "SD", "S", 32.065, 2.0, 21.0, 16),
    ("B", 6, "", "ALA", "CA", "C", 12.0107, 2.275, 22.0, 17),
    ("B", 7, "", "MSE", "SE", "SE", 78.96, 1.7, 23.0, 18),
    ("B", 7, "", "MSE", "CE", "C", 12.0107, 1.7, 23.0, 19),
    ("A", 0, "", "GLY", "CA", "C", 12.0107, 2.175, 35.0, 28),
    ("A", 301, "", "ZN", "ZN", "ZN", 65.409, 1.7, 30.0, 20),
    ("A", 302, "", "CL", "CL", "CL", 35.453, 1.7, 31.0, 21),
    ("A", 303, "", "NAG", "C1", "C", 12.0107, 1.7, 32.0, 22),
    ("A", 303, "", "NAG", "O5", "O", 15.9994, 1.7, 32.0, 23),
    ("A", 403, "", "WAT", "O", "O", 15.9994, 1.7, 33.0, 26),
    ("A", 404, "", "UNL", "XX", "Un", 0.0, 1.7, 34.0, 27),
]

SYNTHETIC_CIF = """data_SYNTH
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1  N  N   . GLY A 1 2 ? 1.000 2.000 3.000 1.00 10.00 3  X 1
ATOM   2  C  CA  . GLY A 1 2 ? 1.500 2.500 3.500 1.00 11.00 3  X 1
ATOM   3  N  N   . ALA A 1 1 ? 4.100 5.200 6.300 1.00 12.00 1  X 1
ATOM   4  C  CA  . ALA A 1 1 ? 4.123 5.456 6.789 1.00 13.25 1  X 1
ATOM   5  C  CB  A ALA A 1 1 ? 4.500 5.500 6.500 0.50 14.00 1  X 1
ATOM   6  C  CB  B ALA A 1 1 ? 9.500 9.500 9.500 0.50 14.00 1  X 1
ATOM   7  H  HA  . ALA A 1 1 ? 4.000 5.000 6.000 1.00 15.00 1  X 1
ATOM   8  C  CA  . LYS A 1 3 ? 1.000 1.000 1.000 1.00 18.00 10 X 1
ATOM   9  C  CA  . LYS A 1 4 A 2.000 1.000 1.000 1.00 18.00 10 X 1
ATOM   10 C  CA  . TRP A 1 5 ? 3.000 1.000 1.000 1.00 18.00 12B X 1
ATOM   11 C  "C1'" . DA  B 2 1 ? 0.000 0.000 9.000 1.00 20.00 5 Y 1
ATOM   12 C  CA  . MET C 1 1 ? 0.000 0.000 7.000 1.00 20.00 5 . 1
HETATM 13 ZN ZN  . ZN  D 3 . ? 5.000 5.000 5.000 1.00 30.00 301 X 1
HETATM 14 C  C1  . NAG E 4 . ? 7.000 5.000 5.000 1.00 32.00 303 X 1
HETATM 15 O  O   . HOH F 5 . ? 8.000 5.000 5.000 1.00 33.00 401 X 1
HETATM 16 O  O   . WAT F 5 . ? 9.000 5.000 5.000 1.00 33.00 403 X 1
ATOM   17 C  CA  . GLY A 1 2 ? 99.0 2.500 3.500 1.00 11.00 3  X 2
ATOM   18 C  CA  . GLY G 1 1 ? 3.000 3.000 3.000 1.00 35.00 0  X 1
#
"""

#: What IMP's read_mmcif returns for SYNTHETIC_CIF.
CIF_EXPECTED = [
    ("X", 0, "", "GLY", "CA", "C", 12.0107, 2.175, 35.0, 18),
    ("X", 1, "", "ALA", "N", "N", 14.0067, 1.85, 12.0, 3),
    ("X", 1, "", "ALA", "CA", "C", 12.0107, 2.275, 13.25, 4),
    ("X", 1, "", "ALA", "CB", "C", 12.0107, 2.06, 14.0, 5),
    ("X", 1, "", "ALA", "HA", "H", 1.007, 1.32, 15.0, 7),
    ("X", 3, "", "GLY", "N", "N", 14.0067, 1.85, 10.0, 1),
    ("X", 3, "", "GLY", "CA", "C", 12.0107, 2.175, 11.0, 2),
    ("X", 10, "", "LYS", "CA", "C", 12.0107, 2.275, 18.0, 8),
    ("X", 10, "", "LYS", "CA", "C", 12.0107, 2.275, 18.0, 9),
    ("X", 12, "B", "TRP", "CA", "C", 12.0107, 2.275, 18.0, 10),
    ("X", 301, "", "ZN", "ZN", "ZN", 65.409, 1.7, 30.0, 13),
    ("X", 303, "", "NAG", "C1", "C", 12.0107, 1.7, 32.0, 14),
    ("X", 403, "", "WAT", "O", "O", 15.9994, 1.7, 33.0, 16),
    ("Y", 5, "", "DA", "C1'", "C", 12.0107, 2.275, 20.0, 11),
    ("C", 5, "", "MET", "CA", "C", 12.0107, 2.275, 20.0, 12),
]


def _rows(table):
    return list(zip(table.chain, table.res_id, table.insertion_code, table.res_name,
                    table.atom_name, table.element, table.mass, table.radius,
                    table.bfactor, table.atom_id))


def _check(table, expected):
    rows = _rows(table)
    assert len(rows) == len(expected)
    for got, want in zip(rows, expected):
        assert got[:6] == want[:6]
        assert got[6] == pytest.approx(want[6], abs=1e-9)
        # IMP's radii are single precision
        assert got[7] == pytest.approx(want[7], abs=1e-6)
        assert got[8] == pytest.approx(want[8], abs=1e-6)
        assert got[9] == want[9]


def _read(path, **kw):
    kw.setdefault("only_standard_residues", False)
    kw.setdefault("radius_no_interaction", False)
    return IMP.bff.parse_structure_table(str(path), **kw)


@pytest.fixture
def pdb(tmp_path):
    p = tmp_path / "synthetic.pdb"
    p.write_text(_synthetic_pdb())
    return p


@pytest.fixture
def cif(tmp_path):
    p = tmp_path / "synthetic.cif"
    p.write_text(SYNTHETIC_CIF)
    return p


def test_pdb_reads_as_imp_does(pdb):
    """Alternate location B, HOH and DOD dropped; hydrogens and ligands kept;
    the second run of chain A is a chain of its own; residues sorted by number
    within each chain, insertion codes in file order; element from the name
    where the column is missing (`N`, `O`, `CL`), `Un` where neither says."""
    _check(_read(pdb), PDB_EXPECTED)


def test_mmcif_reads_as_imp_does(cif):
    """One chain per author chain id (label G's residue 0 joins X and sorts
    first), the label chain where the author one is `.`, the insertion code
    from `auth_seq_id` ("12B") and not from `pdbx_PDB_ins_code`, model 2
    skipped, quoted atom names unquoted."""
    _check(_read(cif), CIF_EXPECTED)


def test_coordinates_are_single_precision_as_imp_stores_them(pdb):
    table = _read(pdb)
    xyz = np.asarray(table.xyz).reshape(-1, 3)
    assert xyz[0].tolist() == [float(np.float32(v)) for v in (4.1, 5.2, 6.3)]
    assert xyz[1].tolist() == [float(np.float32(v)) for v in (4.123, 5.456, 6.789)]


def test_only_the_first_model(tmp_path):
    lines = ["MODEL        1",
             _rec("ATOM", 1, " CA ", " ", "GLY", "A", 1, " ", (1.0, 2.0, 3.0), 0.0, "C"),
             "ENDMDL", "MODEL        2",
             _rec("ATOM", 1, " CA ", " ", "GLY", "A", 1, " ", (99.0, 2.0, 3.0), 0.0, "C"),
             "ENDMDL", "END"]
    p = tmp_path / "models.pdb"
    p.write_text("\n".join(lines) + "\n")
    table = _read(p)
    assert table.n_atoms == 1
    assert np.asarray(table.xyz).reshape(-1, 3)[0, 0] == 1.0


def test_the_flags_mean_what_read_structure_table_means(pdb):
    everything = _read(pdb, keep_water=True)
    assert {"HOH", "DOD"} <= set(everything.res_name)
    standard = _read(pdb, only_standard_residues=True)
    assert set(standard.res_name) <= {"ALA", "SER", "GLY", "LYS", "TRP", "ARG", "MET", "MSE"}
    assert "NAG" not in set(standard.res_name)
    zero_crossing = IMP.bff.parse_structure_table(str(pdb), only_standard_residues=False)
    np.testing.assert_allclose(np.asarray(zero_crossing.radius),
                               np.asarray(_read(pdb).radius) * 2.0 ** (-1.0 / 6.0),
                               rtol=0, atol=1e-12)


def test_every_column_is_the_same_length(cif):
    table = _read(cif)
    n = table.n_atoms
    assert n == len(CIF_EXPECTED)
    for column in (table.radius, table.mass, table.bfactor, table.atom_id, table.res_id,
                   table.chain, table.res_name, table.atom_name, table.element,
                   table.insertion_code):
        assert len(column) == n
    assert np.asarray(table.xyz).size == 3 * n


@pytest.mark.parametrize("fixture", ["structure/T4L/3GUN.pdb", "structure/GBP/1F5N.cif"])
def test_shipped_structures_carry_imps_radii(fixture):
    path = REPO / "examples" / fixture
    if not path.exists():
        pytest.skip("example structure not present")
    table = _read(path)
    assert table.n_atoms > 1000
    expected = [float(np.float32(IMP.bff.imp_atom_radius(r, a)))
                for r, a in zip(table.res_name, table.atom_name)]
    np.testing.assert_allclose(np.asarray(table.radius), expected, rtol=0, atol=0)


def test_refusals_are_refusals(tmp_path):
    gz = tmp_path / "x.pdb.gz"
    gz.write_bytes(b"")
    with pytest.raises(ValueError):
        IMP.bff.parse_structure_table(str(gz))
    txt = tmp_path / "x.txt"
    txt.write_text("nothing")
    with pytest.raises(ValueError):
        IMP.bff.parse_structure_table(str(txt))
    empty = tmp_path / "empty.pdb"
    empty.write_text("HEADER    NOTHING\nEND\n")
    with pytest.raises(ValueError):
        IMP.bff.parse_structure_table(str(empty))
    with pytest.raises(IOError):
        IMP.bff.parse_structure_table(str(tmp_path / "missing.pdb"))

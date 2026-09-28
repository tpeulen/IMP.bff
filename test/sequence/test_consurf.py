"""ConSurf end to end: a database, the search, homologues, rates, grades, the
.grades file, and the Labelizer's records."""

import os
import random

import pytest

import IMP.bff as bff

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "rbp_60x120.fasta")
AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
         "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
         "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("consurf")
    msa = bff.read_sequence_msa(FIXTURE, match_columns_only=False)
    family = [msa.get_sequence(k).replace("-", "") for k in range(msa.get_n_sequences())]
    rng = random.Random(5)
    with open(d / "db.fasta", "w") as fh:
        for k, s in enumerate(family):
            fh.write(f">fam{k}\n{s}\n")
        for k in range(2000):
            fh.write(f">decoy{k}\n" + "".join(rng.choice(AA) for _ in range(rng.randint(60, 300))) + "\n")
    path = str(d / "db.pto")
    bff.create_sequence_database(str(d / "db.fasta"), path)
    options = bff.ConsurfOptions()
    options.database = path
    # the fixture family is diverse: take the homologues ConSurf's stand-alone way
    options.homologs = bff.SequenceHomologOptions.consurf_standalone()
    return d, family, options


def test_one_pass_several_chains(setup):
    d, family, options = setup
    query = family[0]
    results = bff.compute_consurf([query, "ACDEFGHIKLMNPQ", query], options)
    r = results[0]
    assert r.get_is_ok(), r.status
    assert len(r.homologs) >= 5
    assert all(h.identifier.startswith("fam") for h in r.homologs)
    assert r.msa.get_n_sequences() == len(r.homologs) + 1
    n = r.conservation.get_n_positions()
    assert n == len(query) and len(r.grades) == 4 * n
    assert all(1 <= g <= 9 for g in r.grades[0::4])
    # a peptide has no homologues; a repeated chain gets the same result
    assert not results[1].get_is_ok() and "too few homologues" in results[1].status
    assert list(results[2].grades) == list(r.grades)


def test_grades_file_reads_back_as_labelizer_records(setup):
    d, family, options = setup
    query = family[0]
    r = bff.compute_consurf([query], options)[0]
    labels = [f"{THREE[c]}{k + 1}:A" for k, c in enumerate(query)]
    path = str(d / "q.grades")
    bff.write_consurf_grades(r, path, labels)
    records = bff.labelizer_read_consurf_records(path)
    assert len(records) == len(query)
    first = records["A1"]
    assert first.grade == r.grades[0] and first.insufficient == bool(r.grades[3])
    assert abs(first.score - r.conservation.get_scores()[0]) < 1e-3
    assert first.n_sequences == r.msa.get_n_sequences()
    # the alignment written out computes the same conservation
    bff.write_consurf_msa(r, str(d / "q.fasta"))
    again = bff.compute_consurf_from_msa(bff.read_sequence_msa(str(d / "q.fasta"), 0, False))
    assert list(again.grades) == list(r.grades)


def test_the_labelizer_gets_records_straight_from_the_database(setup):
    d, family, options = setup
    s = bff.LabelizerStructure()
    residues = []
    for chain, seq in (("A", family[0]), ("B", "ACDEFGHIKLMNPQ")):
        for k, c in enumerate(seq):
            r = bff.LabelizerResidue()
            r.chain, r.seq_id, r.comp_id = chain, k + 10, THREE[c]
            residues.append(r)
    s.residues = residues
    records = bff.labelizer_conservation_from_database(s, options)
    assert len(records) == len(family[0])
    assert "A10" in records and not any(k.startswith("B") for k in records)
    direct = bff.compute_consurf([family[0]], options)[0]
    assert records["A10"].grade == direct.grades[0]


def test_without_a_database_the_settings_are_asked(setup, tmp_path, monkeypatch):
    monkeypatch.setenv("IMP_BFF_SETTINGS", str(tmp_path / "none.json"))
    with pytest.raises(bff.IOException):
        bff.compute_consurf(["ACDEFGHIKLMNPQRSTVWY"])

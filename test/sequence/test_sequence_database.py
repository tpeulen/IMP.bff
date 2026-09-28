"""SequenceDatabase: FASTA (or FASTA.gz) streamed into binary FASTA in a .pto."""

import gzip
import json
import os

import pytest

import IMP.bff as bff

FASTA = (
    ">UniRef90_P1 first protein n=3 Tax=Homo sapiens\r\n"
    "MKTAy\nIAKQR*\n"
    ">UniRef90_P2\n"
    "\n"
    ">UniRef90_P3 odd letters\n"
    "ACBZXUOJ-.W\n"
    "  QQ\n"
)
FIXTURE = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "rbp_60x120.fasta")
EXPECTED = [
    ("UniRef90_P1 first protein n=3 Tax=Homo sapiens", "MKTAYIAKQR"),
    ("UniRef90_P2", ""),
    ("UniRef90_P3 odd letters", "ACXXXXXXWQQ"),
]


def _check(db):
    assert db.get_number_of_sequences() == len(EXPECTED)
    assert db.get_number_of_residues() == sum(len(s) for _, s in EXPECTED)
    for i, (header, seq) in enumerate(EXPECTED):
        assert db.get_header(i) == header
        assert db.get_sequence(i) == seq
        assert db.get_length(i) == len(seq)
        assert db.get_identifier(i) == header.split()[0]
    with pytest.raises(bff.IndexException):
        db.get_sequence(len(EXPECTED))


def test_fasta_round_trips(tmp_path):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(FASTA)
    out = str(tmp_path / "db.pto")
    assert bff.create_sequence_database(str(fasta), out) == 3
    _check(bff.SequenceDatabase(out))


def test_gzip_is_read_when_built_with_zlib(tmp_path):
    fasta = tmp_path / "in.fasta.gz"
    fasta.write_bytes(gzip.compress(FASTA.encode()))
    out = str(tmp_path / "db.pto")
    if not bff.get_sequence_database_reads_gzip():
        with pytest.raises(bff.IOException):
            bff.create_sequence_database(str(fasta), out)
        return
    assert bff.create_sequence_database(str(fasta), out, "uniref", 1) == 3
    _check(bff.SequenceDatabase(out, "uniref"))


def test_many_sequences_over_many_segments(tmp_path):
    # the alignment fixture's sequences, gaps dropped, repeated past one 1 MB segment
    msa = bff.read_sequence_msa(FIXTURE, match_columns_only=False)
    seqs = [msa.get_sequence(k).replace("-", "") for k in range(msa.get_n_sequences())]
    fasta = tmp_path / "many.fasta"
    n = 0
    with open(fasta, "w") as fh:
        for rep in range(400):
            for k, s in enumerate(seqs):
                fh.write(f">seq{n} rep{rep}\n{s}\n")
                n += 1
    out = str(tmp_path / "many.pto")
    assert bff.create_sequence_database(str(fasta), out, "sequences", 1) == n
    db = bff.SequenceDatabase(out)
    assert db.get_number_of_sequences() == n
    for i in (0, 1, 59, 60, n // 2, n - 1):
        assert db.get_sequence(i) == seqs[i % len(seqs)]
        assert db.get_identifier(i) == f"seq{i}"


def test_a_database_is_found_by_its_settings_name(tmp_path, monkeypatch):
    fasta = tmp_path / "in.fasta"
    fasta.write_text(FASTA)
    bff.create_sequence_database(str(fasta), str(tmp_path / "db.pto"))
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"sequence_search": {"databases": {"tiny": "db.pto"}}}))
    monkeypatch.setenv("IMP_BFF_SETTINGS", str(settings))
    _check(bff.SequenceDatabase("tiny"))
    with pytest.raises(bff.IOException):
        bff.SequenceDatabase("not-configured")


def test_what_is_not_fasta_is_refused(tmp_path):
    bad = tmp_path / "bad.fasta"
    bad.write_text("ACDEF\n>late header\nAC\n")
    with pytest.raises(bff.IOException):
        bff.create_sequence_database(str(bad), str(tmp_path / "bad.pto"))
    with pytest.raises(bff.IOException):
        bff.create_sequence_database(str(tmp_path / "absent.fasta"), str(tmp_path / "x.pto"))

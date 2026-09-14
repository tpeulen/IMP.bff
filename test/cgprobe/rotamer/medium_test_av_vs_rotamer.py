"""AV ↔ rotamer-ensemble table (PRD-108 stage 2): drift pins + loose sanity bounds.

The table okf/validation/av_vs_rotamer.md is authoritative; the differences
between an AV1 cloud and a screened rotamer library are two physical models
disagreeing, recorded here, not gated. This test pins the recorded numbers
(exact -- both computations are deterministic) and asserts only that the two
models describe the same label: mean positions within 20 Å, ⟨R_DA⟩ within
20 Å, no rotamer with weight > 1e-3 clashing with the protein.

The comparison is C++ (ProbeModelComparison.h) since `imp_bff av-vs-rotamer`
became a compiled command; `imp_bff av-vs-rotamer` rewrites the pin file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import IMP.bff

_PINS = Path(__file__).resolve().parents[2] / "references" / "cgprobe_av_vs_rotamer_pins.json"


def _pins():
    with open(_PINS) as fh:
        return json.load(fh)


@pytest.mark.parametrize("case_name,system", [("hgbp1_cutoff30", "hgbp1"), ("t4l_cutoff30", "t4l")])
def test_av_vs_rotamer_pins_and_sanity(case_name, system):
    pins = _pins()
    settings = pins["settings"]
    case = IMP.bff.av_rotamer_case(system, 30)
    positions = list(IMP.bff.compare_av_and_rotamer_positions(
        case, n_samples=settings["n_samples"], temperature=settings["temperature"]))
    rows = list(IMP.bff.compare_av_and_rotamer_pairs(
        case, n_samples=settings["n_samples"], temperature=settings["temperature"],
        forster_radius=IMP.bff.av_rotamer_forster_radius()))
    per_pos = {p.name: p for p in positions}
    ref = pins[case_name]

    # drift pins (deterministic: lattice AV + fixed-seed sampling; full pair matrix)
    for name, rec in ref["positions"].items():
        got = per_pos[name]
        assert got.n_rotamers == rec["n_rotamers"] and got.av_n_points == rec["av_n_points"]
        assert got.d_mean_position == pytest.approx(rec["d_mean_position"], abs=1e-6)
        assert got.partition == pytest.approx(rec["partition"], abs=1e-9)
    ref_rows = {r["pair"]: r for r in ref["pairs"]}
    for row in rows:
        rec = ref_rows[row.pair]
        for key in ("Rmp_av", "Rmp_rot", "RDAMean_av", "RDAMean_rot", "RDAMeanE_av",
                    "RDAMeanE_rot", "sigma_av", "sigma_rot", "kappa2_rot"):
            assert getattr(row, key) == pytest.approx(rec[key], abs=1e-6), (row.pair, key)

    # sanity: two models of the same label (skipping sites the rotamer model
    # finds buried, Z < 0.05 -- FRETpredict's uniform fallback)
    z_cutoff = IMP.bff.av_rotamer_z_cutoff()
    for name, got in per_pos.items():
        assert got.av_n_points > 0 and got.n_rotamers > 0
        if got.partition < z_cutoff:
            continue
        assert got.d_mean_position < 20.0, name
        # the screening is FRETpredict's soft LJ (sigma_scaling 0.5): rotamers
        # may sit ~1.4 A from protein atoms with full weight (recorded in the
        # table as "weight within 2.5 A"); a rotamer *inside* the protein
        # (< 1.0 A) would be a scoring defect
        assert got.min_heavy_atom_distance > 1.0, name
        assert got.min_heavy_atom_distance == pytest.approx(
            ref["positions"][name]["min_heavy_atom_distance"], abs=1e-6)
        assert got.interpenetration_weight == pytest.approx(
            ref["positions"][name]["interpenetration_weight_2.5A"], abs=1e-9)
    for row in rows:
        if not row.rotamer_valid:
            continue
        assert abs(row.RDAMean_av - row.RDAMean_rot) < 20.0, row.pair
        assert abs(row.RDAMeanE_av - row.RDAMeanE_rot) < 20.0, row.pair


def test_the_markdown_table_and_the_pins_agree_on_the_positions():
    case = IMP.bff.av_rotamer_case("hgbp1", 30)
    positions = list(IMP.bff.compare_av_and_rotamer_positions(case, n_samples=2000))
    rows = list(IMP.bff.compare_av_and_rotamer_pairs(case, n_samples=2000))
    table = IMP.bff.av_rotamer_markdown_table(positions, rows, case.title)
    summary = json.loads(IMP.bff.av_rotamer_summary_json(positions, rows))
    assert list(summary["positions"]) == [p.name for p in positions]
    for p in positions:
        assert "| %s" % p.name in table


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

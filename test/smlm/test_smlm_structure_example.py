"""Small IMP example integration tests; no claim of whole-paper reproduction.

The synthetic model is an asymmetric tetrahedron of expected emitters. The
optional real-data smoke case uses ONE saved cell1 particle (920) and an
explicit external structural prior: the cached human NPC's 32 terminal C-alpha
proxies. Those proxies are not actual fluorophore coordinates. Only the NPZ
__meta__ JSON is read, never its 624k scaffold coordinate/radius arrays. Fixture
conversion, pose calculations and one-time exports are test setup, not a Python
objective loop. Optimization and scoring exercise the native example.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import IMP.algebra
import numpy as np
import pytest

import IMP
from IMP import bff

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/imaging/smlm_structure.py"


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("smlm_structure_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(csv_path, model_path, **overrides):
    values = {"localizations": csv_path, "model_points": model_path, "particle_id": 7,
              "zero_based_particle_ids": False, "measurement_unit": "nm", "model_unit": "nm",
              "source_origin": [12000, 8000, 21], "model_origin": [4, 3, -1],
              "roi_min": [-20]*3, "roi_max": [20]*3, "model_key": "coordinates",
              "model_weight_column": "", "observation_weight_column": "",
              "initial_quaternion": [1, 0, 0, 0], "initial_translation": [-0.1, 0.05, -0.08],
              "background_fraction": 0.03, "intrinsic_sigma": 0.1, "cutoff_sigma": 6.0,
              "steps": 60, "max_change": 1.0, "gradient_threshold": 1e-7,
              "restraint_weight": 2.5, "sum_score": False}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def synthetic(tmp_path):
    emitter_angstrom = np.array([[-5, -2, -1], [4, -1, 0.5], [1, 5, 1], [-1, 0.5, 4]])
    model_origin_nm = np.array([4, 3, -1])
    model_path = tmp_path / "arbitrary_emitters.npz"
    weights = np.array([1, 2, 3, 2], dtype=float)
    np.savez(model_path, coordinates=emitter_angstrom/10+model_origin_nm, weights=weights)
    source_origin_nm = np.array([12000, 8000, 21])
    rotation = IMP.algebra.get_rotation_from_fixed_xyz(0, 0, 0.07)
    truth = IMP.algebra.Transformation3D(rotation, IMP.algebra.Vector3D(0.3, -0.2, 0.15))
    pose = [list(rotation.get_rotation_matrix_row(axis)) + [truth.get_translation()[axis]]
            for axis in range(3)]
    measured_angstrom = np.asarray(bff.transform_smlm_points(emitter_angstrom.ravel().tolist(),
                                                           np.asarray(pose).ravel().tolist())).reshape(-1, 3)
    measured_nm = np.repeat(measured_angstrom, 3, axis=0)/10+source_origin_nm
    sigmas_nm = np.tile([0.08, 0.12, 0.1], (len(measured_nm), 1))
    path = tmp_path / "measured.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["xnm", "ynm", "znm", "xnmerr", "ynmerr", "locprecznm", "sitenumbers"])
        for coordinate, precision in zip(measured_nm, sigmas_nm):
            writer.writerow([*coordinate, *precision, 7])
        # Outside the local ROI, so omission of native particle selection fails.
        writer.writerow([*(source_origin_nm+100), 0.08, 0.12, 0.1, 99])
    return arguments(path, model_path), emitter_angstrom, weights


def assert_pose(report_pose):
    matrix = np.asarray(report_pose["model_to_local_measurement_3x4"])
    assert matrix.shape == (3, 4)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix[:, :3].T @ matrix[:, :3], np.eye(3), atol=3e-12)
    assert np.linalg.det(matrix[:, :3]) == pytest.approx(1, abs=3e-12)
    np.testing.assert_allclose(matrix[:, 3], report_pose["translation_angstrom"], atol=1e-13)
    quaternion = report_pose["quaternion_wxyz"]
    assert np.linalg.norm(quaternion) == pytest.approx(1, abs=3e-12)
    rotation = IMP.algebra.Rotation3D(*quaternion)
    np.testing.assert_allclose(matrix[:, :3],
                               [list(rotation.get_rotation_matrix_row(axis)) for axis in range(3)],
                               atol=3e-12)
    return matrix


def test_example_runs_arbitrary_npz_with_native_units_selection_and_rigid_pose(example, synthetic):
    args, emitter_angstrom, weights = synthetic
    imported_model = example.read_emitter_model(args)
    np.testing.assert_allclose(np.asarray(imported_model.get_coordinates()).reshape(-1, 3),
                               emitter_angstrom, atol=8e-15)
    np.testing.assert_allclose(imported_model.get_weights(), weights)
    report = example.run(args)
    assert report["measured_localizations"] == 12
    assert report["expected_emitters"] == 4
    assert report["particle_id"] == 7
    assert report["coordinate_unit"] == "angstrom"
    assert report["measurement_source_unit"] == report["model_source_unit"] == "nm"
    assert report["source_origin"] == args.source_origin
    assert report["model_origin"] == args.model_origin
    assert report["score_mode"] == "weighted_mean_nll"
    assert report["restraint_weight"] == args.restraint_weight
    assert np.isfinite([report["initial_score"], report["final_score"]]).all()
    assert report["final_score"] < report["initial_score"]-1e-4
    initial = assert_pose(report["initial_pose"])
    final = assert_pose(report["final_pose"])
    np.testing.assert_allclose(initial[:, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(initial[:, 3], args.initial_translation, atol=1e-14)
    assert report["limits"]["native_steps"] == args.steps
    assert report["limits"]["max_attribute_change"] == args.max_change
    json.dumps(report, allow_nan=False)  # Same strict JSON contract as main().

    transformed = np.asarray(bff.transform_smlm_points(emitter_angstrom.ravel().tolist(),
                                                       final.ravel().tolist())).reshape(-1, 3)
    original_distances = np.linalg.norm(emitter_angstrom[:, None]-emitter_angstrom, axis=2)
    final_distances = np.linalg.norm(transformed[:, None]-transformed, axis=2)
    np.testing.assert_allclose(final_distances, original_distances, rtol=3e-13, atol=3e-13)

    # Independently verify that reported poses map the INPUT model frame, not
    # the arbitrary inertia frame a rigid-body constructor could introduce.
    measured = bff.select_smlm_particles(
        bff.read_smlm_csv(str(args.localizations), 10, args.source_origin), [7])
    np.testing.assert_allclose(np.asarray(measured.get_sigmas()).reshape(-1, 3),
                               np.tile([0.8, 1.2, 1.0], (12, 1)), atol=2e-15)
    settings = bff.SMLMLikelihoodOptions()
    settings.roi_min, settings.roi_max = args.roi_min, args.roi_max
    settings.background_fraction = args.background_fraction
    settings.intrinsic_sigma, settings.cutoff_sigma = args.intrinsic_sigma, args.cutoff_sigma
    point_model = bff.SMLMPointModel(emitter_angstrom.ravel().tolist(), weights.tolist())
    for field, matrix in (("initial_score", initial), ("final_score", final)):
        expected = point_model.evaluate(measured, settings, matrix.ravel().tolist()).mean_nll
        assert report[field] == pytest.approx(args.restraint_weight*expected, rel=2e-12, abs=2e-12)


def test_cli_rejects_unrecognized_unit_before_loading_data(example, synthetic, monkeypatch, capsys):
    args, _, _ = synthetic
    monkeypatch.setattr("sys.argv", [str(EXAMPLE), str(args.localizations), str(args.model_points),
                                   "--particle-id", "7", "--measurement-unit", "pixels",
                                   "--model-unit", "nm", "--source-origin", "12000", "8000", "21",
                                   "--roi-min", "-20", "-20", "-20", "--roi-max", "20", "20", "20"])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_zero_group_requires_explicit_opt_in_and_runs_native_selection(example, synthetic, tmp_path):
    args, _, _ = synthetic
    with args.localizations.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames, rows = reader.fieldnames, list(reader)
    path = tmp_path / "zero_based_measured.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames)
        writer.writeheader()
        for row in rows:
            if row["sitenumbers"] == "7":
                row["sitenumbers"] = "0"
            writer.writerow(row)
    args.localizations, args.particle_id = path, 0
    with pytest.raises(ValueError, match="zero-based-particle-ids"):
        example.run(args)
    args.zero_based_particle_ids = True
    report = example.run(args)
    assert report["particle_id"] == 0
    assert report["zero_based_particle_ids"]
    assert report["measured_localizations"] == 12
    assert report["expected_emitters"] == 4
    assert report["final_score"] <= report["initial_score"]+1e-9


def test_cli_reports_invalid_initial_quaternion(example, synthetic, monkeypatch, capsys):
    args, _, _ = synthetic
    monkeypatch.setattr("sys.argv", [str(EXAMPLE), str(args.localizations), str(args.model_points),
                                   "--particle-id", "7", "--measurement-unit", "nm",
                                   "--model-unit", "nm", "--source-origin", "12000", "8000", "21",
                                   "--model-origin", "4", "3", "-1",
                                   "--roi-min", "-20", "-20", "-20", "--roi-max", "20", "20", "20",
                                   "--initial-quaternion", "2", "0", "0", "0"])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 1
    assert "SMLM structural modelling failed" in capsys.readouterr().err


def test_cached_paper_site920_with_explicit_terminal_proxy_prior(example, tmp_path, record_property):
    csv_path = ROOT / ".omx/reference/locmofit/data/210121_U2OS_Nup96-SNAP-AF647_cell1_sml.csv"
    specified = os.environ.get("IMP_BFF_NPC_PROXY_NPZ")
    cache_candidates = ([Path(specified)] if specified else []) + [
        Path.home()/".chimol/chimol_demos/npc_human_scaffold.ihm.npz"]
    cache_path = next((path for path in cache_candidates if path.is_file()), None)
    if not csv_path.is_file() or cache_path is None:
        pytest.skip("Cached paper CSV and human NPC metadata prior absent; no downloads")
    with np.load(cache_path, allow_pickle=False) as archive:
        # Deliberately do NOT access archive['coords'] or any scaffold array.
        metadata = archive["__meta__"].item()
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")
        meta = json.loads(metadata)
    proxy = meta["nup96_terminal_proxy"]
    emitter_angstrom = np.asarray(proxy["coords_A"], dtype=float)
    assert emitter_angstrom.shape == (32, 3)
    assert "not a fluorophore coordinate" in proxy["definition"]
    model_path = tmp_path / "explicit_terminal_proxy_prior.npz"
    np.savez(model_path, coordinates=emitter_angstrom)

    data = bff.read_smlm_csv(str(csv_path), 10)
    selected = bff.select_smlm_particles(data, [920])
    del data
    frames = bff.align_smlm_particles(selected, [920])
    centre_angstrom = np.asarray(frames.centers)
    source_to_reference = np.asarray(frames.transforms).reshape(3, 4)
    rotation = IMP.algebra.get_rotation_from_matrix(*source_to_reference[:, :3].ravel().tolist())
    source_transform = IMP.algebra.Transformation3D(rotation,
                                                   IMP.algebra.Vector3D(*source_to_reference[:, 3]))
    global_to_local = IMP.algebra.Transformation3D(IMP.algebra.Vector3D(*(-centre_angstrom)))
    initial = global_to_local*source_transform.get_inverse()
    local_positions = np.asarray(selected.get_coordinates()).reshape(-1, 3)-centre_angstrom
    args = arguments(csv_path, model_path, particle_id=920, model_unit="angstrom",
                     model_origin=[0, 0, 0], source_origin=(centre_angstrom/10).tolist(),
                     roi_min=(local_positions.min(axis=0)-100).tolist(),
                     roi_max=(local_positions.max(axis=0)+100).tolist(),
                     initial_quaternion=list(initial.get_rotation().get_quaternion()),
                     initial_translation=list(initial.get_translation()),
                     steps=20, max_change=1.0, intrinsic_sigma=20.0, restraint_weight=1.0)
    report = example.run(args)
    assert report["measured_localizations"] == selected.get_number_of_localizations()
    assert report["expected_emitters"] == 32
    assert np.isfinite([report["initial_score"], report["final_score"]]).all()
    assert report["final_score"] <= report["initial_score"]+1e-8
    assert_pose(report["initial_pose"])
    assert_pose(report["final_pose"])
    record_property("paper_particle_id", 920)
    record_property("paper_localizations", report["measured_localizations"])
    record_property("structural_prior", proxy["definition"])
    record_property("initial_mean_nll", report["initial_score"])
    record_property("final_mean_nll", report["final_score"])
    print(f"paper site920: {report['measured_localizations']} measured rows; "
          f"32 terminal proxies; mean NLL {report['initial_score']:.9f} -> "
          f"{report['final_score']:.9f}; at most 20 native steps; "
          "single-site structural-prior smoke, not paper reproduction")

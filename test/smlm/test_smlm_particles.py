"""Particle initialization, rigid registration and native MRC axis contract."""
import numpy as np
import pytest

from IMP import bff


def rigid(rotation=None, translation=(0, 0, 0)):
    if rotation is None:
        rotation = np.eye(3)
    return np.column_stack((rotation, translation)).ravel()


def test_particle_frames_center_tilted_cloud_without_scaling():
    rng = np.random.default_rng(8)
    cloud = rng.normal(size=(500, 3)) * [12, 10, 0.4]
    angle = 0.45
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    shifted = cloud @ rotation.T + [40, -20, 17]
    index = bff.SMLMIndex(shifted.ravel(), np.ones(500), [], [7]*500)
    frames = bff.align_smlm_particles(index, [], 0, 100)
    transformed = np.asarray(bff.transform_smlm_points(shifted.ravel(), frames.transforms)).reshape(-1, 3)
    np.testing.assert_allclose(transformed.mean(0), 0, atol=1e-10)
    assert transformed[:, 2].std() < 0.5
    np.testing.assert_allclose(np.linalg.norm(transformed[0]-transformed[1]),
                               np.linalg.norm(shifted[0]-shifted[1]), atol=1e-10)


def test_precision_selection_preserves_positions_and_measured_sigmas():
    coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    index = bff.SMLMIndex(coordinates.ravel(), [1, 10, 2], [1, 1, 0], [5, 6, 7])
    selected = bff.select_smlm_precision(index, [3, 3, 3])
    np.testing.assert_array_equal(selected.get_coordinates(), [1, 2, 3])
    np.testing.assert_array_equal(selected.get_sigmas(), [1, 1, 1])
    assert list(selected.get_particle_ids()) == [5]


def test_rigid_refinement_reduces_score_and_preserves_distance():
    rng = np.random.default_rng(18)
    target = rng.normal(size=(100, 3))*[2, 4, 1]
    theta = 0.15
    r = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moving = target @ r.T + [0.3, -0.4, 0.2]
    index = bff.SMLMIndex(target.ravel(), np.full(100, 0.5))
    options = bff.SMLMRegistrationOptions()
    options.max_iterations = 150
    options.max_translation_step = 0.25
    fit = bff.refine_smlm_rigid(index, moving.ravel(), rigid(), [], options, 0.1)
    assert fit.score < fit.initial_score - 0.1
    placed = np.asarray(bff.transform_smlm_points(moving.ravel(), fit.transform)).reshape(-1, 3)
    assert np.mean((placed-target)**2) < np.mean((moving-target)**2)
    np.testing.assert_allclose(np.linalg.norm(placed[0]-placed[1]),
                               np.linalg.norm(moving[0]-moving[1]), atol=1e-10)


def test_native_mrc_writer_keeps_grid_axis_order(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    result = bff.SMLMAverageResult()
    result.grid_shape = [2, 3, 4]
    result.origin = [11, 22, 33]
    result.spacing = [2, 2, 2]
    result.values = np.arange(24, dtype=float)
    path = tmp_path / "grid.mrc"
    bff.write_smlm_average_mrc(result, str(path))
    with mrcfile.open(path) as handle:
        np.testing.assert_array_equal(handle.data.ravel(), np.arange(24))
        np.testing.assert_allclose(list(handle.header.origin.tolist()), [11, 22, 33])
        assert tuple(handle.data.shape) == (4, 3, 2)


def test_half_map_frames_do_not_depend_on_other_particle():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(80, 3))*[10, 10, 1]
    coords = np.concatenate((points, points+[200, 0, 0]))
    original = bff.SMLMIndex(coords.ravel(), np.ones(160), [], [1]*80+[2]*80)
    first = bff.align_smlm_particles(original, [1, 2]).transforms
    coords[80:] *= 3
    changed = bff.SMLMIndex(coords.ravel(), np.ones(160), [], [1]*80+[2]*80)
    second = bff.align_smlm_particles(changed, [1, 2]).transforms
    np.testing.assert_array_equal(np.asarray(first)[:12], np.asarray(second)[:12])


def test_assembly_frame_recovers_translated_rotational_axis_and_landmark_midplane():
    pivot = np.array([300., -500., 0.])
    operators = []
    for angle in np.arange(8)*np.pi/4:
        r = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        operators.append(rigid(r, pivot-r@pivot))
    landmarks = np.array([[350., -500., -25.], [300., -450., 75.]])
    frame = np.asarray(bff.get_smlm_assembly_frame(np.concatenate(operators), landmarks.ravel()))
    np.testing.assert_allclose(frame[:3], [300, -500, 25], atol=1e-10)
    np.testing.assert_allclose(frame[3:6], [0, 0, 1], atol=1e-10)
    assert frame[6] < 1e-10

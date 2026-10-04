"""Rotational particle averaging must recover spokes, not just ring normals."""
import numpy as np
import pytest

from IMP import bff


def test_angular_registration_recovers_measured_spokes_without_copying_model():
    rng = np.random.default_rng(401)
    phi = np.arange(8)*np.pi/4
    model = np.concatenate([np.column_stack((10*np.cos(phi), 10*np.sin(phi), np.full(8, z)))
                            for z in (-3, 3)])
    clouds, ids = [], []
    for particle, angle in enumerate(np.linspace(0, np.pi/4, 8, endpoint=False)):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        points = np.repeat(model, 6, axis=0) + rng.normal(0, 0.2, (96, 3))
        clouds.append(points @ rotation.T + [20*particle, -30, 5])
        ids.extend([particle]*96)
    points = np.concatenate(clouds)
    index = bff.SMLMIndex(points.ravel(), np.full(len(points), 0.3), [], ids)
    frames = bff.align_smlm_particles(index, [], 0, 80)
    options = bff.SMLMParticleRegistrationOptions()
    options.angular_period = np.pi/4
    options.angular_samples = 12
    options.intrinsic_sigma = 0.1
    options.roi_padding = 5
    options.max_translation_step = 0.5
    options.max_iterations = 80
    registered = bff.register_smlm_particles(index, frames,
                                           bff.SMLMPointModel(model.ravel()), options)
    assert len(registered.particle_ids) == 8
    assert len(registered.rejected_ids) == 0
    before, after = [], []
    for ordinal, particle in enumerate(registered.particle_ids):
        cloud = clouds[particle]
        before.append(np.asarray(bff.transform_smlm_points(
            cloud.ravel(), np.asarray(frames.transforms)[12*particle:12*particle+12])).reshape(-1, 3))
        after.append(np.asarray(bff.transform_smlm_points(
            cloud.ravel(), np.asarray(registered.transforms)[12*ordinal:12*ordinal+12])).reshape(-1, 3))
    before, after = np.concatenate(before), np.concatenate(after)
    # Fourier amplitude is insensitive to the arbitrary equivalent C8 azimuth.
    initial_contrast = abs(np.exp(8j*np.arctan2(before[:, 1], before[:, 0])).mean())
    contrast = abs(np.exp(8j*np.arctan2(after[:, 1], after[:, 0])).mean())
    assert contrast > 0.97
    assert contrast > initial_contrast + 0.8
    assert len(after) == len(points)  # deposition still contains measurements only
    assert np.std(np.linalg.norm(after[:, :2], axis=1)) > 0.1  # not snapped to model
    assert np.all(np.asarray(registered.final_nll) <= np.asarray(registered.initial_nll)+1e-9)


@pytest.mark.parametrize("field,value", [("angular_samples", 0), ("angular_period", 0),
                                      ("roi_padding", -1), ("min_signal_fraction", 2)])
def test_registration_rejects_invalid_policy(field, value):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    index = bff.SMLMIndex(points.ravel(), [1, 1, 1], [], [1, 1, 1])
    frames = bff.align_smlm_particles(index, [])
    options = bff.SMLMParticleRegistrationOptions()
    setattr(options, field, value)
    with pytest.raises((ValueError, RuntimeError)):
        bff.register_smlm_particles(index, frames, bff.SMLMPointModel(points.ravel()), options)


def test_equal_azimuth_scores_preserve_initial_pose_sample_order():
    # A single emitter at the origin has no identifiable orientation. All
    # azimuth scores tie exactly; stable ranking must retain the first sample.
    points = np.concatenate((np.eye(3)*10, -np.eye(3)*10))
    index = bff.SMLMIndex(points.ravel(), [2]*6, [], [7]*6)
    angle = 0.4
    initial = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0]])
    frames = bff.SMLMParticleFrames()
    frames.particle_ids = [7]
    frames.transforms = initial.ravel()
    options = bff.SMLMParticleRegistrationOptions()
    options.angular_samples = 24
    options.max_iterations = 5
    result = bff.register_smlm_particles(index, frames, bff.SMLMPointModel([0, 0, 0]), options)
    assert list(result.particle_ids) == [7]
    np.testing.assert_allclose(np.asarray(result.transforms).reshape(3, 4), initial, atol=1e-14)

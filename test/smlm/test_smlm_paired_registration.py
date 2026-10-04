"""Target channels inherit reference poses; target shape never chooses a pose."""
import numpy as np
import pytest

from IMP import bff


def reference():
    result = bff.SMLMParticleRegistrationResult()
    result.particle_ids = [7, 4, 9]
    result.rejected_ids = [13]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result.transforms = np.concatenate([
        np.column_stack((rotation, [i, -i, 2*i])).ravel() for i in range(3)])
    result.initial_nll = [10, 11, 12]
    result.final_nll = [8, 9, 10]
    result.signal_fraction = [0.8, 0.9, 0.7]
    result.iterations = [3, 4, 5]
    result.converged = [1, 0, 1]
    return result


def test_targets_inherit_exact_source_frames_and_reference_diagnostics():
    index = bff.SMLMIndex([10, 20, 30, 40, 50, 60, -500, 300, 90],
                          [1, 2, 3], [], [9, 9, 7])
    fitted = reference()
    inherited = bff.transfer_smlm_registration(fitted, index, 1)
    assert list(inherited.particle_ids) == [7, 9]
    assert list(inherited.rejected_ids) == [13, 4]
    np.testing.assert_array_equal(np.asarray(inherited.transforms).reshape(-1, 12),
                                  np.asarray(fitted.transforms).reshape(-1, 12)[[0, 2]])
    np.testing.assert_array_equal(inherited.final_nll, [8, 10])
    np.testing.assert_array_equal(inherited.iterations, [3, 5])
    strict = bff.transfer_smlm_registration(fitted, index, 2)
    assert list(strict.particle_ids) == [9]
    # Replace the target geometry by an entirely different, distant cloud.
    changed = bff.SMLMIndex([1e6]*9, [10]*3, [], [9, 9, 7])
    other = bff.transfer_smlm_registration(fitted, changed, 1)
    np.testing.assert_array_equal(other.transforms, inherited.transforms)


def test_paired_density_rotates_target_measurement_covariance_not_reference_noise():
    fitted = reference()
    target = bff.SMLMIndex([0, 0, 0], [1, 3, 2], [], [7])
    paired = bff.transfer_smlm_registration(fitted, target)
    options = bff.SMLMAverageOptions()
    result = bff.average_smlm_particles(target, paired.particle_ids, paired.transforms,
                                       [21, 21, 21], [-10]*3, [1]*3, options)
    density = np.asarray(result.values).reshape(21, 21, 21)
    axis = np.arange(-10, 11)
    variance_x = np.sum(density*axis[None, None, :]**2)/density.sum()
    variance_y = np.sum(density*axis[None, :, None]**2)/density.sum()
    assert variance_x > 7*variance_y  # quarter turn sends broad source y to x


@pytest.mark.parametrize("bad", ["count", "transforms", "ids", "diagnostics", "rigid"])
def test_invalid_paired_reference_contract_is_rejected(bad):
    fitted = reference()
    index = bff.SMLMIndex([0, 0, 0], [1], [], [7])
    minimum = 1
    if bad == "count":
        minimum = 0
    elif bad == "transforms":
        fitted.transforms = [0]
    elif bad == "ids":
        fitted.particle_ids = [7, 7, 9]
    elif bad == "diagnostics":
        fitted.final_nll = [1]
    elif bad == "rigid":
        values = np.asarray(fitted.transforms).copy()
        values[0] = 2
        fitted.transforms = values
    with pytest.raises((ValueError, RuntimeError)):
        bff.transfer_smlm_registration(fitted, index, minimum)


def test_no_paired_targets_returns_an_empty_selection_not_fabricated_density():
    index = bff.SMLMIndex([0, 0, 0], [1], [], [99])
    paired = bff.transfer_smlm_registration(reference(), index)
    assert len(paired.particle_ids) == 0
    assert list(paired.rejected_ids) == [13, 7, 4, 9]

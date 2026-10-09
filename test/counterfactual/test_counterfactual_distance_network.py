"""CounterfactualDistanceNetwork: a correct analysis that gives the wrong answer.

A synthetic network: six sites move along a one-parameter path (40 candidate
structures), all 15 pairs are measured, and one site carries a hidden +6
bias that pushes the usual fit nine structures away at a reduced chi^2 of 1.9. Every quantity is checked against the numpy construction of notebook 09
section 6, and the behaviour the test exists for is asserted: the usual chi^2
analysis is confidently wrong, abducting the site biases puts it right, only
the biased site's intervention changes the conclusion, and the replay exposes
the blind spot that fresh-noise simulation does not.
"""
import itertools

import numpy as np
import pytest

from IMP.bff import CounterfactualDistanceNetwork

N_SITES, N_CAND = 6, 40
PAIRS = list(itertools.combinations(range(N_SITES), 2))
SIGMA, TAU, NOISE = 2.0, 3.0, 1.0
F_TRUE, BAD, BIAS = 25, 3, 6.0


def model_table():
    rng = np.random.default_rng(21)
    start = rng.normal(0, 25, (N_SITES, 3))
    direction = rng.normal(0, 1, (N_SITES, 3))
    direction[0] *= 0.2                       # one site barely moves
    t = np.linspace(0, 12, N_CAND)
    pos = start[None] + t[:, None, None] * direction[None]
    return np.array([[np.linalg.norm(p[i] - p[j]) for i, j in PAIRS] for p in pos])


MODEL = model_table()
A_INC = np.zeros((len(PAIRS), N_SITES))
for k, (i, j) in enumerate(PAIRS):
    A_INC[k, i] = A_INC[k, j] = 1


def measured():
    b = np.zeros(N_SITES)
    b[BAD] = BIAS
    return MODEL[F_TRUE] + A_INC @ b + np.random.default_rng(1).normal(0, NOISE, len(PAIRS))


def network(y):
    net = CounterfactualDistanceNetwork(MODEL.ravel(), N_CAND, np.ravel(PAIRS).tolist(), N_SITES)
    net.set_measurement(y, [SIGMA] * len(PAIRS))
    net.set_site_bias_prior([TAU] * N_SITES)
    return net


def reference(y):
    chi2 = ((y - MODEL) ** 2).sum(1) / SIGMA ** 2
    w_std = np.exp(-(chi2 - chi2.min()) / 2); w_std /= w_std.sum()
    S = SIGMA ** 2 * np.eye(len(PAIRS)) + TAU ** 2 * A_INC @ A_INC.T
    Si = np.linalg.inv(S)
    lp = np.array([-0.5 * (y - m) @ Si @ (y - m) for m in MODEL])
    w = np.exp(lp - lp.max()); w /= w.sum()
    b_f = np.array([TAU ** 2 * A_INC.T @ Si @ (y - m) for m in MODEL])
    return w_std, w, b_f


def test_matches_the_numpy_construction():
    y = measured()
    net = network(y)
    w_std, w, b_f = reference(y)
    assert np.allclose(net.get_candidate_posterior(False), w_std)
    assert np.allclose(net.get_candidate_posterior(True), w)
    assert np.allclose(np.reshape(net.get_site_biases_given_candidates(), (N_CAND, N_SITES)), b_f)
    assert np.allclose(net.get_abducted_site_biases(), w @ b_f)
    assert np.allclose(net.get_ideal_measurement(), y - A_INC @ (w @ b_f))
    f_hat = int(np.argmax(w))
    assert np.allclose(net.get_abducted_noise(), y - MODEL[f_hat] - A_INC @ b_f[f_hat])


def test_the_usual_analysis_is_wrong_and_the_bias_aware_one_is_right():
    net = network(measured())
    standard = np.asarray(net.get_candidate_posterior(False))
    aware = np.asarray(net.get_candidate_posterior(True))
    assert abs(int(standard.argmax()) - F_TRUE) >= 6
    assert standard[F_TRUE] < 1e-3
    assert abs(int(aware.argmax()) - F_TRUE) <= 1
    b = np.asarray(net.get_abducted_site_biases())
    assert int(np.argmax(np.abs(b))) == BAD
    assert b[BAD] == pytest.approx(BIAS, abs=1.5)
    sd = np.asarray(net.get_abducted_site_bias_sd())
    assert np.all(sd > 0) and sd[BAD] < TAU


def test_the_hinge_points_at_the_biased_site():
    net = network(measured())
    hinge = np.reshape(net.get_hinge_posteriors(), (N_SITES, N_CAND))
    standard_answer = int(np.argmax(net.get_candidate_posterior(False)))
    moved = [abs(int(row.argmax()) - standard_answer) for row in hinge]
    assert int(np.argmax(moved)) == BAD
    assert abs(int(hinge[BAD].argmax()) - F_TRUE) <= 1


def test_replay_exposes_what_fresh_noise_hides():
    net = network(measured())
    replay = np.asarray(net.get_counterfactual_replay())
    fresh = np.asarray(net.get_fresh_noise_replay(5))
    frames = np.arange(N_CAND)
    assert np.mean(np.abs(fresh - frames)) < np.mean(np.abs(replay - frames))
    assert replay.shape == (N_CAND,)


def test_no_bias_prior_means_no_difference():
    net = network(measured())
    net.set_site_bias_prior([0.0] * N_SITES)
    assert np.allclose(net.get_candidate_posterior(True), net.get_candidate_posterior(False))
    assert np.allclose(net.get_abducted_site_biases(), 0.0)


def test_bad_input_is_refused():
    with pytest.raises(ValueError):
        CounterfactualDistanceNetwork(MODEL.ravel()[:-1], N_CAND, np.ravel(PAIRS).tolist(), N_SITES)
    with pytest.raises(ValueError):
        CounterfactualDistanceNetwork(MODEL.ravel(), N_CAND, [0, 9] * len(PAIRS), N_SITES)
    net = CounterfactualDistanceNetwork(MODEL.ravel(), N_CAND, np.ravel(PAIRS).tolist(), N_SITES)
    with pytest.raises(ValueError):
        net.get_candidate_posterior(True)          # no measurement yet
    with pytest.raises(ValueError):
        net.set_measurement([1.0] * len(PAIRS), [0.0] * len(PAIRS))

"""CausalLinearGaussian: interventions, abduction and counterfactuals in closed form.

Every number is checked against an independent numpy construction: the joint
Gaussian of X = (I - B)^-1 U, conditioned by the textbook formula, and the
twin-network definition of the natural effects evaluated by Monte Carlo.
"""
import numpy as np
import pytest

from IMP.bff import CausalLinearGaussian

A_LM, B_MY, C_LY = 1.0, 0.8, 0.3      # effector -> hinge -> active site, and direct
G_WM, H_WY = 1.0, -1.2                 # latent W -> hinge, W -> active site


def mediation(confounded=True):
    m = CausalLinearGaussian()
    m.add_variable("L", 0.0, 0.5)
    m.add_variable("W", 0.0, 1.0)
    m.add_variable("M", 0.0, 0.5)
    m.add_variable("Y", 0.0, 0.5)
    m.add_edge("L", "M", A_LM)
    m.add_edge("M", "Y", B_MY)
    m.add_edge("L", "Y", C_LY)
    if confounded:
        m.add_edge("W", "M", G_WM)
        m.add_edge("W", "Y", H_WY)
    return m


def reference_joint(model):
    names = list(model.get_variable_names())
    n = len(names)
    B = np.zeros((n, n))
    sd = {"L": 0.5, "W": 1.0, "M": 0.5, "Y": 0.5}
    for i, child in enumerate(names):
        for p in model.get_parents(child):
            B[i, names.index(p)] = {("L", "M"): A_LM, ("M", "Y"): B_MY, ("L", "Y"): C_LY,
                                    ("W", "M"): G_WM, ("W", "Y"): H_WY}[(p, child)]
    A = np.linalg.inv(np.eye(n) - B)
    S = np.diag([sd[v] ** 2 for v in names])
    return names, A, S


def test_observational_and_interventional_moments():
    m = mediation()
    names, A, S = reference_joint(m)
    world = m.get_interventional()
    assert np.allclose(world.mean, 0.0)
    assert np.allclose(np.reshape(world.covariance, (4, 4)), A @ S @ A.T)
    do = m.get_interventional(["M"], [2.0])
    # do(M = 2): M is constant, Y = c L + b 2 + h W + U_Y
    assert do.get_mean("M") == pytest.approx(2.0)
    assert do.get_sd("M") == pytest.approx(0.0, abs=1e-12)
    assert do.get_mean("Y") == pytest.approx(B_MY * 2.0)
    assert do.get_sd("Y") == pytest.approx(np.sqrt(C_LY ** 2 * 0.25 + H_WY ** 2 + 0.25))


def test_conditioning_matches_the_gaussian_formula():
    m = mediation()
    names, A, S = reference_joint(m)
    C = A @ S @ A.T
    obs, o = [2, 3], np.array([1.5, 0.4])      # M and Y observed
    rest = [0, 1]
    K = C[np.ix_(rest, obs)] @ np.linalg.inv(C[np.ix_(obs, obs)])
    mean = K @ o
    cov = C[np.ix_(rest, rest)] - K @ C[np.ix_(obs, rest)]
    world = m.get_conditional(["M", "Y"], [1.5, 0.4])
    assert world.get_mean("L") == pytest.approx(mean[0])
    assert world.get_mean("W") == pytest.approx(mean[1])
    full = np.reshape(world.covariance, (4, 4))
    assert np.allclose(full[np.ix_(rest, rest)], cov)
    assert world.get_mean("M") == pytest.approx(1.5) and world.get_sd("M") == pytest.approx(0.0, abs=1e-6)


def test_a_fully_observed_unit_has_a_deterministic_counterfactual():
    m = mediation()
    x = {"L": 1.0, "W": 0.7, "M": 1.2, "Y": 0.1}
    # all four observed: every noise term is known exactly
    noise = m.get_abducted_noise(list(x), list(x.values()))
    u_m = x["M"] - A_LM * x["L"] - G_WM * x["W"]
    u_y = x["Y"] - B_MY * x["M"] - C_LY * x["L"] - H_WY * x["W"]
    assert noise.get_mean("U[M]") == pytest.approx(u_m)
    assert noise.get_mean("U[Y]") == pytest.approx(u_y)
    assert noise.get_sd("U[Y]") == pytest.approx(0.0, abs=1e-6)
    # had the effector been absent (do(L = 0)), with this unit's own noise
    cf = m.get_counterfactual(list(x), list(x.values()), ["L"], [0.0])
    m_cf = G_WM * x["W"] + u_m
    y_cf = B_MY * m_cf + H_WY * x["W"] + u_y
    assert cf.get_mean("M") == pytest.approx(m_cf)
    assert cf.get_mean("Y") == pytest.approx(y_cf)
    assert cf.get_sd("Y") == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("confounded", [False, True])
def test_natural_effects_are_the_product_and_the_direct_weight(confounded):
    total, nde, nie = mediation(confounded).get_natural_effects("L", ["M"], "Y", 0.0, 1.0)
    assert nde == pytest.approx(C_LY)
    assert nie == pytest.approx(A_LM * B_MY)
    assert total == pytest.approx(nde + nie)


def test_natural_effects_match_the_twin_definition_by_monte_carlo():
    rng = np.random.default_rng(3)
    n = 200_000
    W, UM, UY = rng.normal(0, 1.0, n), rng.normal(0, 0.5, n), rng.normal(0, 0.5, n)
    M0 = G_WM * W + UM
    Y_t1_M0 = C_LY + B_MY * M0 + H_WY * W + UY
    Y_t0 = B_MY * M0 + H_WY * W + UY
    Y_t1 = C_LY + B_MY * (A_LM + M0) + H_WY * W + UY
    total, nde, nie = mediation().get_natural_effects("L", ["M"], "Y", 0.0, 1.0)
    assert nde == pytest.approx((Y_t1_M0 - Y_t0).mean(), abs=0.01)
    assert nie == pytest.approx((Y_t1 - Y_t1_M0).mean(), abs=0.01)


def test_unit_level_effects_use_the_abducted_noise():
    # linear SCM: unit-level natural effects equal the population ones
    unit = mediation().get_natural_effects("L", ["M"], "Y", 0.0, 1.0, ["M", "Y"], [2.0, -1.0])
    assert unit[1] == pytest.approx(C_LY) and unit[2] == pytest.approx(A_LM * B_MY)


def test_samples_reproduce_the_interventional_moments():
    m = mediation()
    x = np.reshape(m.get_samples(100_000, 7, ["L"], [1.0]), (-1, 4))
    world = m.get_interventional(["L"], [1.0])
    assert np.allclose(x.mean(0), world.mean, atol=0.02)
    assert np.allclose(np.cov(x.T), np.reshape(world.covariance, (4, 4)), atol=0.03)


def test_cycles_and_bad_input_are_refused():
    m = mediation()
    with pytest.raises(ValueError):
        m.add_edge("Y", "L", 1.0)
    with pytest.raises(ValueError):
        m.add_variable("L")
    with pytest.raises(ValueError):
        m.get_interventional(["Q"], [1.0])
    with pytest.raises(ValueError):
        m.get_counterfactual(["M"], [], [], [])
    with pytest.raises(ValueError):
        m.get_natural_effects("L", ["Y"], "Y", 0.0, 1.0)

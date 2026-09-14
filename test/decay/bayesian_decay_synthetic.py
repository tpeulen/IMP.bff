"""A small synthetic experiment in the manifest format `BayesianDecayModel.h` reads.

Three samples -- donor only (D0), donor-acceptor (DA) and a reference dye (REF)
-- in two polarised detectors (gp VV, gs VH), one pulse each; a coarse lifetime,
distance, acceptor and rotational grid; transfer tensors that are random but
positive (the algebra under test does not care what physics produced them);
measured responses that are zero outside their support, so the shift's clamp
bites. The variables and priors are the kinds the CBM56 model uses. Written
2026-09-14 for PRD-142's tests.
"""

import json
import os

import numpy as np

KINT, NR, NA, NRHO, NRHO_A, NC, NBIN = 5, 12, 2, 3, 2, 6, 64
DT = 0.1
SAMPLES, DETS = ("D0", "DA", "REF"), (("gp", 0), ("gs", 1))


def write(directory, seed=0):
    rng = np.random.default_rng(seed)
    K = KINT + 2
    arrays = {}

    def put(name, a):
        arrays[name] = np.ascontiguousarray(a, dtype=np.float64)

    put("E_base", rng.uniform(0.1, 1.0, (K, KINT)))
    put("E_S_R", rng.uniform(0.0, 0.3, (NR, K, KINT)))
    put("E_S_Ag", rng.uniform(0.0, 0.2, (NR, NA, K, KINT)))
    put("E_S_Ag_rot_r", rng.uniform(0.0, 0.1, (NRHO_A, NR, NA, K, KINT)))
    put("E_A_dir", rng.uniform(0.0, 0.5, (NA, K)))
    put("E_A_dir_rot_r", rng.uniform(0.0, 0.2, (NRHO_A, NA, K)))
    put("E_S_rho", rng.uniform(-0.2, 0.2, (NRHO, K, KINT)))
    put("tau_c", np.geomspace(0.3, 8.0, KINT))
    put("rel", np.linspace(0.4, 2.0, NR))
    put("spl", rng.uniform(0.0, 1.0, (NR, NC)))
    q, _ = np.linalg.qr(np.column_stack([np.ones(NC), rng.normal(size=(NC, NC - 1))]))
    put("Q_c", q[:, 1:])
    put("spec_smooth_P", np.eye(KINT))
    variables, off = [], 0

    def var(name, size, transform, family="gaussian_on_z", a=0.0, b=1.0, lo=0.0, hi=1.0, constrained=None):
        nonlocal off
        variables.append(dict(name=name, offset=off, size=size, constrained_size=constrained or size, transform=transform,
                              lo=lo, hi=hi, family=family, a=[a] * size, b=[b] * size))
        off += size

    var("c", NC - 1, "sum_to_zero", "pspline", constrained=NC)
    var("x_d0", 1, "logit", "uniform", 0.0, 0.5, 0.0, 0.5)
    var("spec_eps", KINT, "identity", "spectrum_smoothness")
    var("w_a", NA - 1, "alr", constrained=NA)
    var("w_rho", NRHO - 1, "alr", constrained=NRHO)
    var("w_rho_a", NRHO_A - 1, "alr", constrained=NRHO_A)
    var("r0_d", 1, "logit", "uniform", 0.15, 0.4, 0.15, 0.4)
    var("log10_tau_ref", 1, "identity", "gaussian", 0.6, 0.05)
    var("w_rho_ref", NRHO - 1, "alr", constrained=NRHO)
    var("r0_ref", 1, "logit", "uniform", 0.15, 0.4, 0.15, 0.4)
    var("r0_a", 1, "logit", "uniform", 0.0, 0.45, 0.0, 0.45)
    for nm, med, sd in (("g", 0.95, 0.02), ("C_GD", 0.94, 0.02), ("C_GA", 0.02, 0.2), ("G_GREEN", 1.0, 0.1),
                        ("QY_D", 0.7, 0.1), ("QY_A", 0.3, 0.1), ("EX_AG", 0.01, 0.3)):
        var(nm, 1, "log", "lognormal", med, sd)
    for d, _ in DETS:
        var(f"irf_shift_{d}", 1, "identity", "gaussian", 0.0, 0.1)
        var(f"irf_bg_{d}", 1, "log", "lognormal", 0.03, 1.0)
    for s in ("D0", "DA"):
        for d, _ in DETS:
            var(f"t_shift_{s}_{d}", 1, "identity", "gaussian", 0.0, 0.1)
    for s in SAMPLES:
        var(f"log_scale_{s}", 1, "identity", "gaussian", 9.0, 1.0)
    keys, parts, responses = [], [], []
    scopes = [dict(spectrum="donor", don_ex="1", fret=False, sensitised=False, has_acc=False, direct=None, rho="w_rho", r0="r0_d"),
              dict(spectrum="donor", don_ex="1", fret=True, sensitised=True, has_acc=True, direct="EX_AG*n_da*per_mol", rho="w_rho", r0="r0_d"),
              dict(spectrum="reference", don_ex="1", fret=False, sensitised=False, has_acc=False, direct=None, rho="w_rho_ref", r0="r0_ref")]
    for si, s in enumerate(SAMPLES):
        for d, pol in DETS:
            ch = f"{s}_{d}_{'vh' if pol else 'vv'}"
            var(f"scat_{ch}", 1, "log", "lognormal", 0.02, 1.0)
            var(f"bkg_{ch}", 1, "log", "lognormal", 0.01, 1.0)
            ri = len(responses)
            responses.append(dict(sample=s, det=d, bg_var=f"irf_bg_{d}", shift_var=f"irf_shift_{d}",
                                  offset_var=(f"t_shift_{s}_{d}" if s != "REF" else None)))
            t = np.arange(NBIN) - 6.0
            resp = np.zeros(NBIN)
            resp[2:20] = 1e4 * np.exp(-0.5 * t[2:20] ** 2 / 2.0) + 20.0
            put(f"response_{ri}", resp)
            parts.append(dict(out=len(keys), channel=ch, amp_index=len(keys), det=d, phys=d, offset_ns=0.0, sample=s,
                              response=ri, scope=si, colour=0, pol=pol))
            keys.append(ch)
    put("y", rng.poisson(200.0, (len(keys), NBIN)).astype(float))
    put("mask", np.ones((len(keys), NBIN)))
    manifest = dict(axis=dict(n=NBIN, dt=DT, period=NBIN * DT, K=K, Kint=KINT), soft=0.05, ref_spec_sd=0.05,
                    keys=keys, data_keys=keys, responses=responses, scopes=scopes, parts=parts, variables=variables,
                    fixed_values=dict(log10_lam=[1.0], l1=[0.0175], l2=[0.0526]), dim=off,
                    pspline=dict(n=NC, order=2, tilt_sd=3.0, quad_sd=30.0, rank=NC - 2, family="gaussian", nu=3.0,
                                 space="log", link="softmax", lam=10.0),
                    spec_smooth=dict(lam_s=0.01, weak_sd=3.0, n=KINT, logdet=0.0),
                    arrays={k: dict(shape=list(v.shape), dtype="float64", file=f"{k}.bin") for k, v in arrays.items()})
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    for k, v in arrays.items():
        v.tofile(os.path.join(directory, f"{k}.bin"))
    # a generic point: every coordinate off its default, the shifts off whole channels
    theta = rng.normal(0.0, 0.5, off)
    for v in variables:
        o = v["offset"]
        if v["name"].startswith("log_scale_"):
            theta[o] = 9.0
        elif v["name"].startswith(("irf_shift_", "t_shift_")):
            theta[o] = rng.uniform(0.05, 0.2)
        elif v["name"] == "log10_tau_ref":
            theta[o] = 0.4
        elif v["transform"] == "log":
            theta[o] = np.log(v["a"][0]) + rng.normal(0.0, 0.2)
    theta.tofile(os.path.join(directory, "theta.bin"))
    return manifest, theta

"""Four checks of the FRET network model on photons it did not make itself.

    PYTHONPATH=<build>/lib python examples/spectroscopy/fret_network_validation.py [a b c d]

Each check simulates bursts (bff.simulate_fret_measurement, then a burst search
by interphoton time), fits them (MAP by L-BFGS, then Laplace), and prints what
it measured:

a. **Microtimes narrow the posterior.** The same two-state bursts are fitted
   with their microtimes and with only their arrival times and colours.
b. **A dark acceptor is not low FRET.** One conformer, and an acceptor that
   blinks into a state that neither accepts nor emits. Two models are fitted:
   two conformers with steady dyes, and one conformer with a blinking acceptor.
   Without PIE and microtimes they explain the photons equally well; with them,
   only the photophysical one does.
c. **One pair's non-monotonic distance, and the joint fit.** A double-well
   landscape observed by two pairs. Along the path, pair 1's distance rises and
   falls, and pair 2's rises. Pair 1 alone, read with discrete states, picks
   two states by BIC, but misplaces them: one mixes a well with the path's
   excursion (41 and 58 A for true wells at 50 and 40). The joint landscape
   fit, with the structural path prior, recovers the wells, the barrier and D.
d. **Diffusion through the focus is not blinking.** Two-state bursts crossing
   a Gaussian focus are fitted with the full arrival model (the count rate is
   information) and the conditional one (arrival times given).
"""

from __future__ import annotations

import sys
import time

import numpy as np

import IMP.bff as bff

NB = 32
BIN = 0.25  # ns per microtime bin


def dye_pair(acceptor_blinks=False):
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 1.0, 0.6, 3.0, 1.0)
    if acceptor_blinks:
        # Dark and not accepting: to the donor it is as if there were none.
        a.add_state("dark", 0.0, 0.0, 3.0, 0.0)
        a.set_rate("bright", "dark", 8.0)
        a.set_rate("dark", "bright", 12.0)
    return d, a


def instrument(nb=NB, pie=True):
    pulses = ["dp", "ap"] if pie else ["dp"]
    exc = [60.0, 1.0, 0.0, 25.0] if pie else [60.0, 1.0]
    excitation = bff.PhotophysicsCrosstalkMatrix(pulses, ["donor", "acceptor"], exc)
    emission = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
                                               [0.4, 0.03, 0.05, 0.45])
    ins = bff.FRETInstrument(excitation, emission, nb, BIN if nb > 1 else 1.0)
    ins.set_background(0, 0.5)
    ins.set_background(1, 0.8)
    if nb > 1:
        t = np.arange(nb)
        for p in range(len(pulses)):
            irf = np.exp(-0.5 * ((t - (2 + p * nb // 2)) / 0.8) ** 2)
            for c in range(2):
                ins.set_irf(p, c, list(irf))
    return ins


def measurement(name, d, a, ins, distances=None, path=None):
    m = bff.FRETMeasurement(name, d, a, 55.0)
    m.set_instrument(ins)
    if distances is not None:
        for k, r in enumerate(distances):
            m.set_state_distance(k, r)
    if path is not None:
        m.set_distance_map(path, 2.0)
    return m


def options(n, duration=1.0, focus=False, seed=11):
    o = bff.FRETSimulationOptions()
    o.n_molecules, o.duration, o.focus, o.seed = n, duration, focus, seed
    return o


def bursts(process, m, o, gap=0.15, min_photons=30):
    return bff.select_bursts(bff.simulate_fret_measurement(process, m, o), gap, min_photons)


def strip_microtimes(data):
    return bff.FRETPhotonData(list(data.get_macrotimes()), list(data.get_channels()), [],
                              list(data.get_segment_starts()), list(data.get_segment_stops()))


def fit(net, jitter=0.3, seed=0, iterations=300):
    theta = np.asarray(net.get_theta())
    theta = theta + np.random.default_rng(seed).normal(0.0, jitter, theta.size)
    opts = bff.FRETLandscapeFitOptions()
    opts.max_iterations = iterations
    result = net.fit(list(theta), opts)
    return result, net.laplace(list(result.get_theta()))


def report(laplace, names=None):
    out = {}
    for name, value, sigma in zip(laplace.get_names(), laplace.get_values(),
                                  laplace.get_natural_sigmas()):
        if names is None or name in names:
            out[name] = (value, sigma)
    return out


def two_state_process(k01=6.0, k10=4.0):
    p = bff.FRETHiddenProcess(2)
    p.set_rate(0, 1, k01)
    p.set_rate(1, 0, k10)
    return p


def check_a():
    """Microtimes on and off, same photons.

    Widths are compared at the truth (the Laplace precision there is the
    information the photons carry), so two fits landing at different MAPs do
    not make the comparison; the MAPs are printed beside them.
    """
    process = two_state_process()
    d, a = dye_pair()
    m = measurement("pair", d, a, instrument(), distances=[45.0, 65.0])
    data = bursts(process, m, options(600))
    rows, maps = {}, {}
    for label, ins, photons in (("microtime", instrument(), data),
                                ("macrotime", instrument(nb=1), strip_microtimes(data))):
        truth = bff.FRETNetworkModel(two_state_process())
        truth.add_measurement(measurement("pair", d, a, ins, distances=[45.0, 65.0]), photons)
        rows[label] = report(truth.laplace(list(truth.get_theta())))
        net = bff.FRETNetworkModel(two_state_process(4.0, 6.0))
        net.add_measurement(measurement("pair", d, a, ins, distances=[48.0, 62.0]), photons)
        _result, lap = fit(net)
        maps[label] = report(lap)
    print(f"a. {data.get_n_segments()} bursts, {data.get_n_photons()} photons; "
          "sigma at the truth, and the MAP")
    for name in rows["microtime"]:
        on, off = rows["microtime"][name], rows["macrotime"][name]
        print(f"   {name:20s} sigma with microtimes {on[1]:6.3f}, without {off[1]:6.3f}"
              f" (ratio {off[1] / on[1]:4.2f});   MAP {maps['microtime'][name][0]:7.2f}"
              f" / {maps['macrotime'][name][0]:7.2f}   true {on[0]:6.2f}")


def check_b():
    """A blinking acceptor against a second conformer."""
    process = bff.FRETHiddenProcess(1)
    d, a = dye_pair(acceptor_blinks=True)
    truth = measurement("pair", d, a, instrument(), distances=[45.0])
    data = bursts(process, truth, options(300, seed=21))
    print(f"b. {data.get_n_segments()} bursts, {data.get_n_photons()} photons")
    for label, pie, nb in (("macrotime, no PIE", False, 1), ("microtime + PIE", True, NB)):
        photons = data if nb > 1 else strip_microtimes(data)
        if not pie:
            # Without PIE the acceptor pulse's photons are not recorded at all.
            photons = keep_donor_pulse(photons, data)
        logl = {}
        for model in ("two conformers", "blinking acceptor"):
            if model == "two conformers":
                proc = two_state_process(5.0, 5.0)
                dd, aa = dye_pair()
                dist = [44.0, 80.0]
            else:
                proc = bff.FRETHiddenProcess(1)
                dd, aa = dye_pair(acceptor_blinks=True)
                dist = [44.0]
            net = bff.FRETNetworkModel(proc)
            net.add_measurement(measurement("pair", dd, aa, instrument(nb, pie), distances=dist),
                                photons)
            result, lap = fit(net, jitter=0.1)
            logl[model] = (result.get_log_likelihood(), len(result.get_theta()))
        (l2, k2), (l1, k1) = logl["two conformers"], logl["blinking acceptor"]
        print(f"   {label:20s} logL two conformers {l2:12.1f} ({k2} free)"
              f"   blinking acceptor {l1:12.1f} ({k1} free)   difference {l1 - l2:9.1f}")


def keep_donor_pulse(photons, full):
    """Photons of the donor pulse only (microtime in the first half period)."""
    t = np.asarray(photons.get_macrotimes())
    c = np.asarray(photons.get_channels())
    b = np.asarray(full.get_microtimes())
    keep = b < NB // 2
    index = np.cumsum(keep) - 1
    starts, stops = [], []
    for s, e in zip(photons.get_segment_starts(), photons.get_segment_stops()):
        inside = np.flatnonzero(keep[s:e + 1]) + s
        if inside.size >= 2:
            starts.append(int(index[inside[0]]))
            stops.append(int(index[inside[-1]]))
    return bff.FRETPhotonData(list(t[keep]), list(c[keep]), [], starts, stops)


def check_c():
    """A non-monotonic pair alone, and the joint landscape fit.

    The regime matters: the molecule must dwell in each well for many photons
    (milliseconds) and cross through a populated path region, or every pair
    sees one averaged distance and nothing is identifiable. Hence a low
    barrier, D = 1.5 q^2/ms on q in [0, 1], and 5 ms bursts.
    """
    q_knots = [0.0, 0.25, 0.5, 0.75, 1.0]
    heights = [4.0, 0.0, 1.5, 0.0, 4.0]  # two wells, a low (1.5 kT) barrier
    pair1 = [50.0, 50.0, 72.0, 40.0, 40.0]  # up, then down, across the transition
    pair2 = [38.0, 40.0, 55.0, 70.0, 70.0]  # monotonic
    process = bff.FRETHiddenProcess(0.0, 1.0, 24, 5)
    process.set_landscape(heights)
    process.set_diffusion(1.5)
    d, a = dye_pair()
    data = []
    for i, path in enumerate((pair1, pair2)):
        m = measurement(f"pair{i + 1}", d, a, instrument(), path=path)
        data.append((m, bursts(process, m, options(400, duration=5.0, seed=31 + i))))
    print(f"c. {[x.get_n_segments() for _, x in data]} bursts, "
          f"{[x.get_n_photons() for _, x in data]} photons per pair")

    m1, d1 = data[0]
    for k in (2, 3, 4):
        proc = bff.FRETHiddenProcess(k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    proc.set_rate(i, j, 1.0)
        dist = list(np.linspace(42.0, 70.0, k))
        net = bff.FRETNetworkModel(proc)
        net.add_measurement(measurement("pair1", d, a, instrument(), distances=dist), d1)
        result, lap = fit(net, jitter=0.1)
        n = d1.get_n_photons()
        bic = -2.0 * result.get_log_likelihood() + len(result.get_theta()) * np.log(n)
        means = sorted(v for name, (v, s) in report(lap).items() if ".mean[" in name)
        print(f"   pair 1 alone, {k} discrete states: BIC {bic:12.1f}  distances "
              + " ".join(f"{v:5.1f}" for v in means))

    proc = bff.FRETHiddenProcess(0.0, 1.0, 24, 5)
    proc.set_landscape([2.0] * 5)
    proc.set_diffusion(1.0)
    net = bff.FRETNetworkModel(proc)
    for i, (m, x) in enumerate(data):
        path = (pair1, pair2)[i]
        prior = list(np.asarray(path) + 1.5)  # the structure is not exact
        net.add_measurement(measurement(m.get_name(), d, a, instrument(), path=prior), x)
        net.set_map_prior_from_path(i, q_knots, prior, 3.0)
    # The roughness penalty is omega sum (d2u / h^2)^2 over knots h apart: on
    # q in [0, 1] with h = 0.25 a 4 kT well costs ~4000 omega, so 1e-2 flattens
    # the landscape (23% within 2 sigma) and 1e-5 lets the data shape it.
    net.set_roughness_weight(1e-5)
    result, lap = fit(net, jitter=0.0, iterations=500)
    u = np.asarray(lap.get_landscape())
    s = np.asarray(lap.get_landscape_sigma())
    q = np.asarray(process.get_coordinates())
    truth = np.asarray(process.get_landscape())
    truth = truth - truth.mean()
    populated = truth - truth.min() < 4.0
    within = np.mean(np.abs(u - truth)[populated] <= 2.0 * s[populated] + 1e-12)
    D = report(lap).get("hidden.D", (np.nan, np.nan))
    print(f"   joint landscape ({result.get_status()}, {result.get_n_iterations()} it): "
          f"{100 * within:.0f}% of the populated grid within 2 sigma; "
          f"D {D[0]:.2f} +- {D[1]:.2f} (true 1.50)")
    for qi, ui, si, ti in zip(q[::3], u[::3], s[::3], truth[::3]):
        print(f"     q {qi:4.2f}  u {ui:6.2f} +- {si:4.2f}  true {ti:6.2f}")


def blinking_donor():
    """A donor that may blink: what a count-rate drop would be blamed on."""
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    d.add_state("dark", 0.0, 0.0, 4.0)
    d.set_rate("bright", "dark", 2.0)
    d.set_rate("dark", "bright", 20.0)
    return d


def check_d():
    """The full and the conditional arrival models under focus modulation.

    The truth has no blinking. Both models are allowed a donor dark state;
    the question is how much blinking each invents to explain the intensity
    rising and falling as the molecule crosses the focus.
    """
    process = two_state_process(6.0, 4.0)
    d, a = dye_pair()
    m = measurement("pair", d, a, instrument(), distances=[45.0, 65.0])
    # Long transits, so the thinned bursts keep enough photons.
    data = bursts(process, m, options(300, duration=6.0, focus=True, seed=41))
    print(f"d. {data.get_n_segments()} bursts, {data.get_n_photons()} photons, focus crossing, "
          "no blinking in the truth")
    for label, model in (("full", bff.FRET_ARRIVAL_FULL),
                         ("conditional", bff.FRET_ARRIVAL_CONDITIONAL)):
        net = bff.FRETNetworkModel(two_state_process(4.0, 6.0))
        net.add_measurement(measurement("pair", blinking_donor(), a, instrument(),
                                        distances=[48.0, 62.0]), data)
        net.set_arrival_model(0, model)
        result, lap = fit(net, jitter=0.1)
        values = report(lap)
        on = values["pair.donor.rate[bright->dark]"][0]
        off = values["pair.donor.rate[dark->bright]"][0]
        net.set_theta(list(result.get_theta()))  # occupancies at the fit, not the start
        occupancy = np.asarray(net.segment_occupancies(0, "donor")).reshape(-1, 2)
        k01, k10 = values["hidden.rate[0->1]"], values["hidden.rate[1->0]"]
        print(f"   {label:12s} donor dark {100 * on / (on + off):5.1f}% (kon {on:6.2f}, koff "
              f"{off:7.2f}); per-burst dark time {100 * occupancy[:, 1].mean():5.1f}%;  "
              f"rates {k01[0]:5.2f} +- {k01[1]:4.2f}, {k10[0]:5.2f} +- {k10[1]:4.2f} (true 6, 4)")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["a", "b", "c", "d"]
    for key in wanted:
        start = time.time()
        {"a": check_a, "b": check_b, "c": check_c, "d": check_d}[key]()
        print(f"   ({time.time() - start:.0f} s)", flush=True)

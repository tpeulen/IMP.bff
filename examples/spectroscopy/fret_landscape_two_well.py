"""
A free-energy landscape and a diffusion coefficient from photons
=================================================================

The two-well validation of Dingeldein & Covino, "A differentiable
photon-by-photon likelihood for continuous free-energy landscapes and
diffusion coefficients from single-molecule FRET" (arXiv:2608.21061, Fig. 2),
with ``IMP.bff.FRETLandscapeModel``.

The donor-acceptor distance diffuses on an asymmetric double well (wells at
4.5 and 7.3 nm, barrier about 5 kT above the deeper well) with
D = 1.5 nm^2/ms. Photons are simulated with the paper's photophysics
(Table I: brightness 24 /ms per channel, backgrounds 1.6 and 4.0 /ms,
crosstalk 0.97/0.03 and 0.08/0.92, R0 = 6 nm, 200 ms traces). The fit
follows the paper's pipeline: data-driven initial guess, MAP by L-BFGS on
the log posterior (roughness, anchor and background priors of Table II),
then Laplace uncertainties from the empirical Fisher information.

Everything is C++; this script only sets the numbers and prints them. The
fit needs an IMP.bff built with tttrlib (the optimiser is tttrlib's L-BFGS).

    python fret_landscape_two_well.py [--traces 60] [--grid 80] [--dt 5e-4]

The paper uses 300 traces, a 200-point grid and dt = 5e-6 ms; the defaults
here are smaller so the example runs in a few minutes on one core. The run
recorded in okf/fret-landscape.md used `--traces 300 --grid 100 --dt 1e-4`.
"""

import argparse
import time

import numpy as np

import IMP.bff


def true_landscape(x):
    z = (np.asarray(x) - 5.9) / 1.4
    return 4.5 * (z ** 2 - 1) ** 2 - 0.5 * z


def main(n_traces=60, n_grid=80, dt=5e-4, duration=200.0, seed=1, max_iterations=400,
         out=None):
    model = IMP.bff.FRETLandscapeModel(3.75, 8.75, n_grid, 25, 6.0)
    knots = np.asarray(model.get_knots())
    truth = np.asarray(model.pack_parameters(list(true_landscape(knots)), 1.5,
                                             [24.0, 24.0], [1.6, 4.0]))
    t0 = time.time()
    photons = model.simulate(truth, n_traces, duration, dt, seed)
    print(f"simulated {photons.get_n_photons()} photons in {n_traces} traces "
          f"({time.time() - t0:.0f} s)")
    model.set_photons(photons)
    model.set_roughness_weight(2.15e-4)
    model.set_anchor_sigma(1.0)
    model.set_background_prior([1.6, 4.0], [0.16, 0.4])

    t0 = time.time()
    guess = model.initial_guess()
    print(f"initial guess: bin width {guess.get_bin_width()} ms, "
          f"D0 = {model.get_diffusion(guess.get_theta()):.2f} ({time.time() - t0:.0f} s)")
    options = IMP.bff.FRETLandscapeFitOptions()
    options.max_iterations = max_iterations
    t0 = time.time()
    fit = model.fit(guess.get_theta(), options)
    theta = np.asarray(fit.get_theta())
    print(f"fit: {fit.get_status()} after {fit.get_n_iterations()} iterations "
          f"({time.time() - t0:.0f} s), log posterior {fit.get_log_posterior():.1f} "
          f"(truth {model.log_posterior(truth):.1f})")

    lap = model.laplace(theta)
    lap_true = model.laplace(truth)
    x = np.asarray(model.get_grid())
    u_hat = np.asarray(lap.get_landscape())
    sigma = np.asarray(lap.get_landscape_sigma())
    u_true = np.asarray(model.get_landscape(truth))
    # The offset of u is free; align the truth on the populated region
    # (Boltzmann weights of the truth) and compare where u* is within 6 kT
    # of its minimum -- the walls beyond carry no photons.
    w = np.exp(-(u_true - u_true.min()))
    w /= w.sum()
    u_true_aligned = u_true - np.sum(w * (u_true - u_hat))
    region = (u_true - u_true.min()) < 6.0
    z = np.abs(u_hat - u_true_aligned)[region] / sigma[region]
    d_hat, d_sig = lap.get_diffusion(), lap.get_diffusion_sigma()
    print(f"D     = {d_hat:.3f} +- {d_sig:.3f} nm^2/ms  (true 1.5, "
          f"{abs(d_hat - 1.5) / d_sig:.2f} sigma)")
    bar_true = lap_true.get_barrier()
    print(f"barrier u(x_b)-u(x_min) = {lap.get_barrier():.2f} +- {lap.get_barrier_sigma():.2f} kT "
          f"(true {bar_true:.2f} at the same kind of points; x_min {lap.get_x_min():.2f}, "
          f"x_b {lap.get_x_barrier():.2f} nm)")
    print(f"landscape: {np.mean(z <= 1):.0%} of the populated grid within 1 sigma, "
          f"{np.mean(z <= 2):.0%} within 2 sigma, max |error| "
          f"{np.max(np.abs(u_hat - u_true_aligned)[region]):.2f} kT, median sigma "
          f"{np.median(sigma[region]):.2f} kT")
    for name, est, sig, true in (("a_g", lap.get_amplitudes()[0], lap.get_amplitude_sigmas()[0], 24.0),
                                 ("a_r", lap.get_amplitudes()[1], lap.get_amplitude_sigmas()[1], 24.0),
                                 ("beta_g", lap.get_backgrounds()[0], lap.get_background_sigmas()[0], 1.6),
                                 ("beta_r", lap.get_backgrounds()[1], lap.get_background_sigmas()[1], 4.0)):
        print(f"{name:7s}= {est:.3f} +- {sig:.3f}  (true {true}, {100 * (est - true) / true:+.1f}%)")
    if out:
        np.savez(out, x=x, u_hat=u_hat, sigma=sigma, u_true=u_true_aligned, theta=theta,
                 truth=truth)
    return dict(d=d_hat, d_sigma=d_sig, within1=np.mean(z <= 1), within2=np.mean(z <= 2),
                barrier=lap.get_barrier(), barrier_sigma=lap.get_barrier_sigma(),
                barrier_true=bar_true)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--traces", type=int, default=60)
    p.add_argument("--grid", type=int, default=80)
    p.add_argument("--dt", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument("--out", default=None, help="save the curves to this .npz")
    a = p.parse_args()
    main(a.traces, a.grid, a.dt, seed=a.seed, max_iterations=a.max_iterations, out=a.out)

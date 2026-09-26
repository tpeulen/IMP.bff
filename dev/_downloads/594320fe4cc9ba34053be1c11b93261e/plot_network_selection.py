"""
Choosing FRET pairs for structure *and* kinetics
================================================
``IMP.bff.select_probe_pairs`` (Olga) picks the pairs that best resolve the
*structure*: the expected RMSD between the true conformer and the one the
measurements would single out. It does not ask whether the network can tell
the *rates* apart. Two pairs that both separate A from B see the same
exchange, and a network built from them can pin every state and still leave a
transition undetermined.

``IMP.bff.ProbeNetworkSelection`` mixes terms by a weighted sum of their
losses, each in ``[0, 1]`` relative to nothing measured:

* ``ProbeResolutionTerm`` -- Olga's expected RMSD;
* ``ProbeKineticsTerm`` -- how well the rates of a kinetic scheme are
  determined. Each pair is measured on its own molecules, so pairs share the
  rates. Per pair, the Fisher information of simulated bursts on the log-rates
  after the pair's own state distances are paid for (a Schur complement); the
  loss is the geometric-mean posterior variance of the log-rates relative to
  their prior;
* ``ProbeLabellingTerm`` -- the Labelizer's site scores and a cost per
  mutation (not used here; see the tests).

A three-state scheme, A <-> B <-> C plus A <-> C, and thirty candidate pairs.
"""
import numpy as np
import pylab as plt

import IMP.bff as bff

# %%
# The kinetic scheme (rates in 1/ms), and thirty candidate pairs with random
# distances per state -- what an accessible-volume screen over a structural
# model would hand over.
process = bff.FRETHiddenProcess(3)
rates = [(0, 1, 2.0), (1, 0, 3.0), (1, 2, 1.5), (2, 1, 2.5), (0, 2, 0.5), (2, 0, 0.7)]
for i, j, k in rates:
    process.set_rate(i, j, k)

rng = np.random.default_rng(1)
distances = rng.uniform(35.0, 75.0, (3, 30))   # states x candidates, Angstrom
n_pairs = distances.shape[1]
r0 = 52.0
donor = bff.FRETDye("donor")
donor.add_state("bright", 1.0, 0.8, 4.0)
acceptor = bff.FRETDye("acceptor")
acceptor.add_state("bright", 1.0, 0.6, 3.0, 1.0)
template = bff.FRETMeasurement("pair", donor, acceptor, r0)
# One excitation, two detection channels, bright enough for ~40 photons in
# a 2 ms burst.
exc = bff.PhotophysicsCrosstalkMatrix(["p"], ["donor", "acceptor"], [60.0, 1.0])
em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
 [0.4, 0.03, 0.05, 0.45])
ins = bff.FRETInstrument(exc, em, 1, 0.25)
ins.set_background(0, 0.5)
ins.set_background(1, 0.5)
template.set_instrument(ins)

# %%
# The structural side, as Olga sees it: the three states are the ensemble's
# frames, with their mean FRET efficiency per pair, and their RMSDs a line
# A - B - C.
effs = 1.0 / (1.0 + (distances / r0) ** 6)
rmsds = np.array([[0.0, 6.0, 12.0], [6.0, 0.0, 6.0], [12.0, 6.0, 0.0]])
resolution = bff.ProbeResolutionTerm(effs, rmsds, measurement_error=0.05)

# %%
# The kinetic side: bursts simulated per candidate, scored by the FRET network
# model, reduced to the rates.
options = bff.FRETSimulationOptions()
options.n_molecules = 300
options.duration = 2.0
options.seed = 11
kinetics = bff.ProbeKineticsTerm(process, template, options=options,
                                 n_bursts_per_pair=1000.0)
kinetics.add_pairs(distances)
print("rates:", list(kinetics.get_rate_names()))

# %%
# Three selections of three pairs: resolution only, kinetics only, and the mix.
def run(w_res, w_kin, k=3):
    sel = bff.ProbeNetworkSelection(n_pairs)
    sel.add_term(resolution, w_res)
    sel.add_term(kinetics, w_kin)
    units, _ = sel.select(k)
    losses = np.asarray(sel.get_term_losses())
    return list(units), losses, np.asarray(kinetics.get_rate_sigmas())


results = {name: run(*w) for name, w in
           [("resolution", (1.0, 0.0)), ("kinetics", (0.0, 1.0)), ("mixed", (1.0, 1.0))]}
print(f"{'selection':<11} {'pairs':<12} {'RMSD loss':>9} {'rate loss':>9}  "
      f"{'worst sd(log k)':>15}")
for name, (units, losses, sigmas) in results.items():
    print(f"{name:<11} {str([int(u) for u in units]):<12} "
          f"{losses[-1, 0]:>9.4f} {losses[-1, 1]:>9.4f}  "
          f"{sigmas.max():>15.3f}")

# %%
# Both losses along each selection. Three states are resolved structurally by
# almost any three pairs that separate them, so Olga's choice among those is
# arbitrary as far as the rates go. The mix spends that freedom on the rates:
# the same structural precision, a rate variance several times smaller.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
for name, (units, losses, _) in results.items():
    steps = np.arange(1, len(units) + 1)
    axes[0].plot(steps, losses[:, 0], "o-", label=name)
    axes[1].plot(steps, losses[:, 1], "o-", label=name)
axes[0].set_ylabel("expected RMSD / prior")
axes[1].set_ylabel("rate variance / prior (geometric mean)")
for ax in axes:
    ax.set_xlabel("pairs selected")
axes[1].set_yscale("log")
axes[0].legend()
fig.tight_layout()
plt.show()

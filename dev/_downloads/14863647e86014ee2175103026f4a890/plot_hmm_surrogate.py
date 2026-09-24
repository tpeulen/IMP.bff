"""
Training an HMM surrogate (amortised neural estimator)
=======================================================

Photon-by-photon hidden Markov modelling (H2MM) normally recovers a model by
iterating Baum-Welch EM on every dataset. A *surrogate* takes a different
route: train a small neural network **once** on data simulated from the HMM
generative model, then estimate the parameters of a dataset in a **single
forward pass**.

This example trains a surrogate with ``IMP.bff.HmmSurrogate``, saves it, and
compares it with tttrlib's EM on simulated data where the answer is known.
The photon data and the EM fit are tttrlib's; the network and its training
are IMP.bff's, so this needs an IMP.bff built with tttrlib.

The estimate is approximate. It is fast and well behaved where EM struggles
-- when two states are so close that the likelihood surface is flat -- but on
well-separated states EM is more accurate. The surrogate predicts parameters;
it does not steer EM. Its estimate can still be handed to
``tttrlib.HMM.optimize`` as a start, which the last section does.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import IMP.bff
import tttrlib

# %%
# Choose the regime
# -----------------
# A surrogate is specific to a ``(n_states, n_streams)`` pair *and* to the
# burst length and inter-photon spacing it was trained on, so these numbers
# should match the experiment you intend to analyse.

N_STATES = 2
N_STREAMS = 2      # donor and acceptor
N_BURSTS = 150     # bursts per simulated dataset
BURST_LEN = 80     # photons per burst
MEAN_DT = 4.0      # mean inter-photon gap, macro-time ticks
SEED = 12345

# %%
# Train
# -----
# Training simulates ``n_samples`` labelled datasets and regresses the model
# parameters on a 24-element, burst-order-invariant summary of each. The
# simulation and the network both run in C++.

options = IMP.bff.NeuralNetTrainOptions()
options.hidden_layer_sizes = [256, 256, 128]
options.max_iter = 800
options.early_stopping = True

surrogate = IMP.bff.HmmSurrogate.train(
    N_STATES, N_STREAMS,
    2500,          # n_samples: more is better, with diminishing returns
    N_BURSTS, BURST_LEN, MEAN_DT,
    options, SEED,
)

print(surrogate)
print("features:", surrogate.get_net().get_n_inputs())
print("targets: ", surrogate.get_net().get_n_outputs())

# %%
# The training curve. Early stopping halts training once the held-out loss
# stops improving, so it is usually shorter than ``max_iter``.

loss = np.asarray(surrogate.get_loss_curve())
validation = np.asarray(surrogate.get_validation_curve())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(loss, label="training")
if validation.size:
    ax.plot(validation, label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("half mean squared error (standardised)")
ax.set_yscale("log")
ax.set_title("Surrogate training")
ax.legend()
fig.tight_layout()

# %%
# Save it
# -------
# The model is one msgpack document -- msgpack is the native format of every
# network in IMP.bff -- a surrogate map nesting the map of its network: exact
# float64 weights, compact, safe to share, and loadable wherever IMP.bff is.
# ``msgpack.unpackb(surrogate.to_msgpack(), raw=False)`` shows it as a plain
# dict.

path = os.path.join(tempfile.mkdtemp(), "hmm_surrogate_2state.msgpack")
surrogate.to_file(path)
reloaded = IMP.bff.HmmSurrogate.from_file(path)

# %%
# Compare with EM
# ---------------
# Simulate datasets with known FRET efficiencies and estimate them both ways.
# ``separation`` is the gap between the two true efficiencies -- the variable
# that decides whether the problem is identifiable at all.


def simulate(e_lo, e_hi, switch_prob, rng):
    """A two-state kinetic dataset, as the bursts tttrlib.HMM.set_bursts takes."""
    times, streams = [], []
    for _ in range(N_BURSTS):
        t = np.concatenate(
            [[0], np.cumsum(rng.poisson(MEAN_DT, BURST_LEN - 1) + 1)]
        ).astype(np.int64)
        s = np.empty(BURST_LEN, dtype=np.int32)
        state = rng.random() < 0.5
        for j in range(BURST_LEN):
            if j and rng.random() < switch_prob:
                state = not state
            s[j] = rng.random() < (e_hi if state else e_lo)
        times.append(t)
        streams.append(s)
    return times, streams


def load(times, streams):
    engine = tttrlib.HMM()
    engine.set_bursts(
        tttrlib.VectorVectorInt64([tttrlib.VectorInt64(t.tolist()) for t in times]),
        tttrlib.VectorVectorInt32([tttrlib.VectorInt32(s.tolist()) for s in streams]),
        N_STREAMS,
    )
    return engine


separations, err_surrogate, err_em, err_polished = [], [], [], []
for trial in range(24):
    rng = np.random.default_rng(SEED + 1000 + trial)
    e_lo, e_hi = sorted(rng.uniform(0.15, 0.85, 2))
    times, streams = simulate(e_lo, e_hi, 0.02, rng)
    engine = load(times, streams)
    truth = np.array([e_lo, e_hi])

    estimate = reloaded.predict_hmm(engine)            # a tttrlib.HmmModel
    fit = engine.fit(N_STATES, 1, 500, 1e-7, SEED, True, False)
    polished = engine.optimize(estimate, 500)          # EM from the estimate

    separations.append(e_hi - e_lo)
    err_surrogate.append(np.abs(np.sort(estimate.obs_np[:, 1]) - truth).mean())
    err_em.append(np.abs(np.sort(fit.obs_np[:, 1]) - truth).mean())
    err_polished.append(np.abs(np.sort(polished.obs_np[:, 1]) - truth).mean())

print("mean |E error|  surrogate %.3f  EM %.3f  EM from surrogate %.3f"
      % (np.mean(err_surrogate), np.mean(err_em), np.mean(err_polished)))

# %%
# Plotted against state separation, the trade-off is clear: the surrogate's
# error is nearly flat, while EM is excellent on well-separated states and
# degrades as they merge and the maximum-likelihood estimate drifts.

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(separations, err_surrogate, label="surrogate (one forward pass)")
ax.scatter(separations, err_em, marker="x", label="EM (Baum-Welch)")
ax.scatter(separations, err_polished, marker="+", label="EM from the surrogate")
ax.set_xlabel("true separation of the two FRET states")
ax.set_ylabel("mean |E error|")
ax.set_title("Where each estimator wins")
ax.legend()
fig.tight_layout()

plt.show()

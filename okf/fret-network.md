---
okf_version: "0.2"
---

# A conformational process seen by a network of FRET pairs (`FRETNetworkModel`)

The single-pair landscape model ([fret-landscape.md](fret-landscape.md))
generalised along the three axes the owner asked for (2026-09-23/24): **bursts
with microtimes**, not only interphoton times; a **FRET state that is not one
index**, since dye photophysics and conformation are different processes; and
**several label pairs observing one hidden process**, with structure priors.
All C++ (`include/FRETNetwork.h`, `include/FRETNetworkSimulation.h`,
`src/FRETNetwork*.cpp`), wrapped in `pyext/include/IMP_bff.fretlandscape.i`.

## Where to pick this up

1. **Speed of the landscape case.** The joint two-pair landscape fit (24-point
   grid, 112 k photons) takes about 8 minutes. The uniformization series costs
   `nnz × q × τ` per gap, and `q` is dominated by the grid hopping `D/dq²`.
   OpenMP is off in the local build (see fret-landscape.md). The dense path
   (below) only pays off for small state spaces.
2. **Diffusing bursts bias the fit unless brightness is modelled (measured
   2026-09-24).** Bursts simulated by tttrlib had molecules diffusing through
   a 3-D Gaussian focus, with PIE and tttrlib's burst search. The conditional
   arrival model, with each state's signal-to-background ratio fixed at the
   focus centre, overestimates D: 2.10 ± 0.23 (2.25 ms transits) and
   2.35 ± 0.22 (1 ms), true 1.50. It also puts the barrier ~0.5 kT high and
   overshoots a distance map by ~10 Å. The C++ and the PyTorch fits agree.
   A free signal scale per pair fits 0.30: burst photons come, on average,
   from 30 % of the centre brightness. With it, D = 1.70 ± 0.54 and 100 % of
   the landscape is within 2σ, but the bands are wide. The evidence is in the
   local prototype `prototypes/fret_network_landscape/` (`log.md`, scripts
   01–04, PyTorch). The fix is the next item.
3. **A presence/focus factor is planned, not built.** It is a hidden factor
   (absent / in focus, optionally 0/1/2 molecules), under which only background
   emits. It lets the whole photon stream be one segment: no burst search and
   no selection bias. The factor machinery (`KineticNetwork` amalgamation,
   per-state emission) takes it without an API change.
4. **Ensemble (MSM) priors: prototyped in PyTorch, not in C++ (2026-09-25).**
   `prototypes/fret_network_landscape/ensemble_prior.py` and script 07 build the
   prior from a conformational ensemble, either time-connected frames (an MSM
   by counting at a lag) or a given MSM:
   - log populations and log exchange fluxes of a reversible chain;
   - per-microstate distances per pair from the frames' label distances.

   Against a force field 4× too slow, the photons recover the relaxation
   (0.084 ± 0.035 ms, exact 0.075). The prior keeps the populations and the
   transition-region distances, which a weak prior lets wander.
   *Trap:* build the MSM at a lag where its implied timescales have converged.
   At a short lag a coarse MSM underestimates the relaxation (0.042 at 0.05 ms,
   exact 0.075). The C++ `FRETHiddenProcess` would need the population/flux
   parametrisation and these priors to take it.
5. **Position-parameterised structure prior.** Today each pair's distance map
   has an independent Gaussian prior from structure
   (`set_map_prior_from_path`, per-state `set_parameter_prior`). The planned
   form gives each hidden state 3-D mean label positions (FPS/NPS), so that one
   state's distances across pairs are geometrically consistent by
   construction; the API was shaped for it.
6. **The roughness weight is not scale-free.** The penalty is
   `ω Σ (Δ²u/h²)²`. On q ∈ [0, 1] with 5 knots, ω = 1e-2 flattened a 4 kT
   landscape (23 % within 2σ, bands overconfident); 1e-5 recovers it (95 %),
   and 0 gives 100 %. Choose ω by held-out segment likelihood, which
   `segment_log_likelihoods` supports, rather than by default.
7. **Burst-selection bias is documented, not corrected.** Segments start from
   the detection-weighted distribution; the threshold's preference for bright
   states is not modelled.

## Choosing the pairs

Which pairs to measure is decided by `ProbeNetworkSelection`
(`include/ProbeNetworkSelection.h`), and this model supplies its dynamics
term. Each pair is its own double mutant, measured on its own molecules, so
the pairs of a network share the hidden process's rates but no trajectory.
`ProbeKineticsTerm` therefore simulates bursts per candidate
(`simulate_fret_measurement`, `select_bursts`), takes the per-burst scores
(`segment_scores`) over the log-rates and the pair's state means, and keeps
the rate information left after the means are paid for, the Schur complement
`G_p = F_kk − F_kd F_dd⁻¹ F_dk`. The network's information is `F0 + Σ G_p`, and
the loss `[det F0 / det(F0 + Σ G_p)]^(1/n_k)` is the geometric-mean posterior
variance of the log-rates relative to their prior. It is mixed by weight with
Olga's structural resolution (`ProbeResolutionTerm`) and the Labelizer's site
scores (`ProbeLabellingTerm`).

With three states, any three pairs that separate them resolve the structure,
so Olga's choice among them is arbitrary for the rates. The mix spends that
freedom on the rates: on 30 random candidates, the same structural precision
with a rate variance 2.6x smaller (`examples/labels/plot_network_selection.py`).
Steady dyes only so far: with blinking dyes the rates of the dyes would have to
be paid for as well, like the means.

Homo-oligomers: `ProbeOligomerPairs` turns the dye positions of every site on
every protomer into the measurable site pairs and their distance mixtures (a
trimer's `(i, i)` row mixes three distances). In `ProbeKineticsTerm` a mixture
enters as offsets around the pair's free mean, so its spread is kept.

## The pieces

| piece | what it is |
|---|---|
| `FRETHiddenProcess` | the conformational process **shared by every pair**: K discrete conformers with a rate matrix, or a landscape u(q) with D on a path coordinate (SqRA grid) |
| `FRETDye` | one dye's photophysics as data: named states, rates between them (optionally light-driven, optionally conditioned on the hidden state), and per state an excitation, quantum yield and lifetime. For the acceptor, **accepts** (quenches the donor) is separate from **emits**: bright / dark-but-accepting (triplet, cis) / bleached are all expressible |
| `FRETInstrument` | pulses (PIE/ALEX) × channels: excitation and detection crosstalk matrices, an IRF per pulse and channel, microtime bins, gains, background rate and microtime density per channel |
| `FRETMeasurement` | one label pair: its dyes, R0, instrument, and its **distance map** from the hidden state: a distance distribution per conformer, or a spline r_p(q) with a linker spread, which may be non-monotonic |
| `FRETPhotonData` | photons (macrotime, channel, microtime bin) cut into **segments**: bursts, a whole trace, or the whole stream |
| `FRETNetworkModel` | one hidden process and N measurements: log-likelihood, exact gradient, priors (structure prior on distance maps), MAP by tttrlib's L-BFGS, Laplace with bands on u(q) and each r_p(q), per-segment occupancies of each factor, per-photon posteriors |
| `simulate_fret_measurement`, `select_bursts` | bursts with microtimes from the same model, with an optional Gaussian focus crossing, and an interphoton-time burst search that puts the selection bias into simulated data |

## Design decisions

- **Joint state by amalgamation.** (hidden × donor × acceptor) is built by
  `KineticNetwork` (a CTBN), not by Kronecker index arithmetic.
- **Emission per photon** is `Σ_j w_j λ_c(r_j) f_c(t | r_j)`: detection rate
  times microtime density, *mixed* over the state's distance distribution (the
  dye is static within one excitation cycle but moves between photons). It is
  not a product of averages. The microtime densities come from
  `PhotophysicsTransferKinetics` (donor decay under FRET, sensitised acceptor
  rise), IRF-convolved and wrapped periodically over the laser period,
  integrated per bin. Pile-up and dead time are not modelled.
- **Two arrival models.** `FRET_ARRIVAL_FULL` kills between photons with
  Λ_tot(s), so the count rate is information: valid for constant brightness
  (immobilised molecules). `FRET_ARRIVAL_CONDITIONAL` (Gopich–Szabo / H2MM)
  takes arrival times as given, uses the generator alone between photons, and
  normalises each photon's factor. It is the default for bursts. It is
  insensitive to the focus modulating the *arrival rate*, but not to the
  focus modulating the *signal-to-background ratio*: off-centre, background
  is a larger share of the photons than the per-state factors assume. See
  "Where to pick this up", item 2.
- **Propagation between photons** uses uniformization on the sparse joint
  generator: the generator is non-symmetric, block-triangular with bleaching,
  and near-defective, so the landscape model's eigenvector trick is unsafe
  here. Where it is cheaper (`8 n³ < nnz q τ`: a small state space with a fast
  rate), a **dense matrix exponential** is used instead, with Van Loan's block
  giving the exact adjoint and the occupancy.
- **Gradient**: exact adjoint with respect to generator entries, photon
  factors and the start distribution, chained to the parameters by central
  differences of the photon-free builders, in internal (log/logit)
  coordinates.
- **A fit backs off** from parameters that make no model (a reducible chain):
  they score −∞ rather than throwing.

## Validation

`examples/spectroscopy/fret_network_validation.py` (checks a–d) and the local
prototype `prototypes/fret_network_landscape/` (not in git, by the repo's
`.gitignore`), which draws them. Simulated PIE bursts, 32 microtime bins of
0.25 ns.

| check | result |
|---|---|
| a. microtimes on/off, σ at the truth | rates 1.38 vs 2.09 and 1.13 vs 1.61 ms⁻¹, distances 1.37 vs 2.67 and 1.77 vs 2.77 Å: 1.4–2× narrower |
| b. blinking acceptor vs a second conformer | without PIE and microtimes ΔlogL 0.2 (a tie); with them, 16.5 for the blinking acceptor, with one parameter fewer |
| c. one non-monotonic pair vs the joint fit | pair 1 alone: BIC picks 2 discrete states, misplaced (41, 58 Å for wells at 50, 40). Joint landscape fit of both pairs with the path prior: 95 % of the populated grid within 2σ, D 1.56 ± 0.34 (true 1.50) |
| d. focus crossing, no blinking in the truth | full arrival model: 11.8 % invented donor dark time; conditional: 0.0 % |

Tests: `test/landscape/test_fret_network_*.py`:
- the filter against scipy `expm`, on a bleaching scheme and with fast rates;
- gradients against central differences (discrete and landscape, both
  arrival models, dense path);
- occupancies and posteriors against brute force;
- conditional = full minus the arrival term;
- joint-stationary = product start without bleaching;
- a zero-width distance distribution reproduces the point model;
- the network reduces to the landscape model;
- simulated photons follow each state's channels and microtimes.

---
okf_version: "0.2"
status: implemented (native C++ since 2026-10-09)
---

# Counterfactuals in FRET modelling, and a directed factor graph

Written 2026-10-06, from a discussion with the owner. This page has two parts.
The first part makes the case that interventions and counterfactuals can make
FRET models say things they can't say today, with nine worked examples. The
second part proposes how `InferenceFactorGraph` should carry direction so it
can represent them. Both are implemented in IMP.bff C++ (section 4a). The
examples are worked
numerically in
[`doc/workshop/09_counterfactuals.ipynb`](../doc/workshop/09_counterfactuals.ipynb),
on smFRET data simulated with tttrlib (`SimEngine` photon simulation and
`sim_state_at_times` kinetics). The simulator's per-photon state log is the
ground truth every claim is checked against.

## 1. The ladder

Pearl separates three kinds of question. Each needs more model than the one
before.

| Level | Question | Needs |
|---|---|---|
| 1. Association | p(y \| x): what do I see when I see x? | A joint distribution (a factor graph) |
| 2. Intervention | p(y \| do(x)): what happens if I *set* x? | A directed graph: cut x's incoming arrows, fix x |
| 3. Counterfactual | p(y_x′ \| x, y): this molecule showed y under x. What would it have shown under x′? | A structural causal model (SCM): mechanisms plus explicit exogenous noise U |

A counterfactual is computed in three steps:

1. **Abduction:** infer this unit's noise, p(U | observed).
2. **Action:** apply do(x′).
3. **Prediction:** push the same U through the modified model.

A *twin network* draws this as a single graph: a factual copy and a
counterfactual copy of the endogenous variables that share their exogenous
parents. Counterfactual inference then becomes ordinary Bayesian inference on a
larger graph. That is why it belongs in a factor graph.

Three facts fix how counterfactuals may be used:

- **Level 1 ≠ level 2.** If closed molecules bind ligand better, then
  P(closed | bound) is larger than P(closed | do(bound)). Ensemble correlation
  is not mechanism. (In notebook cell 1, the two values are 0.66 and 0.30.)
- **Level 2 ≠ level 3.** Two SCMs can share the same arrows *and* the same
  interventional distributions, yet give opposite counterfactuals. Take
  Y = U versus Y = T xor U, with U ~ Bernoulli(½). Both give p(Y | do(T)) = ½.
  But for a unit with T = 1 and Y = 1, the first says Y would still have been 1
  under T = 0, and the second says 0. **The functional form, and the way the
  noise enters, decides the answer.**
- **So a restraint on a counterfactual is an assumption, never evidence.** In
  the graph it is a PRIOR-role factor (role `assumption`), never a LIKELIHOOD.
  Otherwise the data that informed the mechanism are counted twice.

## 2. FRET as a structural causal model

```
U_conf ──► X (conformation) ◄── L (label present / site / ligand / mutation)
                 │
U_dye  ──► D (dye positions) ◄── site, linker
                 │
                R_DA ──► E, donor decay ◄── κ², R0, Q_D(X)   (photophysics)
                               │
U_phot ───────────────────────► photons (colour, micro/macro time)
```

Every experimental handle is a do(): choosing the labelling site, labelling
donor-only, titrating ligand or quencher, making a mutant, crosslinking a
hinge. Every exogenous U is something the data can partly reveal for each
molecule (abduction). Photon noise is the U we can see best, because single-
molecule data record it photon by photon.

## 3. Nine examples

The level of each example is stated honestly. Level-3 statements are only
worth more than level-2 ones when the *same molecule* is observed. Ensemble
data can't tell them apart.

### 3.1 Real intermediate or dynamic averaging? (level 3, per burst)

- **Question:** a burst shows an intermediate E. Is it a third state, or two
  states exchanging within the burst?
- **Counterfactual:** the E and ⟨τ⟩_f this burst would have shown had the
  conformation been frozen, do(k = 0), keeping the burst's own photon noise
  and starting state.
- **Known in practice:** the *static FRET line* is the population version of
  this counterfactual. The twin network gives the per-burst version.
  Abducting each burst's state path from its photons (forward-filtering,
  backward-sampling over photon arrivals, as H2MM does) gives P(dynamic | burst)
  and a counterfactual position on the static line.
- **Notebook:**
  - **Data:** tttrlib simulates diffusing molecules with two states
    exchanging at 0.3 ms⁻¹, 17 % donor-only molecules and background, plus a
    buffer run. Bursts come from tttrlib's burst search.
  - **Inference:** all eight parameters (both efficiencies, τ₀, both rates,
    the donor-only fraction, both background rates) are inferred in an IMP.bff
    Bayesian graph (GraphPorts with priors, burst and buffer likelihood
    GraphNodes, `MCMCSampler`).
  - **Parameters:** every one is recovered except the exchange rates, which
    come out about 25 % high (0.38 against 0.30 ms⁻¹). That cause is open.
  - **Counterfactual:** abduction over posterior draws classifies 92 % of
    1056 bursts correctly as dynamic or static, and identifies 100 % of
    donor-only bursts.
  - **Sweep:** at 0.1 and 1.0 ms⁻¹, 94 % of bursts are classified correctly.
- **Data:** existing MFD bursts.
- **What it adds:** separates bursts by mechanism instead of fitting a
  population mixture, and makes "static line" a computed object rather than a
  drawn curve.

### 3.2 Allostery: which path carries the signal? (level 3, mediation)

- **Setup:** effector L at site 1. Pair 1 reads the hinge M, pair 2 reads the
  active site Y.
- **Effects:**
  - The *natural indirect effect*, Y(L=1, M(L=0)) → Y(L=1, M(L=1)), is the part
    of the allosteric response transmitted through the hinge.
  - The *natural direct effect* is the remainder.
  - Both combine two worlds for the same molecule, so they are counterfactual
    by definition (Pearl 2001).
- **Notebook (linear SCM, true NIE = 0.80, NDE = 0.30):**

  | Experimental design | What it gives |
  |---|---|
  | Pairs measured on separate molecules | The total effect only (1.10) |
  | Both pairs on the same molecule (three-colour, or trace pairs) | NIE/NDE correctly, *unless* a latent cause W moves both hinge and active site |
  | Same molecule, with such a W present | The naive estimate gives NIE = −0.16: wrong sign |
  | An intervention on the mediator (hinge-locking crosslink or mutant, do(M)) | NIE = 0.80 again |

- **What it adds:** pathway decomposition of allosteric coupling. The causal
  graph also says *which* experiment identifies it. A double-mutant cycle is
  the level-2 shadow of the same question.

### 3.3 Conformational selection or induced fit? (level 3, per binding event)

- **Data:** surface-immobilised TIRF traces in which the binding event is
  visible (labelled ligand, or a FRET jump).
- **Counterfactual:** the probability of necessity,
  PN = P(open at t_b + Δ under do(no ligand) | bound, closed at t_b + Δ).
- **Model:** conformational dynamics as a Gumbel-max SCM (Oberst & Sontag
  2019). Each step's Gumbel noise is shared between the factual and ligand-free
  worlds, and abducted from the observed transitions.
- **Notebook:** on frame traces from tttrlib's four-state kinetics, PN is
  0.14 for conformational selection and 0.80 for induced fit. The per-event distributions separate.
- **What it adds:** reads the mechanism off each binding event, so a mix of
  both mechanisms is visible as a bimodal PN, which ensemble kinetic model
  comparison averages away.

### 3.4 Is the intermediate necessary? (level 3, kinetic networks)

- **Setup:** A ⇄ I ⇄ B from H2MM or HMM.
- **Counterfactual:** for this A → B trajectory, would B have been reached by
  time t had I been blocked? Use the same Gumbel-max construction, with I
  removed in the twin.
- **What it adds:** a per-trajectory probability that the path *uses* I. That
  is stronger than "a model with I fits better", and directly comparable to
  mutants that destabilise I.

### 3.5 Distance or photophysics? (level 3, mediation)

- **Setup:** the conformation changes R_DA *and* the donor's quantum yield
  (PET quenching by Trp, sticking). Both reach E.
- **Counterfactual:** the natural direct effect,
  E(state B, Q_D as in state A), is the part of ΔE that comes from distance.
- **Data:** lifetimes per state (the donor-only and donor–acceptor decays
  already measured).
- **What it adds:** a γ correction per state derived from the graph, and a
  restraint that states explicitly "this ΔE is distance, not quenching".

### 3.6 Label perturbation as an inferred quantity (level 2)

- **Usual assumption:** "Dyes don't perturb the structure" means deleting the
  arrow L → X. Every FRET-restrained model makes this silently.
- **Proposal:** keep the arrow with strength β and prior β ~ N(0, σ). Several
  labelling sites are several do(site) interventions, so with a structural
  model β is identifiable.
- **Result:** the published structure is then the counterfactual "unlabelled"
  structure, with the perturbation uncertainty propagated.

### 3.7 The second pair, predicted for the same molecules (level 3)

- **Steps:** abduct each molecule's conformation from its pair-A burst, then
  predict what pair B would read on *those* molecules.
- **Uses:**
  - **Pair choice:** choose the B whose counterfactual readout is most
    informative given A. This is labelizer's question, asked per molecule
    (see [labelizer](../doc/labelizer.md)).
  - **Model check:** once B is measured, a mismatch between the predicted
    per-molecule joint distribution and the measured one flags a structural
    model that an ensemble fit to B alone would absorb.

### 3.8 Mutants and variants (level 2, transportability)

- **Setup:** a mutant panel is a set of do(sequence) interventions.
- **What it adds:** with mechanisms shared across mutants, the panel predicts
  populations and ligand response for unmeasured variants. Per-mutant fits
  can't do this. It isn't a counterfactual, but it needs the same directed
  graph.

### 3.9 Testing a FRET network: can a correct analysis be wrong? (level 3)

- **Question:** a network was designed, measured and fitted correctly, yet
  the conclusion can still be wrong when an assumption fails. A sticking dye,
  an AV model that misses a linker interaction, or κ² far from 2/3 all act as a
  **bias at one labelling site**, carried into every pair that uses it, and
  the fit absorbs it into the structure. Can counterfactuals catch this?
- **Answer:** not by magic. A bias the network can't tell apart from
  structure is not identifiable. But making the assumption an exogenous
  variable (one bias b_s per site, with a prior) turns it into four answerable
  questions:
  1. **Abduction:** network redundancy (each site in several pairs) makes b_s
     partly identifiable.
  2. **do(b = 0):** what the data would have been with ideal dyes, keeping
     this experiment's noise, and what the standard analysis then picks.
  3. **do(b_s = 0), one site at a time:** the site the conclusion *hinges* on
     (probability of necessity, applied to an assumption). That is the dye to
     re-measure: label a different residue nearby, or swap donor and acceptor.
  4. **Counterfactual replay:** had the structure been f′, with this
     experiment's abducted biases and noise, would the pipeline have found
     f′? This is a calibration curve for *this* data set. Simulation-based
     calibration with fresh noise (level 2) can't show it.
- **Notebook:** IMP.bff accessible volumes on the 58 frames of the recorded
  hGBP1 transition, 8 sites and 21 readable pairs. The truth is frame 45, and
  a +6 Å bias at site 18 is hidden from the analysis.
  - The standard χ² analysis reports frame 57 with P(true frame) = 0 and a
    reduced χ² of 1.8, so nothing looks wrong.
  - The bias-aware model returns frame 45 (P = 0.53) and abducts 5.2 Å at
    site 18.
  - Only do(b₁₈ = 0) restores the true frame.
  - The replay shows the network collapsing frames 42–57 onto the end state,
    whereas with fresh noise (level 2) it recovers every frame.
- **What it adds:** a network test that asks "had the structure been
  different, would this data set have shown it?" rather than "is the fit
  good?". It also gives a ranked list of which assumptions a published
  structure depends on.

### Summary

| # | Topic | Level | New data? |
|---|---|---|---|
| 3.1 | Dynamic or real intermediate, per burst | 3 | No (MFD) |
| 3.2 | Allosteric pathway (mediation) | 3 | Two pairs on the same molecule; do(M) when confounded |
| 3.3 | Conformational selection or induced fit, per event | 3 | TIRF traces with visible binding |
| 3.4 | Necessity of an intermediate | 3 | No (H2MM) |
| 3.5 | Distance or photophysics | 3 | Lifetimes per state |
| 3.6 | Label perturbation β | 2 | Several labelling sites |
| 3.7 | Second pair, same molecules | 3 | Pair A; B to check |
| 3.8 | Mutant transfer | 2 | Mutant panel |
| 3.9 | FRET network bias test | 3 | No (the existing network); re-labelling to confirm |

## 4. Graph design: direction carried by factors (implemented)

**Before 2026-10-09.** `InferenceFactorGraph` had only undirected factors
(PRIOR, LIKELIHOOD, HYPER, LINK), which it moralises into cliques. The mediation
example gives one clique {L, W, M, Y}, and so does the reversed model in which
the active site drives the hinge. Direction, and with it do() and twin
networks, could not be represented.

**Now: a mixed graph in which the factor carries the arrow.**

- `set_factor_children(factor, children)`.
  - **With children,** the factor is a conditional p(children | scope ∖
    children): a mechanism, i.e. arrows from the rest of the scope into each
    child.
  - **Without children,** it is an undirected potential, exactly as PRIOR,
    HYPER and LINK were before.
  - A variable has at most one mechanism, and children must be in the scope.
- **Bidirected edges** (a latent common cause, such as W in 3.2) need no edge
  type of their own. They are a variable with `role = "exogenous"` that is a
  parent of both sides.
- **Unchanged:** moralising a directed factor gives the same clique over its
  scope. The moral graph, elimination order, treewidth, junction tree and
  sampling blocks are identical; tests assert it.
- **Queries:** `get_parents`, `get_children`, `get_descendants`,
  `get_mechanism_of`, `get_is_acyclic`, `get_exogenous_variables`,
  `get_intervened_variables`.
- **Operations,** each returning a new graph:
  - `get_intervened(keys)` removes the mechanisms of the keys (the mutilated
    graph) and marks them held. A mechanism that generates several children
    is removed only whole.
  - `get_twin(keys, suffix="@cf")` copies the descendants of the keys and
    their mechanisms with suffixed keys, then intervenes on the copied keys.
    Exogenous ancestors stay shared, which couples the two worlds. Data
    likelihoods are not copied: the data belong to the factual world.
  - A counterfactual restraint is an ordinary PRIOR over factual and `@cf`
    keys, with role `assumption`.
- **Cycles:** kinetic schemes (A ⇄ B) are one node, a rate matrix inside
  `KineticSchemeNode`. `get_is_acyclic()` checks the variable-level graph, and
  `describe()` reports "mechanisms : n (acyclic)" or "(CYCLIC)".
- **Serialisation:** `children` per factor and `intervened` per graph in
  `to_json()`, round-tripped exactly by `from_json()`.

**Alternative considered.** Keep direction only in the `GraphNode`/`GraphPort`
evaluation graph (already a DAG of structural equations) and derive the factor
graph from it by moralisation. It was rejected as the *primary* home because
the factor graph is the object saved with data and handed to inference.

## 4a. Native counterfactual engines (C++)

| Class | What it does | Test | C++ example |
|---|---|---|---|
| `InferenceFactorGraph` (directed factors) | parents and children, acyclicity, `get_intervened`, `get_twin`, JSON | `test/factorgraph/test_directed_factors.py` | `examples/counterfactual/twin_factor_graph.cpp` |
| `CausalLinearGaussian` | linear-Gaussian SCM: interventional and conditional worlds, abduction of the noise, counterfactual worlds, natural direct and indirect effects (population or one unit), samples | `test/counterfactual/test_causal_linear_gaussian.py` | `allosteric_mediation.cpp` |
| `CounterfactualDistanceNetwork` | site biases as exogenous variables: standard and bias-aware posteriors, abducted biases and noise, do(b = 0), per-site hinge, counterfactual and fresh-noise replay | `test/counterfactual/test_counterfactual_distance_network.py` | `network_site_bias.cpp` |
| `CounterfactualMarkovChain` | Gumbel-max counterfactual trajectories of an observed Markov chain, occupancy, probability of necessity | `test/counterfactual/test_counterfactual_markov_chain.py` | `ligand_necessity.cpp` |

- **Tests:** every number is checked against an independent numpy
  construction.
- **Examples:** the four C++ examples are built and run as part of the
  module's example tests (`examples/Files.cmake`).
- **Notebook:** `doc/workshop/09_counterfactuals.ipynb` calls the same classes
  in sections 3, 4, 6 and 7. Section 2's photon model is still numpy; its C++
  home is the planned stream mode of `BurstML` (plan item 2).
- **Result:** the C++ results reproduce the notebook's earlier numpy ones
  exactly:
  - PN 0.14 for conformational selection and 0.80 for induced fit;
  - the network fit at frame 57 under the standard analysis and frame 45 with
    biases, 5.2 Å abducted at site 18;
  - natural effects 1.10 = 0.30 direct + 0.80 indirect.

## 5. Limits

- **Identifiability.** Counterfactuals are generally not identified from
  observational or interventional data. They rest on the SCM's functional
  form (3.1 assumes Markov switching with exponential microtimes; 3.3 assumes
  Gumbel-max dynamics). Results must state that model.
- **Abduction is inference.** Photon noise can't be inverted, so U has a
  posterior that needs forward-filtering, backward-sampling, sampling or
  Laplace. The ucfret no-sampler requirement means Laplace over the twin's
  dimensions.
- **Cost.** Each extra world duplicates the descendants of the intervention.
  `block_cost` and treewidth grow accordingly; the graph shows it before
  anything runs.
- **No double counting.** A counterfactual restraint is a prior. If the same
  data also enter as a likelihood, the restraint must be conditionally
  independent of them given the mechanism, or it must be dropped.

## 6. Plan

1. **Direction in `InferenceFactorGraph` and the native engines: done**
   (2026-10-09, section 4a).
2. **The burst likelihood: build on tttrlib's `BurstML`, wrap it as a bff node.**
   - **Why the time goes elsewhere today.** Measured 2026-10-06 on 500
     bursts × 150 photons with eight parameters: `MCMCSampler` costs 5.72 ms
     per evaluation against 5.76 ms for the bare numpy likelihood. The bff
     sampler and graph add nothing; the Python photon recursion in the
     notebook's `GraphNode` is the cost.
   - **What `BurstML` is.** tttrlib's `BurstML` is the Gopich–Szabo
     maximum-likelihood burst model: diffusion through the focus on a radial
     grid, kinetics between states, and photon colours with per-colour
     background. Its own header credits Hoffmann et al.; that attribution
     needs correcting in tttrlib. It is already C++, validated to 1e-12
     against the original MEX, and it makes the burst definition (gap t_th,
     minimum n_th photons) part of the likelihood.
   - **Test on the notebook's tttrlib simulation** (two states, k = 0.3 ms⁻¹
     each way, background 1 / 0.5 kHz, 947 bursts split at a 0.1 ms gap):
     - E = 0.201 / 0.796, populations 0.50;
     - exchange-rate sum 0.68 ms⁻¹, i.e. k ≈ 0.34 against 0.30 (13 % high,
       against 25 % for the notebook's model). This supports the suspicion
       that the notebook's per-burst constant background fraction, which
       ignores diffusion, causes part of its bias;
     - background under-estimated (0.32 / 0.71 against 0.5 / 1.0 kHz);
     - 107 ms per evaluation at jmax = 30, and `fit().errors` came back
       empty.
   - **What `BurstML` lacks for the graphs here.** The work belongs in
     tttrlib (photons), and bff wraps it as a likelihood `GraphNode` for
     `MCMCSampler`:
     - micro-time emission (donor lifetime per state), without which there is
       no τ₀ and no E–τ static-line counterfactual;
     - a donor-only species as a disconnected state;
     - per-burst posterior state paths (forward-filter, backward-sample in
       the eigenbasis) for the abduction step;
     - a gradient, for NUTS;
     - speed: cache the eigendecomposition across parameters that don't
       change it, and allow a smaller jmax;
     - the empty Hessian errors.

   - **The burst definition made Bayesian** (prototype, 2026-10-07,
     [`prototypes/burst_free_likelihood/`](../prototypes/burst_free_likelihood/README.md)).
     - **Model:** add a "no molecule in the focus" state with entry and exit at
       the edge of the radial grid, and write the likelihood for the whole
       photon stream as a Markov-modulated Poisson process. Background is
       emitted in every state, and donor micro-times are included.
     - **Fit:** on the same tttrlib simulation, k = 0.301 ± 0.016 and
       0.328 ± 0.018 ms⁻¹ (second seed 0.325 / 0.312), against 0.30.
       Background, efficiencies and τ₀ all lie within 1.6 sd of the truth.
       The bias of the burst-based fits (0.38 for the notebook's model, 0.34
       for `BurstML`) is gone.
     - **Photons:** the per-photon posterior assigns 98.2 % of photons
       correctly to molecule or background, against 90.4 % for a burst
       search, which misses 24 % of molecule photons.
     - **Cost:** 0.27 s per evaluation on 300 000 photons in numpy.
     - **Counterfactuals on the stream.** Posterior paths of the whole
       stream are sampled by forward-filtering, backward-sampling. Each photon
       gets a molecule-or-background label and noise uniforms, and two twins
       are run:
       - **do(k = 0):** P(dynamic) per transit is calibrated (predicted 0.20
         gives observed 0.22, 0.81 gives 0.76, 0.99 gives 0.97) and 92 %
         correct on bright transits;
       - **do(background = 0):** brings ⟨τ⟩_f from 2.9 ns off to within
         0.19 ns of the molecule's own photons. Transits defined by the
         posterior carry 16 % background on average.
       - Together they reproduce the static-line test without bursts:
         background-free static transits sit on the line, and frozen twins
         land on it.
     - **Next:** this becomes `BurstML`'s stream mode (empty state, micro-time
       emission, posterior paths), with the axial coordinate, donor-only
       molecules and detector dead time still to add.

3. **The network bias test (3.9) in network design (labelizer and Olga).**
   Surveyed 2026-10-07. Nothing in labelizer or the Olga code models a
   systematic per-site error:
   - `ProbeResolutionTerm` uses one scalar `measurement_error` for all pairs,
     with independent per-pair χ²;
   - `ProbeLabellingTerm` models the probability that labelling fails, not a
     distance bias;
   - the selector rewards reusing a site but has no minimum number of pairs
     per site.

   The data the test needs are already inside `labelizer/network.py`
   `select_network`: the model table `d`/`eff` (conformers × pairs), the
   site incidence (`pairs`, passed to `set_pair_sites`), `ensemble.rmsd`, and
   per-site keys. Hook points, in order of value:
   a. **`ProbeBiasRobustnessTerm`**, a new `ProbeNetworkTerm` in
      `include/ProbeNetworkSelection.h`. It computes Olga's expected RMSD
      under the marginal likelihood with covariance S = σ²I + τ_b² A Aᵀ:
      the site biases are integrated out, as in notebook 09. That needs a
      Mahalanobis χ² per pair of frames, not the per-pair kernels in
      `internal/ProbePairKernels.h`. `get_loss_with(pairs, sites)` already
      receives both pairs and sites.
   b. **Redundancy:** penalise sites that appear in fewer than k pairs. A site
      in one or two pairs has an unidentifiable bias (3.9). This can be a term
      or eligibility logic in `select()`; site mode already groups pairs by
      site.
   c. **Hinge and blind-spot report** after selection, in `labelizer/network.py`
      next to `mode_resolution`:
      - leave one site's bias free and report the worst-case loss;
      - replay over the candidate conformers with biases drawn from the prior
        (at design time there is no data to abduct), and report the confusion
        between true and recovered conformer;
      - shown in the app next to `rmsd_curve` (`labelizer/jobs.py`).
   d. **Per-site prior widths τ_s** from data that already exists:
      `ProbeModelComparison.h` records the AV-versus-rotamer disagreement per
      site (`d_mean_position`, `sigma_av`/`sigma_rot`, `interpenetration_weight`),
      and the dye-behaviour fields of the labelizer site features. A term can
      take them as a per-site map, as `ProbeLabellingTerm` takes label scores.
   e. **On measured data,** do(b = 0), the per-site hinge and the replay belong
      in the analysis: `InferenceFactorGraph` / FRETNetwork with per-site bias
      variables (role `exogenous`). Today they exist only in notebook 09
      section 6.

4. **Section 2 of the notebook:**
   - diagnose the 25 % overestimate of the exchange rates, testing the
     background-fraction approximation, burst selection and the photon cap;
   - then add IRF, crosstalk and γ as graph variables, and polarisation.

## References

- J. Pearl, *Direct and indirect effects*, UAI 2001; *Causality*, 2nd ed., 2009.
- M. Oberst, D. Sontag, *Counterfactual off-policy evaluation with Gumbel-max
  structural causal models*, ICML 2019.
- V. Veitch et al., *Counterfactual invariance to spurious correlations*,
  NeurIPS 2021 ([arXiv:2106.00545](https://arxiv.org/abs/2106.00545)).
- D. Ibeling, T. Icard, *Probabilistic reasoning across the causal hierarchy*,
  AAAI 2020 ([arXiv:2001.02889](https://arxiv.org/abs/2001.02889)).
- Y. Perov et al., *MultiVerse: causal reasoning using importance sampling in
  probabilistic programming*, AABI 2020
  ([PMLR 118](https://proceedings.mlr.press/v118/perov20a.html)).
- ChiRho, a causal probabilistic programming language on Pyro (multi-world
  tensors).
- A. Vlontzos et al., *Estimating categorical counterfactuals via deep twin
  networks* ([arXiv:2109.01904](https://arxiv.org/abs/2109.01904)).

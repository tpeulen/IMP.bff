---
title: ESM-2 contacts for the FRET network score -- measured against DCA, and as a dynamics signal
status: ESM-2 contacts native but no dynamics signal; the elastic network (ANM, k = 2, calibrated) is the dynamics term
updated: 2026-10-01
---

# ESM-2 contacts in the FRET network score

The probe-network selection (`ProbeNetworkSelection`) takes a per-pair cost
(`ProbePairCostTerm`, loss `1 − Π(1 − c_p)`). Following FRETNet-Designer, that
cost was meant to come from co-evolution: mean-field DCA
(`compute_sequence_coevolution`) on an MSA. ESM-2 predicts residue contacts
from its own attention maps (Rao et al. 2021; Lin et al. 2023), from the
single sequence with no MSA. It uses a logistic regression over the
symmetrised, APC-corrected attention of every layer and head
(`EsmContactPredictionHead`).

The owner asked for two uses, both as alternatives (DCA stays):
1. contact probability as an alternative source of the pair cost;
2. a *dynamics* reward: pairs predicted in contact but apart in the given
   structure, as candidates for a conformational change.

Before any code, model sizes were compared on structures, against imp.bff's
own DCA on the same chains (prototypes in
`prototypes/esm_retrieval/contacts/`: `prep.py`, `esm_contacts.py`,
`dca_run.py`, `score.py`, `score_fret.py`).

## Set

- **20 PDB chains** (chain A): 1UBQ 2LZM 3CHY 1PGA 1SHG 1HRC 1FKB 1TIM 3DFR
  1LYZ 2RN2 5P21 1GFL 1LK2, and three two-state proteins:

  | protein | open | closed |
  |---|---|---|
  | adenylate kinase (AdK) | 4AKE | 1AKE |
  | maltose-binding protein (MBP) | 1OMP | 1ANF |
  | ribose-binding protein (RBP) | 1URP | 2DRI |

- **Residues:** the entity sequence (`label_seq`); Cβ (Cα for Gly). The two
  states of a protein are matched by author residue number.
- **ESM-2:** transformers `predict_contacts` (eager attention; SDPA returns no
  attention maps), on the cordeshub GPU.
- **DCA:** imp.bff mean-field DCA on MSAs from the fast ConSurf search (up to
  2,000 homologues, the closest first, ConSurf's homologue filter). Caveat:
  these MSAs are thinner than dedicated DCA alignments, e.g. Neff 6 (1PGA),
  9 (1GFL), 39 (1LK2), 170–380 for most others. 1UBQ found no homologues,
  and 3DFR hit a transient SD-card read error; 13 chains are left for DCA.

## Contacts: long-range precision

Long range is |i − j| ≥ 24; a true contact is Cβ < 8 Å. Mean over chains of
the precision of the top L/5, L/2 and L predictions:

| source | 17 chains | the 13 with DCA | GPU, 20 chains |
|---|---|---|---|
| ESM-2 8M | 0.370 / 0.277 / 0.215 | | |
| ESM-2 35M | 0.643 / 0.499 / 0.378 | 0.632 / 0.476 / 0.365 | |
| ESM-2 150M | 0.776 / 0.657 / 0.524 | 0.749 / 0.617 / 0.500 | 0.57 s |
| **ESM-2 650M** | **0.864 / 0.749 / 0.582** | **0.843 / 0.715 / 0.553** | 0.81 s |
| DCA (these MSAs) | | 0.377 / 0.238 / 0.154 | search ~60 s + DCA 1–37 s |

On the same chains, ESM-2 650M finds more than twice as many true long-range
contacts as DCA on ConSurf-style MSAs, without an MSA, and even 35M beats
DCA. Precision grows with size up to 650M. 150M is 10 points lower, so
**650M is the choice for contacts** (the search keeps 35M).

## Dynamics: negative

**Test 1:** among pairs apart in one state (Cβ > 8 Å, |i − j| ≥ 6), are those
in contact in the other state ranked high? AUC:

| score | AdK o→c / c→o | MBP o→c / c→o | RBP o→c / c→o |
|---|---|---|---|
| ESM-2 650M contact probability | 0.84 / 0.99 | 0.92 / 0.92 | 0.96 / 0.97 |
| **distance in the input structure alone** | 0.84 / 0.99 | 0.99 / 0.99 | 0.95 / 0.99 |
| DCA | — / 0.61 | 0.42 / 0.58 | 0.49 / 0.58 |

The contacts specific to one state are mostly near-contacts the other state
almost has, so the structure's own distance predicts them as well as ESM-2
does. Beyond 12 Å the counts are small (AdK n = 31: ESM 0.73, distance 0.70).

**Test 2 (the FRET question):** which pairs change distance by more than 5 Å
between the states? The score is the planned "predicted but unsatisfied
contact" `p · σ((d − 8)/2)`: AUC 0.27–0.36 for every model size, and 0–5 % of
the top 20 change against a base rate of 11–30 %. That is worse than chance.
The large distance changes of these domain closures happen between residues
ESM-2 correctly predicts are not in contact. The contact maps describe the
fold, not the motion.

**Decision:** the dynamics reward is not built. ESM-2 contacts are added
only as the alternative pair cost. A dynamics term would need a different
signal (e.g. elastic network normal modes from the structure), which is a
separate question.

## Native contacts (650M) and what they mean as a pair cost

`ProteinLanguageModel::get_contacts` on ESM-2 650M
(`/Volumes/SD1TB/sequences/esm2_650m_contacts.gguf`, 1.3 GB, F16) agrees
with transformers to 0.0045 at most over the 20 chains. Precision is
identical (0.864 / 0.749 / 0.582). On the CPU it takes a median 0.57 s per
chain, 1.25 s at 370 residues (imp.bff `f6fe22d59`).

**As the probe-pair cost it changes little.** On MBP (1OMP) the candidate
FRET pairs are the 47,060 pairs with Cβ 20–80 Å and |i − j| ≥ 24. Their
contact probabilities have a median of 0.006; 9 exceed 0.1 and 3 exceed 0.5.
Pairs at FRET distances are not in contact, and ESM-2 says so correctly. DCA
on the same pairs ranks differently (Spearman 0.19), and the two share 0 of
their top 50 flagged pairs (1 of 200).

FRETNet-Designer's penalty means "the two sites are coupled, so mutating
both may disturb the function". At FRET distances that coupling is
long-range (evolutionary or allosteric), not physical contact, and a contact
head predicts physical contact only. `probe_pair_contacts` therefore works
as a fast, MSA-free cost (1.7 s against about 80 s for search plus DCA on
MBP), but with nearly zero for every FRET-range pair it barely steers the
selection. It is not a substitute for the meaning of the co-evolution term.
DCA remains the choice when that meaning is wanted.

Where ESM-2 is more promising for FRET design, not tested here:
- **Site tolerance:** the masked-marginal `log p(C) − log p(wild type)` per
  site, i.e. how well a cysteine (the label site) is tolerated. That is a
  site score for `ProbeLabellingTerm`, not a pair cost.
- **Long-range couplings:** the "categorical Jacobian" (Zhang et al., PNAS
  2024) gives an ESM-2 coupling map not limited to contacts. It is closer in
  meaning to DCA but costs 20·L forward passes.

## Elastic network: the dynamics signal that works (2026-10-01)

The anisotropic network model (ANM) from one state's Cα: 15 Å springs; the
distance fluctuation of each pair in the k softest modes,
`σ² = Σ_m (ê_ij · (u_j − u_i))² / λ_m`. The test is the same as for ESM-2:
which pairs change by more than 5 Å in the other state (prototypes
`enm.py`, `enm_variants.py`, `enm_heldout.py`, `enm_calibrate.py`).

Choosing k on AdK, MBP and RBP, scored from both states (6 cases), at 5 Å:

| score | mean AUC (min) | top 20 changing, mean (min) |
|---|---|---|
| k = 1 | 0.81 (0.69) | 0.89 (0.55) |
| **k = 2** | **0.90 (0.76)** | **0.97 (0.85)** |
| k = 3 | 0.90 (0.76) | 0.66 (0.00) |
| 10 modes, 1/λ² | 0.90 (0.77) | 0.60 (0.00) |
| *ESM-2 contacts (for comparison)* | *0.27–0.36* | *0–0.05* |

On these three, the plain input distance gets AUC 0.66–0.91. With k ≥ 3 the
top of the list can be taken over by subdomain motions of 3–3.7 Å (not
termini: e.g. MBP residues 312–326), below the 5 Å threshold.

**Held out:** GlnBP 1GGG/1WDN, LAO-BP 2LAO/1LST, lactoferrin 1LFH/1LFG,
guanylate kinase 1EX6/1EX7. At 5 Å the AUC stays 0.81–0.85 for every
variant, and about 80 % of the top 20 change (base rate 17 %). k = 2 drops
from 0.97 to 0.76 in the top 20 and is no longer best, so choosing it was
partly fitted to the first three. Pooled over all seven proteins, k = 1 and
k = 2 are equal in the top 20 and k = 2 is better in AUC. It stays the
default, and `n_modes` is a parameter.

**Calibration to a probability:** P(|Δd| > 5 Å) = σ(a·x + b), with
x = clip(log(σ²/σ²₉₅), −8, 4), the fluctuation relative to the structure's
own 95th percentile over pairs |i − j| ≥ 6. It is fitted with each protein
weighted once and checked leaving one protein out:

| held out | AUC | mean p | observed | top 20 | p > 0.5: share changing |
|---|---|---|---|---|---|
| AdK | 0.87 | 0.13 | 0.30 | 1.00 | 0.98 |
| MBP | 0.96 | 0.18 | 0.11 | 1.00 | 0.81 |
| RBP | 0.90 | 0.21 | 0.10 | 0.95 | 0.48 |
| GlnBP | 0.86 | 0.19 | 0.22 | 0.50 | 0.68 |
| LAO-BP | 0.87 | 0.18 | 0.17 | 0.80 | 0.72 |
| lactoferrin | 0.70 | 0.17 | 0.19 | 0.55 | 0.33 |
| GK | 0.93 | 0.15 | 0.11 | 1.00 | 0.65 |

All seven: **a = 0.666, b = 0.295** (k = 2). Lactoferrin is the known
failure: its softest modes move the two lobes against each other, while
the observed change is the cleft of one lobe.

**In imp.bff:**
- `ElasticNetworkModes(coordinates, cutoff)`;
- `get_pair_distance_fluctuations(modes, pairs, n_modes)`;
- `get_pair_change_probabilities(modes, pairs, n_modes, a, b)`;
- `ProbePairBenefitTerm(probabilities)`: loss `Π(1 − d_p)`, 1 with nothing
  selected, mixed by weight with the resolution, kinetics, labelling and
  cost terms in `ProbeNetworkSelection`.

## Next

- Native contacts and `probe_pair_contacts`: done, see above.
- **1UBQ:** 0 homologues in the fast search. Suspected cause: ubiquitin's
  near-identical members fill the candidate cap and are removed by the 95 %
  identity filter. Check against the k-mer path.

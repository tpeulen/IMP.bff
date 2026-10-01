---
title: ESM-2 contacts for the FRET network score -- measured against DCA, and as a dynamics signal
status: Phase A done (model comparison); native contacts and the pair cost next
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

## Next

- **Native contacts:** `ProteinLanguageModel::get_contacts` on ESM-2 650M.
  The converter keeps `contact_head`, and the result is checked against
  transformers' `predict_contacts`.
- **Pair cost:** `get_probe_pair_contact_costs(contacts, residue pairs)` gives
  costs for `ProbePairCostTerm`, one source alongside DCA.
- **1UBQ:** 0 homologues in the fast search. Suspected cause: ubiquitin's
  near-identical members fill the candidate cap and are removed by the 95 %
  identity filter. Check against the k-mer path.

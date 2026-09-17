.. _workshop:

=====================================================
Workshop: from a structure to a measured ensemble
=====================================================

Eight steps, and the question they answer together is the one a FRET
experiment has to answer: which residues to label, what they will report, and
what the measured distances say about the structure.

Two molecules, each where it belongs. **hGBP1** carries the single-chain
steps, and its conformational transition is a recorded trajectory of 58
frames, so the ensemble the measurements have to resolve is real rather than
interpolated — the notebooks animate it. **BmrA**, a homodimeric ABC
transporter, carries the two steps that need two protomers: choosing sites
when one cysteine mutation labels every protomer, and docking one protomer
against the other.

Every notebook is executed against the build before it is published, and the
numbers in them come from those runs. What is measured and what is simulated
is set out in :doc:`README <README>`: the structures, volumes, rotamers,
scores, selection, docking and inversion are real computations; the measured
distances that ship with hGBP1 belong to its dimer and cannot be used against
the single-chain topology, which notebook 07 demonstrates rather than asserts.

.. toctree::
   :maxdepth: 1

   README
   01_accessible_volumes
   02_rotamers
   03_labelizer
   03b_labelizer_pairs
   04_pair_selection_olga
   05_site_selection_homodimer
   06_docking_homodimer
   07_distance_networks
   08_ensemble_maxent

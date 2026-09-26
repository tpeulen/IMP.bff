/*
 * Greedy selection of a probe network by a weighted mix of scores:
 * structural resolution (Olga), dynamics (rate information), labelling
 * (Labelizer scores). Terms are IMP::Objects held by the selector; there is
 * no director, because a term is evaluated inside the selector's inner loop.
 *
 * Comes after the FRET network (the kinetics term simulates and scores with
 * it) and after the Labelizer, whose `labelizer_combined_by_key` dict feeds
 * the labelling term. The frame and site arrays reuse the typemaps of
 * IMP_bff.probepairselection.i; `(out_matrix, n_out_rows, n_out_cols)` is
 * core.i's.
 */

IMP_SWIG_OBJECT(IMP::bff, ProbeNetworkTerm, ProbeNetworkTerms);
IMP_SWIG_OBJECT(IMP::bff, ProbeResolutionTerm, ProbeResolutionTerms);
IMP_SWIG_OBJECT(IMP::bff, ProbeKineticsTerm, ProbeKineticsTerms);
IMP_SWIG_OBJECT(IMP::bff, ProbeLabellingTerm, ProbeLabellingTerms);
IMP_SWIG_OBJECT(IMP::bff, ProbeNetworkSelection, ProbeNetworkSelections);

%feature("compactdefaultargs") IMP::bff::ProbeResolutionTerm::ProbeResolutionTerm;
%feature("compactdefaultargs") IMP::bff::ProbeKineticsTerm::ProbeKineticsTerm;
%feature("compactdefaultargs") IMP::bff::ProbeLabellingTerm::ProbeLabellingTerm;
%feature("compactdefaultargs") IMP::bff::ProbeNetworkSelection::ProbeNetworkSelection;
%feature("compactdefaultargs") IMP::bff::ProbeNetworkSelection::select;

%apply(double* IN_ARRAY2, int DIM1, int DIM2) {
    (double* predicted_measurements, int n_frames, int n_pairs),
    (double* rmsds, int n_rmsd_rows, int n_rmsd_cols),
    (double* state_distances, int n_states, int n_pairs)
};
%apply(int* IN_ARRAY2, int DIM1, int DIM2) {
    (int* pair_sites, int n_site_rows, int n_site_cols)
};
%apply(int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int** out_units, int* n_out_units)};
%apply(double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double** out_losses, int* n_out_losses)};
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};

// Implementation hooks, called by reset() and commit().
%ignore IMP::bff::ProbeNetworkTerm::do_reset;
%ignore IMP::bff::ProbeNetworkTerm::do_commit;

%include "IMP/bff/ProbeNetworkSelection.h"

/*
 * Anisotropic elastic network modes of a structure, and from them the pairs
 * likely to change distance: the dynamics signal for choosing FRET pairs.
 * Before the probe-network selection, whose benefit term takes them.
 */

IMP_SWIG_VALUE(IMP::bff, ElasticNetworkModes, ElasticNetworkModesList);

%ignore IMP::bff::ElasticNetworkModes::ElasticNetworkModes();
%feature("compactdefaultargs") IMP::bff::ElasticNetworkModes::ElasticNetworkModes;
%feature("compactdefaultargs") IMP::bff::get_pair_distance_fluctuations;
%feature("compactdefaultargs") IMP::bff::get_pair_change_probabilities;

%apply(double* IN_ARRAY2, int DIM1, int DIM2) {
    (double* coordinates, int n_points, int n_dim)
};
%apply(int* IN_ARRAY2, int DIM1, int DIM2) {
    (int* pairs, int n_pair_rows, int n_pair_cols)
};
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};

%include "IMP/bff/ElasticNetwork.h"

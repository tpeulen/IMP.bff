/* A residue's backbone-conditioned side-chain rotamer distribution, and the
 * quenching-centre ensembles built over it. */
IMP_SWIG_VALUE(IMP::bff, ProteinSidechainDunbrackRotamers,
               ProteinSidechainDunbrackRotamersList);
IMP_SWIG_VALUE(IMP::bff, QuencherRotamerEnsemble, QuencherRotamerEnsembles);
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};
%attribute(IMP::bff::QuencherRotamerEnsemble, int, n_rotamers, size);
%include "IMP/bff/structure/ProteinSidechainRotamerLibrary.h"

/*
 * A TCSPC decay as a node: the multi-exponential model curve a
 * time-correlated instrument produces, so a lifetime fit joins a parse fit
 * on the graph instead of returning to Python once per iteration.
 *
 * The kernels are tttrlib's, taken header-only from the vendored copy of
 * DecayConvolution.h; this class is the graph around them, exactly as
 * GraphExpression is the graph around tttrlib's expression engine.
 */
%apply(double* IN_ARRAY1, int DIM1) {(double* in_response, int n_response)};
%apply(double* IN_ARRAY1, int DIM1) {(double* in_data_y, int n_data_y)};
%apply(double* IN_ARRAY1, int DIM1) {(double* in_data_ey, int n_data_ey)};
%apply(double* IN_ARRAY1, int DIM1) {(double* in_table, int n_table)};
%apply(double* IN_ARRAY1, int DIM1) {(double* in_pattern, int n_pattern)};
%shared_ptr(IMP::bff::TCSPCDecay);
%include "IMP/bff/TCSPCDecay.h"

/* A sampled curve convolved with a measured response -- generic, and what
   feeds TCSPCDecay's instrument stage when the model is already a decay. */
%shared_ptr(IMP::bff::Convolution);
%include "IMP/bff/Convolution.h"

/* A generalized-normal peak on an even axis: a response nobody measured. */
%shared_ptr(IMP::bff::GeneralizedNormalCurve);
%include "IMP/bff/GeneralizedNormalCurve.h"

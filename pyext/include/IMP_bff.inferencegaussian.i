/*
 * Exact linear-Gaussian inference (aGrUM's CLG canonical forms and variable
 * elimination, okf/prds/prd-151.md): a Gaussian factor in canonical form
 * (K, h, g) over a named scope, and variable elimination over an
 * InferenceFactorGraph whose factors carry such forms.
 *
 * Array outputs are named `(out_view, n_out_view)`, which the managed
 * ARGOUTVIEWM_ARRAY1 pair in types.i claims: each is an ndarray owning its
 * buffer. Matrices come back flat and row-major; `.reshape(d, d)` gives K or
 * the covariance, `.reshape(d, -1)` the null-space basis (one column per
 * unconstrained direction).
 */
%include "IMP/bff/InferenceCanonicalForm.h"
%include "IMP/bff/InferenceGaussianElimination.h"

%extend IMP::bff::InferenceCanonicalForm {
  std::string __repr__() const { return self->get_description(); }
}

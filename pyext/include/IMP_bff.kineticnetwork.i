/*
 * Kinetic schemes composed from factors (aGrUM's CTBN amalgamation, PRD-150):
 * conditional intensity matrices, their amalgamation into the joint generator
 * `K[target, source]`, and the stationary and transient distributions.
 *
 * Array outputs are named `(out_view, n_out_view)`, which the managed
 * ARGOUTVIEWM_ARRAY1 pair in types.i claims: each is an ndarray owning its
 * buffer. The generator comes back flat; `.reshape(n, n)` gives
 * `K[target, source]`.
 */
%include "IMP/bff/KineticNetwork.h"

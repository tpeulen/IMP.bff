# xsf (vendored, header-only)

From https://github.com/scipy/xsf at e998115 (2026-10-06): SciPy's
special-function library, the C++ that `scipy.special` itself calls (Cephes
ports, gamma family, Bessel, Fresnel, erf, ...). BSD 3-Clause, see LICENSE.

Used **privately** by `src/Numerics.cpp` only -- never from a public header,
so nothing here is installed and no consumer sees an xsf symbol. It is here so
bff's special functions agree with scipy to the bit rather than to Boost's
tolerance, which is what lets ChiSurf drop scipy without refitting anything.

Refresh: replace `include/` wholesale from a new xsf checkout and rerun
`test/numerics/test_special_functions.py`. Do not edit files here; fix
upstream.

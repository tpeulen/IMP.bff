# Public headers of the `util` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_util_headers "BffSettings.h;CommandLine.h;ComputeBackend.h;IMPCompatibility.h;Numerics.h;Odeint.h;OpenMP.h;Registry.h;SparseLinearAlgebra.h;SpecialFunctions.h")
# Sources without a public header of their own.
set(imp_bff_util_private_sources "")

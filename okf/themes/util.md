# util: Utilities: settings, command line, numerics, registry

Settings, the `imp_bff` command-line dispatcher, compute-backend selection, OpenMP, the plug-in registry, special functions, ODE integration (`Odeint`) and sparse linear algebra.

- **Inputs:** none: numbers and strings.
- **Relations:** Everything else. Must not include `probe`, `structure` or `imp` headers (`test/test_theme_headers.py`).
- **Layout:** sources in `src/util/`, public headers in `include/util/` (reached as `<IMP/bff/util/X.h>`; `<IMP/bff/util.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `BffSettings.h`, `CommandLine.h`, `ComputeBackend.h`, `IMPCompatibility.h`, `Numerics.h`, `Odeint.h`, `OpenMP.h`, `Registry.h`, `SparseLinearAlgebra.h`, `SpecialFunctions.h`.

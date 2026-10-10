# util: Utilities: settings, command line, numerics, registry

Settings, the `imp_bff` command-line dispatcher, compute-backend selection, OpenMP, the plug-in registry, special functions, ODE integration (`Odeint`) and sparse linear algebra.

- **Inputs:** none: numbers and strings.
- **Relations:** Everything else. Must not include `probe`, `structure` or `imp` headers (`test/test_theme_headers.py`).
- **Layout:** sources in `src/util/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/util/Headers.cmake`.

Public headers: `BffSettings.h`, `CommandLine.h`, `ComputeBackend.h`, `IMPCompatibility.h`, `Numerics.h`, `Odeint.h`, `OpenMP.h`, `Registry.h`, `SparseLinearAlgebra.h`, `SpecialFunctions.h`.

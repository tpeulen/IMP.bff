Place the private header files in this directory. They will be
available to your code with

     #include <IMP/bff/internal/myheader.h>

All headers should include `IMP/bff/bff_config.h` as their
first include and surround all code with `IMPBFF_BEGIN_INTERNAL_NAMESPACE`
and `IMPBFF_END_INTERNAL_NAMESPACE` to put it in the
IMP::bff::internal namespace and manage compiler warnings.

Vendored copies live here too and are the exception to the rule above: they
keep their upstream namespaces and include guards so they can be refreshed by
a plain `cp`. `pcg_*.h` (PCG random numbers), `json.h` (nlohmann/json), and
`LatticeDiffusion.h` — the masked-lattice diffusion solver and its adjoint
behind `DiffusionSolver.cpp` — shared with tttrlib
(`../tttrlib/modules/math/include/` is the source; never edit the copies;
`test/test_vendored_headers.py` fails when they diverge). It honours a
namespace macro (`TTTRLIB_LATTICE_NAMESPACE`), so a bff source may define it as
`IMP::bff::internal` before including, as `DiffusionSolver.cpp` does.

The neural network is bff's own, not a copy: `MlpCore.h` (the differentiable
MLP kernels, the `bff.neural_net` document tree and ONNX import;
`NetworkDocument.h` is the one msgpack encoder/decoder of that document) and
`AdamUpdate.h` (the one Adam step) live in `IMP::bff::internal` and change
here without following tttrlib.

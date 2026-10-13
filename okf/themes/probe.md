# probe: Dye probes: accessible volumes, rotamers, diffusion, FPS

Dye probes on a structure: accessible volumes, rotamer libraries and sampling, diffusion simulation, FPS, path maps, contact potentials and the Brownian walk.

- **Inputs:** coordinates.
- **Relations:** The coordinate-level heart of imp.bff; the `imp/` layer wraps it for IMP.
- **Layout:** sources in `src/probe/`, public headers in `include/probe/` (reached as `<IMP/bff/probe/X.h>`; `<IMP/bff/probe.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `BrownianWalk.h`, `ContactPotentials.h`, `DensityGrid.h`, `DiffusionSolver.h`, `ElasticNetwork.h`, `FPS.h`, `FPSRotamer.h`, `GridDiffusionSolver.h`, `Linker.h`, `NPS.h`, `OccupancyGrid.h`, `PathMap.h`, `ProbeAccessibleVolume.h`, `ProbeAccessibleVolumeBuilder.h`, `ProbeAccessibleVolumeDefaults.h`, `ProbeComponentTemplate.h`, `ProbeDataPaths.h`, `ProbeDiffusionSimulation.h`, `ProbeForceFieldCIF.h`, `ProbeLibrary.h`, `ProbeNetworkSelection.h`, `ProbePairSelection.h`, `ProbePotentialTables.h`, `ProbeRestraints.h`, `ProbeRotamer.h`, `ProbeRotamerLibrary.h`, `ProbeSampling.h`, `ProbeSimulation.h`, `ProbeTopology.h`, `RRT.h`.

Sources without a public header of their own: `CommandLineFpsDistance.cpp`, `DiffusionSolverKrylov.cpp`.

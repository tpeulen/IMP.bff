# probe: Dye probes: accessible volumes, rotamers, diffusion, FPS

Dye probes on a structure: accessible volumes, rotamer libraries and sampling, diffusion simulation, FPS, path maps, contact potentials and the Brownian walk.

- **Inputs:** coordinates.
- **Relations:** The coordinate-level heart of imp.bff; the `imp/` layer wraps it for IMP.
- **Layout:** sources in `src/probe/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/probe/Headers.cmake`.

Public headers: `BrownianWalk.h`, `ContactPotentials.h`, `DensityGrid.h`, `DiffusionSolver.h`, `ElasticNetwork.h`, `FPS.h`, `FPSRotamer.h`, `GridDiffusionSolver.h`, `Linker.h`, `NPS.h`, `OccupancyGrid.h`, `PathMap.h`, `ProbeAccessibleVolume.h`, `ProbeAccessibleVolumeBuilder.h`, `ProbeComponentTemplate.h`, `ProbeDataPaths.h`, `ProbeDiffusionSimulation.h`, `ProbeForceFieldCIF.h`, `ProbeLibrary.h`, `ProbeNetworkSelection.h`, `ProbePairSelection.h`, `ProbePotentialTables.h`, `ProbeRestraints.h`, `ProbeRotamer.h`, `ProbeRotamerLibrary.h`, `ProbeSampling.h`, `ProbeSimulation.h`, `ProbeTopology.h`, `RRT.h`.

Sources without a public header of their own: `CommandLineFpsDistance.cpp`, `DiffusionSolverKrylov.cpp`.

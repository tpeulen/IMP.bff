# structure: Structure and trajectory I/O, selections, clustering, surfaces

Structure, trajectory and hierarchy I/O (PDB, mmCIF, Mol2, RMF), selection expressions, clustering, solvent-accessible surface, rotamer scoring, Z-matrices, ConSurf and the MMFDB profile.

- **Inputs:** coordinates, trajectories.
- **Relations:** Also owns the command groups for RMSD, potentials and trajectories.
- **Layout:** sources in `src/structure/`, public headers in `include/structure/` (reached as `<IMP/bff/structure/X.h>`; `<IMP/bff/structure.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `Clustering.h`, `Consurf.h`, `DockingPrecision.h`, `HierarchyFrame.h`, `MMFDBProfile.h`, `Mol2IO.h`, `MolecularGraph.h`, `ProteinSidechainRotamerLibrary.h`, `RmfIO.h`, `RotamerScoring.h`, `SelectionExpression.h`, `SolventAccessibleSurface.h`, `StructureIO.h`, `StructureTable.h`, `TrajectoryAnalysis.h`, `TrajectoryIO.h`, `VdwRadii.h`, `ZMatrix.h`.

Sources without a public header of their own: `CommandLinePotentials.cpp`, `CommandLineRmsd.cpp`, `CommandLineTrajectory.cpp`.

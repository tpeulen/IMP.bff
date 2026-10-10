# Public headers of the `structure` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_structure_headers "Clustering.h;Consurf.h;DockingPrecision.h;HierarchyFrame.h;MMFDBProfile.h;Mol2IO.h;MolecularGraph.h;ProteinSidechainRotamerLibrary.h;RmfIO.h;RotamerScoring.h;SelectionExpression.h;SolventAccessibleSurface.h;StructureIO.h;StructureTable.h;TrajectoryAnalysis.h;TrajectoryIO.h;VdwRadii.h;ZMatrix.h")
# Sources without a public header of their own.
set(imp_bff_structure_private_sources "CommandLinePotentials.cpp;CommandLineRmsd.cpp;CommandLineTrajectory.cpp")

# Public headers of the `sequence` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_sequence_headers "SequenceAlignment.h;SequenceClusters.h;SequenceCoevolution.h;SequenceConservation.h;SequenceDatabase.h;SequenceHomologs.h;SequenceMSA.h;SequenceSearch.h;SequenceServer.h")
# Sources without a public header of their own.
set(imp_bff_sequence_private_sources "CommandLineSequence.cpp")

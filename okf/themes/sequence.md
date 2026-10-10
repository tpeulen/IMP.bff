# sequence: Sequences: alignment, MSA, conservation, coevolution

Sequence analysis: alignment, MSA, search, homologs, clusters, conservation and coevolution, sequence database and server.

- **Inputs:** sequences.
- **Relations:** Feeds `structure` (ConSurf) and `learn`.
- **Layout:** sources in `src/sequence/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/sequence/Headers.cmake`.

Public headers: `SequenceAlignment.h`, `SequenceClusters.h`, `SequenceCoevolution.h`, `SequenceConservation.h`, `SequenceDatabase.h`, `SequenceHomologs.h`, `SequenceMSA.h`, `SequenceSearch.h`, `SequenceServer.h`.

Sources without a public header of their own: `CommandLineSequence.cpp`.

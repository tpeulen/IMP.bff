# sequence: Sequences: alignment, MSA, conservation, coevolution

Sequence analysis: alignment, MSA, search, homologs, clusters, conservation and coevolution, sequence database and server.

- **Inputs:** sequences.
- **Relations:** Feeds `structure` (ConSurf) and `learn`.
- **Layout:** sources in `src/sequence/`, public headers in `include/sequence/` (reached as `<IMP/bff/sequence/X.h>`; `<IMP/bff/sequence.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `SequenceAlignment.h`, `SequenceClusters.h`, `SequenceCoevolution.h`, `SequenceConservation.h`, `SequenceDatabase.h`, `SequenceHomologs.h`, `SequenceMSA.h`, `SequenceSearch.h`, `SequenceServer.h`.

Sources without a public header of their own: `CommandLineSequence.cpp`.

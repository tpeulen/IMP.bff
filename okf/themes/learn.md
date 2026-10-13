# learn: Learned models: neural nets, embeddings, protein language model

Learned components: dense networks (`NeuralNet`, training, ONNX/safetensors), the embedding index, the HMM surrogate and the native ESM-2 protein language model.

- **Inputs:** sequences, feature vectors.
- **Relations:** Standalone; consumed by `sequence` and `fret`.
- **Layout:** sources in `src/learn/`, public headers in `include/learn/` (reached as `<IMP/bff/learn/X.h>`; `<IMP/bff/learn.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `EmbeddingIndex.h`, `HMMSurrogate.h`, `NeuralNet.h`, `NeuralNetTraining.h`, `ProteinLanguageModel.h`.

# learn: Learned models: neural nets, embeddings, protein language model

Learned components: dense networks (`NeuralNet`, training, ONNX/safetensors), the embedding index, the HMM surrogate and the native ESM-2 protein language model.

- **Inputs:** sequences, feature vectors.
- **Relations:** Standalone; consumed by `sequence` and `fret`.
- **Layout:** sources in `src/learn/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/learn/Headers.cmake`.

Public headers: `EmbeddingIndex.h`, `HMMSurrogate.h`, `NeuralNet.h`, `NeuralNetTraining.h`, `ProteinLanguageModel.h`.

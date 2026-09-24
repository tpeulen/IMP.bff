/*
 * Training a dense network: the options, the result with its loss curves,
 * and the two entry points. The msgpack document train_neural_net returns is the
 * one IMP.bff.NeuralNet evaluates (IMP_bff.core.i).
 */
%{
#include <IMP/bff/NeuralNetTraining.h>
%}

IMP_SWIG_VALUE(IMP::bff, NeuralNetTrainOptions, NeuralNetTrainOptionsList);
IMP_SWIG_VALUE(IMP::bff, NeuralNetTraining, NeuralNetTrainings);
/* Assignable from a Python list: without this SWIG's setter wants a
   VectorInt proxy. */
%naturalvar IMP::bff::NeuralNetTrainOptions::hidden_layer_sizes;
%naturalvar IMP::bff::NeuralNetTrainOptions::activation;

%include "IMP/bff/NeuralNetTraining.h"

%pythoncode %{
def train_neural_net_arrays(X, Y, options=None):
    """Fit a network to 2D arrays `X` (samples x features) and `Y`.

    A convenience over `train_neural_net`: shapes are read from the arrays
    and a 1D `Y` is one target. Returns the `NeuralNetTraining`, whose
    `get_network()` is the `bff.neural_net` msgpack document (`bytes`).
    """
    import numpy as np
    X = np.ascontiguousarray(X, dtype=float)
    Y = np.ascontiguousarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X must be (n, n_features) and Y (n,) or (n, n_targets)")
    if options is None:
        options = NeuralNetTrainOptions()
    return train_neural_net_with_history(
        X.ravel().tolist(), X.shape[0], X.shape[1],
        Y.ravel().tolist(), Y.shape[1], options)
%}

/**
 *  \file IMP/bff/NeuralNetTraining.h
 *  \brief Fitting a dense network to data: Adam, minibatches, early stopping.
 *
 * The counterpart of NeuralNet.h: that header evaluates a `bff.neural_net`
 * msgpack document, this one writes it. The forward and backward passes are bff's
 * `internal/MlpCore.h` (the portable GEMM), the optimiser is
 * `internal/AdamUpdate.h`, and the random numbers are the vendored pcg32, so
 * a given seed gives the same network on every platform and build.
 *
 * Ported from tttrlib's `NeuralNet::train` (tttrlib 2026-09, before tttrlib
 * dropped its ML code): the same scikit-learn `MLPRegressor` defaults and the
 * same numerics -- Glorot-uniform initialisation, half mean squared error,
 * L2 `alpha` on the weights only, Adam, shuffled minibatches, and early
 * stopping on a held-out split that keeps the best-validation weights.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_NEURALNETTRAINING_H
#define IMPBFF_NEURALNETTRAINING_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/NeuralNet.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Hyper-parameters for train_neural_net().
/*! The defaults are scikit-learn's `MLPRegressor` (Adam, ReLU, L2 `alpha`,
    early stopping on a held-out split), so a network trained here and one
    trained in Python are comparable. Every field is public and plain.
 */
class IMPBFFEXPORT NeuralNetTrainOptions {
 public:
  //! Widths of the hidden layers, input side first; non-positive entries are skipped.
  std::vector<int> hidden_layer_sizes = {256, 256, 128};
  //! Hidden-layer nonlinearity by scikit-learn's name: `relu`, `tanh`,
  //! `logistic`, `identity`, or `softplus`, `silu`, `sin`. The output layer
  //! is always linear (this is a regressor).
  std::string activation = "relu";
  //! Maximum number of epochs.
  int max_iter = 800;
  //! Rows per minibatch.
  int batch_size = 200;
  //! Adam step size.
  double learning_rate = 1e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  //! L2 penalty on the weights (not the biases).
  double alpha = 1e-4;
  //! Hold out `validation_fraction` of the rows (when there are at least 10)
  //! and stop once the held-out loss has not improved by `tol` for
  //! `n_iter_no_change` epochs; the best held-out weights are kept.
  bool early_stopping = true;
  double validation_fraction = 0.1;
  int n_iter_no_change = 10;
  //! Minimum improvement of the held-out loss that resets the patience counter.
  double tol = 1e-4;
  //! Seeds the initial weights, the split and the minibatch order (and, in
  //! FP4 training, the stochastic rounding).
  int seed = 0;
  //! Arithmetic of the training GEMMs: `"float64"` (default), or `"nvfp4"` /
  //! `"mxfp4"` -- NVIDIA's NVFP4 pretraining recipe (arXiv 2509.25149) on
  //! bff's integer FP4 kernels: fprop, dgrad and wgrad of every FP4 layer in
  //! FP4, 2-D 16 x 16 weight scaling, a random Hadamard transform on the
  //! wgrad inputs, stochastic rounding of the gradients, float64 master
  //! weights and Adam state (`internal/MlpFp4Train.h` states the recipe).
  //! The result also carries the FP4 inference network
  //! (NeuralNetTraining::get_quantized_network()).
  //! `"ternary"`: BitNet b1.58's quantisation-aware training -- every
  //! ternary layer's forward pass on ternary weights (absmean per tensor)
  //! and int8 activations (absmax per row) through the ternary x int8
  //! kernel, the backward pass by the straight-through estimator in
  //! float64, float64 master weights and Adam state
  //! (`internal/MlpTernaryTrain.h`); the result carries the ternary
  //! network (format "ternary").
  std::string precision = "float64";
  //! FP4 only: keep the first / last layer in float64 (the paper keeps a few
  //! sensitive layers in higher precision, most of them at the end).
  bool fp4_keep_first_layer = true;
  bool fp4_keep_last_layer = true;
  //! FP4 only: the random Hadamard transform of the wgrad inputs.
  bool fp4_hadamard = true;
  //! FP4 only: stochastic rounding of the gradients (else round to nearest even).
  bool fp4_stochastic_rounding = true;
  //! Ternary only: keep the first / last layer in float64 (BitNet keeps
  //! its embedding and output head in high precision; here the input layer
  //! sees the physical inputs and the last is the regression head).
  bool ternary_keep_first_layer = true;
  bool ternary_keep_last_layer = true;
  //! Ternary only: the backward GEMMs of the ternary layers (the STE,
  //! float64 master weights and Adam state in every mode;
  //! `internal/MlpTernaryGrad.h`): `"float64"` (default, BitNet's recipe:
  //! dgrad and wgrad in float64 on the dequantised operands),
  //! `"int8_dgrad"` (SwitchBack: dgrad on the ternary x int8 kernel with the
  //! output gradient int8 per row; wgrad float64), `"int8"` (Jetfire-style:
  //! also wgrad int8 x int8 on fprop's int8 activations, the gradient int8
  //! per 32-row block with stochastic rounding keyed by `seed`).
  std::string ternary_backward = "float64";
  //! Ternary only: the optimisation schedule. `"bitnet"` (default; lower
  //! held-out error than `"constant"` on both test sets, okf/neural-net.md)
  //! or `"constant"`: `learning_rate`, `beta2` and `alpha` every step,
  //! early stopping as set (the schedule before 2026-09-26). `"bitnet"`: the
  //! two-stage recipe of BitNet b1.58 ("The Era of 1-bit LLMs: Training
  //! Tips, Code and FAQ", Ma, Wang, Wei): over the `T = max_iter x batches`
  //! steps the learning rate is `peak_k x (1 - t / T) x min(1, (t + 1) /
  //! warm-up)` with `peak_1 = ternary_lr_stage1 x learning_rate` before the
  //! split step `ternary_stage_split x T` and `peak_2 = ternary_lr_stage2 x
  //! learning_rate` after it (linear decay with a drop at the split);
  //! decoupled weight decay `ternary_weight_decay` (AdamW-style, weights
  //! only) and the L2 `alpha` in stage 1, neither in stage 2 (the latent
  //! weights' magnitude is the trits' "confidence"); Adam's second moment
  //! `ternary_beta2`; early stopping's patience counts only in stage 2 (the
  //! loss curve of ternary training is S-shaped: stage 1's held-out loss
  //! does not predict the end), the best held-out weights are kept as ever.
  std::string ternary_schedule = "bitnet";
  double ternary_lr_stage1 = 1.5;
  double ternary_lr_stage2 = 1.0;
  double ternary_weight_decay = 0.1;
  double ternary_stage_split = 0.5;
  //! Linear warm-up, a fraction of the steps (BitNet: 375 of ~1e5 steps).
  double ternary_warmup = 0.01;
  double ternary_beta2 = 0.95;

  IMP_SHOWABLE_INLINE(NeuralNetTrainOptions,
                      out << "NeuralNetTrainOptions(max_iter " << max_iter
                          << ", learning_rate " << learning_rate << ")");
};
IMP_VALUES(NeuralNetTrainOptions, NeuralNetTrainOptionsList);

class NeuralNetTraining;

#ifndef SWIG
// Declared (exported) ahead of the class, so the friend declaration inside
// it names this function: a friend-first declaration has no dllexport, and
// MSVC then rejects the exported one below (C2375, different linkage).
// Defaults are added by the documented declaration further down.
IMPBFFEXPORT NeuralNetTraining train_neural_net_with_history(
    const std::vector<double>& X, int n_samples, int n_features,
    const std::vector<double>& Y, int n_targets, const NeuralNetTrainOptions& opts);
#endif

//! What train_neural_net_with_history() produced: the network and its curves.
class IMPBFFEXPORT NeuralNetTraining {
 public:
  NeuralNetTraining() {}
  //! The `bff.neural_net` msgpack document; `IMP.bff.NeuralNet(get_network())` loads it.
  const MsgpackBytes& get_network() const { return network_; }
  //! Mean half squared error per epoch on the training rows (standardised units).
  const std::vector<double>& get_loss_curve() const { return loss_curve_; }
  //! Half squared error per epoch on the held-out rows; empty without early stopping.
  const std::vector<double>& get_validation_curve() const { return validation_curve_; }
  //! Epochs run.
  int get_number_of_epochs() const { return static_cast<int>(loss_curve_.size()); }
  //! The options' precision: "float64", "nvfp4", "mxfp4" or "ternary".
  const std::string& get_precision() const { return precision_; }
  //! FP4 / ternary training: the `bff.quantized_neural_net` msgpack
  //! document of the trained network in its training format -- FP4 layers
  //! with exactly the weights the forward pass used (2-D scaled) and FP4
  //! activations (W4A4), or ternary layers with fprop's trits and int8
  //! activations (W1.58A8), the kept layers in float64 -- so
  //! `QuantizedNeuralNet.from_msgpack(get_quantized_network())` evaluates
  //! what training evaluated. Empty for float64 training. get_network() is
  //! the float64 master weights in either case.
  const MsgpackBytes& get_quantized_network() const { return quantized_network_; }

  IMP_SHOWABLE_INLINE(NeuralNetTraining,
                      out << "NeuralNetTraining(" << loss_curve_.size() << " epochs)");

 private:
  friend NeuralNetTraining train_neural_net_with_history(
      const std::vector<double>&, int, int, const std::vector<double>&, int,
      const NeuralNetTrainOptions&);
  MsgpackBytes network_;
  MsgpackBytes quantized_network_;
  std::string precision_ = "float64";
  std::vector<double> loss_curve_;
  std::vector<double> validation_curve_;
};
IMP_VALUES(NeuralNetTraining, NeuralNetTrainings);

//! Fit a network to `(X, Y)` and return it with its loss curves.
/*! Inputs and targets are standardised (population standard deviation, a
    constant column scaled by 1) and the scalers stored with the model, so the
    network consumes and produces unscaled values.
    \param[in] X row-major `n_samples x n_features` inputs
    \param[in] n_samples rows of `X` and `Y`
    \param[in] n_features columns of `X`
    \param[in] Y row-major `n_samples x n_targets` targets
    \param[in] n_targets columns of `Y`
    \param[in] opts hyper-parameters
    \throws IMP::ValueException on inconsistent sizes or options
 */
IMPBFFEXPORT NeuralNetTraining train_neural_net_with_history(
    const std::vector<double>& X, int n_samples, int n_features,
    const std::vector<double>& Y, int n_targets,
    const NeuralNetTrainOptions& opts = NeuralNetTrainOptions());

//! Fit a network to `(X, Y)` and return its `bff.neural_net` msgpack document.
/*! The same fit as train_neural_net_with_history(), without the curves.
    `IMP.bff.NeuralNet(train_neural_net(...))` evaluates it.
 */
IMPBFFEXPORT MsgpackBytes train_neural_net(
    const std::vector<double>& X, int n_samples, int n_features,
    const std::vector<double>& Y, int n_targets,
    const NeuralNetTrainOptions& opts = NeuralNetTrainOptions());

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_NEURALNETTRAINING_H

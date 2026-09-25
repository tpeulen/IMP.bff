/**
 * \file NeuralNetTraining.cpp
 * \brief Fitting a dense network with Adam and explicit backpropagation --
 *        in float64, in FP4 (internal/MlpFp4Train.h), or ternary
 *        (internal/MlpTernaryTrain.h).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/NeuralNetTraining.h>

#include <IMP/bff/internal/AdamUpdate.h>
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpFp4Train.h>
#include <IMP/bff/internal/MlpTernaryTrain.h>
#include <IMP/bff/internal/MlpGemm.h>
#include <IMP/bff/internal/NetworkDocument.h>
#include <IMP/bff/internal/pcg_random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: IMP compiles bff as one unity translation unit.
namespace neural_net_training_detail {

namespace mc = IMP::bff::internal::mlpcore;
// The batch products run through Mat.h's blocked kernels (MlpGemm.h).
using TrainGemm = IMP::bff::internal::MatGemm;

//! A uniform double in [0, 1) from two pcg32 draws, 53 bits of mantissa.
inline double uniform01(pcg32& rng) {
  const std::uint64_t hi = rng();
  const std::uint64_t lo = rng();
  return static_cast<double>(((hi << 32) | lo) >> 11) * (1.0 / 9007199254740992.0);
}

//! Copy `count` rows of `src` (row-major, `n_cols` wide) selected by
//! `order[from .. from + count)`.
inline std::vector<double> gather_rows(const std::vector<double>& src, int n_cols,
                                       const std::vector<int>& order, int from, int count) {
  std::vector<double> out(static_cast<std::size_t>(count) * n_cols);
  for (int i = 0; i < count; ++i) {
    const double* s = src.data() + static_cast<std::size_t>(order[from + i]) * n_cols;
    std::copy(s, s + n_cols, out.data() + static_cast<std::size_t>(i) * n_cols);
  }
  return out;
}

}  // namespace neural_net_training_detail

NeuralNetTraining train_neural_net_with_history(
    const std::vector<double>& X, int n_samples, int n_features,
    const std::vector<double>& Y, int n_targets,
    const NeuralNetTrainOptions& opt) {
  namespace d = neural_net_training_detail;
  namespace mc = IMP::bff::internal::mlpcore;
  using IMP::bff::internal::Activation;
  using IMP::bff::internal::DenseLayer;
  using IMP::bff::internal::MlpModel;

  if (n_samples <= 0 || n_features <= 0 || n_targets <= 0)
    IMP_THROW("train_neural_net: empty training set", IMP::ValueException);
  if (X.size() != static_cast<std::size_t>(n_samples) * n_features)
    IMP_THROW("train_neural_net: X must be n_samples * n_features long", IMP::ValueException);
  if (Y.size() != static_cast<std::size_t>(n_samples) * n_targets)
    IMP_THROW("train_neural_net: Y must be n_samples * n_targets long", IMP::ValueException);
  if (opt.batch_size <= 0)
    IMP_THROW("train_neural_net: batch_size must be positive", IMP::ValueException);
  if (opt.max_iter < 0)
    IMP_THROW("train_neural_net: max_iter must be >= 0", IMP::ValueException);
  Activation hidden_act = Activation::ReLU;
  try {
    hidden_act = IMP::bff::internal::activation_from_string(opt.activation);
  } catch (const std::exception& e) {
    IMP_THROW(e.what(), IMP::ValueException);
  }

  namespace f4t = IMP::bff::internal::mlpfp4::train;
  namespace tnt = IMP::bff::internal::mlpternary::train;
  const bool ternary = opt.precision == "ternary";
  const bool fp4 = opt.precision != "float64" && !ternary;
  if (fp4 && opt.precision != "nvfp4" && opt.precision != "mxfp4")
    IMP_THROW("train_neural_net: precision must be float64, nvfp4, mxfp4 or ternary, not '"
                  << opt.precision << "'",
              IMP::ValueException);
  tnt::Config tcfg;
  tcfg.keep_first = opt.ternary_keep_first_layer;
  tcfg.keep_last = opt.ternary_keep_last_layer;
  tnt::Workspace tws;
  f4t::Config fcfg;
  fcfg.format = opt.precision == "mxfp4" ? IMP::bff::internal::mlpfp4::Format::MXFP4
                                         : IMP::bff::internal::mlpfp4::Format::NVFP4;
  fcfg.hadamard = opt.fp4_hadamard;
  fcfg.stochastic = opt.fp4_stochastic_rounding;
  fcfg.keep_first = opt.fp4_keep_first_layer;
  fcfg.keep_last = opt.fp4_keep_last_layer;
  const f4t::Hadamard16 hadamard;
  // the stochastic-rounding stream: counter-based draws keyed by the seed,
  // the step, the layer and the operand (MlpFp4.h, SrKey)
  f4t::SrStream sr_rng(0x5352'0000'0000'0000ULL ^
                       static_cast<std::uint64_t>(static_cast<std::uint32_t>(opt.seed)));
  f4t::Workspace fws;

  NeuralNetTraining result;
  result.precision_ = opt.precision;
  MlpModel model;

  // --- standardise inputs and targets, keeping the scalers with the model
  model.x_scaler.fit(X.data(), n_samples, n_features);
  model.y_scaler.fit(Y.data(), n_samples, n_targets);
  std::vector<double> Xs(X), Ys(Y);
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_features; ++j) {
      double& v = Xs[static_cast<std::size_t>(i) * n_features + j];
      v = (v - model.x_scaler.mean[j]) / model.x_scaler.scale[j];
    }
    for (int j = 0; j < n_targets; ++j) {
      double& v = Ys[static_cast<std::size_t>(i) * n_targets + j];
      v = (v - model.y_scaler.mean[j]) / model.y_scaler.scale[j];
    }
  }

  // --- layer geometry
  std::vector<int> dims;
  dims.push_back(n_features);
  for (int h : opt.hidden_layer_sizes)
    if (h > 0) dims.push_back(h);
  dims.push_back(n_targets);
  const std::size_t n_layers = dims.size() - 1;

  pcg32 rng(static_cast<std::uint64_t>(static_cast<std::uint32_t>(opt.seed)));

  // Glorot-uniform initialisation, as scikit-learn's MLP; the output layer is
  // linear because this is a regressor.
  model.layers.resize(n_layers);
  for (std::size_t l = 0; l < n_layers; ++l) {
    DenseLayer& dl = model.layers[l];
    const int n_in = dims[l], n_out = dims[l + 1];
    const double limit = std::sqrt(6.0 / (n_in + n_out));
    dl.n_in = n_in;
    dl.n_out = n_out;
    dl.activation = (l + 1 == n_layers) ? Activation::Identity : hidden_act;
    dl.weight.resize(static_cast<std::size_t>(n_out) * n_in);
    for (std::size_t k = 0; k < dl.weight.size(); ++k)
      dl.weight[k] = (2.0 * d::uniform01(rng) - 1.0) * limit;
    dl.bias.assign(static_cast<std::size_t>(n_out), 0.0);
  }

  // --- flat parameters, their gradient, and which entries are biases (no L2)
  std::vector<double> params;
  mc::flatten(model.layers, params);
  const std::size_t n_params = params.size();
  std::vector<double> grad(n_params, 0.0);
  std::vector<char> is_bias(n_params, 0);
  {
    std::size_t p = 0;
    for (const DenseLayer& dl : model.layers) {
      p += dl.weight.size();
      std::fill(is_bias.begin() + static_cast<std::ptrdiff_t>(p),
                is_bias.begin() + static_cast<std::ptrdiff_t>(p + dl.bias.size()), 1);
      p += dl.bias.size();
    }
  }
  IMP::bff::internal::AdamState adam;
  adam.reset(n_params);

  // --- train/validation split for early stopping
  std::vector<int> order(n_samples);
  std::iota(order.begin(), order.end(), 0);
  for (int i = n_samples - 1; i > 0; --i)
    std::swap(order[i], order[rng() % static_cast<std::uint32_t>(i + 1)]);

  int n_val = 0;
  if (opt.early_stopping && n_samples >= 10)
    n_val = std::max(1, static_cast<int>(n_samples * opt.validation_fraction));
  const int n_train = n_samples - n_val;
  if (n_train <= 0)
    IMP_THROW("train_neural_net: validation_fraction leaves no training rows",
              IMP::ValueException);

  const std::vector<double> Xtr = d::gather_rows(Xs, n_features, order, 0, n_train);
  const std::vector<double> Ytr = d::gather_rows(Ys, n_targets, order, 0, n_train);
  const std::vector<double> Xva = d::gather_rows(Xs, n_features, order, n_train, n_val);
  const std::vector<double> Yva = d::gather_rows(Ys, n_targets, order, n_train, n_val);

  mc::Workspace ws;
  // Half mean squared error on a set, ``||y - t||^2 / (2 n)``.
  auto half_mse = [&](const std::vector<double>& in, const std::vector<double>& target,
                      int n_rows) {
    if (n_rows == 0) return 0.0;
    if (fp4) {
      f4t::quantize_weights(model.layers, fcfg, fws);
      f4t::forward<d::TrainGemm>(model.layers, fcfg, in.data(), n_rows, fws);
    } else if (ternary) {
      tnt::quantize_weights(model.layers, tcfg, tws);
      tnt::forward<d::TrainGemm>(model.layers, tcfg, in.data(), n_rows, tws);
    } else {
      mc::forward<d::TrainGemm>(model.layers, in.data(), n_rows, ws, 0);
    }
    const std::vector<double>& y = fp4 ? fws.output() : ternary ? tws.output() : ws.output();
    double acc = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      const double r = y[i] - target[i];
      acc += r * r;
    }
    return acc / (2.0 * n_rows);
  };

  std::vector<int> batch_order(n_train);
  std::iota(batch_order.begin(), batch_order.end(), 0);

  double best_val = std::numeric_limits<double>::infinity();
  int n_bad = 0;
  std::vector<double> best_params = params;
  std::vector<double> dY;

  try {
    for (int epoch = 0; epoch < opt.max_iter; ++epoch) {
      for (int i = n_train - 1; i > 0; --i)
        std::swap(batch_order[i], batch_order[rng() % static_cast<std::uint32_t>(i + 1)]);

      double epoch_loss = 0.0;
      int n_batches = 0;
      for (int start = 0; start < n_train; start += opt.batch_size) {
        const int bs = std::min(opt.batch_size, n_train - start);
        const std::vector<double> xb = d::gather_rows(Xtr, n_features, batch_order, start, bs);
        const std::vector<double> yb = d::gather_rows(Ytr, n_targets, batch_order, start, bs);

        if (fp4) {
          f4t::quantize_weights(model.layers, fcfg, fws);
          f4t::forward<d::TrainGemm>(model.layers, fcfg, xb.data(), bs, fws);
        } else if (ternary) {
          tnt::quantize_weights(model.layers, tcfg, tws);
          tnt::forward<d::TrainGemm>(model.layers, tcfg, xb.data(), bs, tws);
        } else {
          mc::forward<d::TrainGemm>(model.layers, xb.data(), bs, ws, 0);
        }
        const std::vector<double>& y = fp4 ? fws.output() : ternary ? tws.output() : ws.output();

        // dL/dy for L = ||y - t||^2 / (2 bs), and the loss itself
        dY.resize(y.size());
        double batch_loss = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
          const double r = y[i] - yb[i];
          dY[i] = r / static_cast<double>(bs);
          batch_loss += r * r;
        }
        epoch_loss += batch_loss / (2.0 * bs);
        ++n_batches;

        std::fill(grad.begin(), grad.end(), 0.0);
        if (fp4)
          f4t::backward<d::TrainGemm>(model.layers, fcfg, fws, dY.data(), bs, sr_rng, hadamard,
                                      grad.data());
        else if (ternary)
          tnt::backward<d::TrainGemm>(model.layers, tcfg, tws, dY.data(), bs, grad.data());
        else
          mc::backward<d::TrainGemm>(model.layers, ws, dY.data(), nullptr, nullptr, grad.data());
        if (opt.alpha > 0.0)  // L2 on weights only
          for (std::size_t i = 0; i < n_params; ++i)
            if (!is_bias[i]) grad[i] += opt.alpha * params[i];

        IMP::bff::internal::adam_update(params.data(), grad.data(), n_params, adam,
                                        opt.learning_rate, opt.beta1, opt.beta2,
                                        opt.epsilon);
        mc::unflatten(model.layers, params.data(), n_params);
      }

      result.loss_curve_.push_back(n_batches ? epoch_loss / n_batches : 0.0);

      if (n_val > 0) {
        const double vl = half_mse(Xva, Yva, n_val);
        result.validation_curve_.push_back(vl);
        if (vl < best_val - opt.tol) {
          best_val = vl;
          n_bad = 0;
          best_params = params;
        } else if (++n_bad >= opt.n_iter_no_change) {
          break;  // patience exhausted; keep the best weights seen
        }
      }
    }

    if (n_val > 0)  // restore the best held-out weights
      mc::unflatten(model.layers, best_params.data(), best_params.size());
    model.validate();
  } catch (const std::exception& e) {
    IMP_THROW(std::string("train_neural_net: ") + e.what(), IMP::ValueException);
  }

  result.network_ = internal::model_to_msgpack(model);
  if (ternary)
    result.quantized_network_ = internal::ternary_to_msgpack(tnt::to_model(model, tcfg));
  if (fp4)
    result.quantized_network_ = internal::quantized_to_msgpack(
        opt.precision, true, IMP::bff::internal::mlpquant::QuantModel(),
        f4t::to_model(model, fcfg));
  return result;
}

MsgpackBytes train_neural_net(const std::vector<double>& X, int n_samples, int n_features,
                             const std::vector<double>& Y, int n_targets,
                             const NeuralNetTrainOptions& opts) {
  return train_neural_net_with_history(X, n_samples, n_features, Y, n_targets, opts)
      .get_network();
}

IMPBFF_END_NAMESPACE

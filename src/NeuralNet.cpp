/**
 * \file NeuralNet.cpp
 * \brief A dense network: evaluated, differentiated and imported.
 *
 * Every kernel is `internal/MlpCore.h`'s, with the products through
 * `internal/MlpGemm.h`'s MatGemm; the int8 path is `internal/MlpQuant.h`, the
 * FP4 paths `internal/MlpFp4.h` with the kernels of `internal/MlpFp4Kernels.h`.
 * This file is argument checking, the compute door, and the views.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/NeuralNet.h>

#include <IMP/bff/ComputeBackend.h>
#include <IMP/bff/internal/OutputView.h>

#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpGemm.h>
#include <IMP/bff/internal/MlpFp4.h>
#include <IMP/bff/internal/MlpFp4Kernels.h>
#include <IMP/bff/internal/MlpTernary.h>
#include <IMP/bff/internal/MlpQuant.h>
#include <IMP/bff/internal/NetworkDocument.h>

#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace mc = IMP::bff::internal::mlpcore;
// Named, not anonymous: IMP compiles bff as one unity translation unit.
namespace neural_net_detail {
using Gemm = IMP::bff::internal::MatGemm;
}

struct NeuralNet::Impl {
    internal::MlpModel model;
    // Flattened once at construction (and again by set_parameters): the door
    // takes plain arrays, and a network is evaluated far more often than it
    // is loaded.
    std::vector<int> n_in, n_out, activation;
    std::vector<double> weights, biases;
    mutable std::string last_backend = "cpu";

    explicit Impl(internal::MlpModel m) : model(std::move(m)) { flatten_for_door(); }

    void flatten_for_door() {
        n_in.clear(); n_out.clear(); activation.clear(); weights.clear(); biases.clear();
        for (const internal::DenseLayer& l : model.layers) {
            n_in.push_back(l.n_in);
            n_out.push_back(l.n_out);
            activation.push_back(static_cast<int>(l.activation));
            weights.insert(weights.end(), l.weight.begin(), l.weight.end());
            biases.insert(biases.end(), l.bias.begin(), l.bias.end());
        }
    }
};

namespace neural_net_detail {

//! Publish `v` as a `(rows, cols)` managed view.
void publish_matrix(const std::vector<double>& v, int rows, int cols,
                    double** out, int* n_rows, int* n_cols) {
    int n = 0;
    double* buffer = internal::new_double_view(v.size(), out, &n);
    if (out == nullptr) {  // a C++ caller that did not ask for this one
        std::free(buffer);
        return;
    }
    if (buffer != nullptr && !v.empty())
        std::copy(v.begin(), v.end(), buffer);
    if (n_rows) *n_rows = buffer != nullptr ? rows : 0;
    if (n_cols) *n_cols = cols;
}

void publish_vector(const std::vector<double>& v, double** out, int* n) {
    if (out == nullptr) return;
    internal::copy_to_view(v, out, n);
}

void check_batch(const char* who, const char* what, int n_rows, int n_cols,
                 int want_cols) {
    if (n_rows < 0)
        IMP_THROW("NeuralNet::" << who << ": " << what << " has a negative row count",
                  IMP::ValueException);
    if (n_cols != want_cols)
        IMP_THROW("NeuralNet::" << who << ": " << what << " has " << n_cols
                                << " columns, expected " << want_cols,
                  IMP::ValueException);
}

std::string read_file_bytes(const std::string& path, const char* who) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) IMP_THROW("NeuralNet::" << who << ": cannot open '" << path << "'", IMP::IOException);
    std::stringstream ss;
    ss << fh.rdbuf();
    return ss.str();
}

}  // namespace neural_net_detail

NeuralNet::NeuralNet(const MsgpackBytes& document)
    : impl_(std::make_shared<Impl>(internal::model_from_msgpack(document, "NeuralNet"))) {}

NeuralNet::NeuralNet(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

NeuralNet::~NeuralNet() {}

// ---------------------------------------------------------------- import

NeuralNet NeuralNet::from_onnx(const OnnxBytes& model) {
    internal::MlpModel m;
    try {
        m = mc::model_from_onnx(model);
        m.validate();
    } catch (const IMP::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IMP_THROW("NeuralNet::from_onnx: " << e.what(), IMP::ValueException);
    }
    return NeuralNet(std::make_shared<Impl>(std::move(m)));
}

NeuralNet NeuralNet::from_onnx_file(const std::string& path) {
    return from_onnx(neural_net_detail::read_file_bytes(path, "from_onnx_file"));
}

NeuralNet NeuralNet::from_safetensors(const SafetensorsBytes& data,
                                      const std::string& hidden_activation) {
    internal::MlpModel m;
    try {
        m = mc::model_from_safetensors<nlohmann::json>(
                reinterpret_cast<const unsigned char*>(data.data()), data.size(),
                hidden_activation);
    } catch (const IMP::Exception&) {
        throw;
    } catch (const std::exception& e) {
        IMP_THROW("NeuralNet::from_safetensors: " << e.what(), IMP::ValueException);
    }
    return NeuralNet(std::make_shared<Impl>(std::move(m)));
}

NeuralNet NeuralNet::from_safetensors_file(const std::string& path,
                                           const std::string& hidden_activation) {
    return from_safetensors(neural_net_detail::read_file_bytes(path, "from_safetensors_file"), hidden_activation);
}

MsgpackBytes NeuralNet::to_msgpack() const {
    return internal::model_to_msgpack(impl_->model);
}

// ---------------------------------------------------------------- shape

int NeuralNet::get_n_inputs() const { return impl_->model.n_inputs(); }
int NeuralNet::get_n_outputs() const { return impl_->model.n_outputs(); }
int NeuralNet::get_n_layers() const { return static_cast<int>(impl_->model.layers.size()); }
int NeuralNet::get_n_parameters() const {
    return static_cast<int>(mc::n_parameters(impl_->model.layers));
}
std::string NeuralNet::get_last_backend() const { return impl_->last_backend; }

namespace neural_net_detail {
const internal::DenseLayer& layer_at(const internal::MlpModel& m, int layer, const char* who) {
    if (layer < 0 || layer >= static_cast<int>(m.layers.size()))
        IMP_THROW("NeuralNet::" << who << ": layer " << layer << " out of range [0, "
                                << m.layers.size() << ")",
                  IMP::ValueException);
    return m.layers[static_cast<std::size_t>(layer)];
}
}  // namespace neural_net_detail

std::string NeuralNet::get_layer_activation(int layer) const {
    return internal::activation_to_string(neural_net_detail::layer_at(impl_->model, layer, "get_layer_activation").activation);
}

void NeuralNet::get_layer_weights(int layer, double** out_matrix, int* n_out_rows,
                                  int* n_out_cols) const {
    const internal::DenseLayer& l = neural_net_detail::layer_at(impl_->model, layer, "get_layer_weights");
    neural_net_detail::publish_matrix(l.weight, l.n_out, l.n_in, out_matrix, n_out_rows, n_out_cols);
}

void NeuralNet::get_layer_bias(int layer, double** out_view, int* n_out_view) const {
    neural_net_detail::publish_vector(neural_net_detail::layer_at(impl_->model, layer, "get_layer_bias").bias, out_view, n_out_view);
}

// ---------------------------------------------------------------- evaluation

void NeuralNet::predict(const std::vector<double>& x, int n_rows,
                        double** out_view, int* n_out_view) const {
    const int n_in = get_n_inputs(), n_out = get_n_outputs();
    if (n_rows < 0) IMP_THROW("NeuralNet::predict: n_rows must be >= 0", IMP::ValueException);
    if (x.size() != static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_in))
        IMP_THROW("NeuralNet::predict: x must be n_rows * n_inputs long",
                  IMP::ValueException);
    std::vector<double> y;
    impl_->last_backend = "cpu";

    // The accelerator is offered the whole forward pass and may decline it --
    // no device, or a batch small enough that the round trip costs more than
    // the arithmetic. Nothing is written when it declines.
    const ImpBffComputeBackend* backend = get_compute_backend();
    bool done = false;
    if (n_rows > 0 && backend && backend->abi >= 2 && backend->mlp_forward) {
        std::vector<double> xs(x);
        mc::detail::scale_in(xs, n_rows, n_in, impl_->model.x_scaler);
        y.assign(static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_out), 0.0);
        const int rc = backend->mlp_forward(
                get_n_layers(), impl_->n_in.data(), impl_->n_out.data(),
                impl_->activation.data(), impl_->weights.data(),
                impl_->biases.data(), xs.data(), n_rows, y.data());
        if (rc == 0) {
            mc::detail::unscale_out(y, n_rows, n_out, impl_->model.y_scaler);
            impl_->last_backend = get_compute_backend_name();
            done = true;
        } else {
            y.clear();
        }
    }
    if (!done) {
        std::vector<double> d1, d2;
        mc::model_predict<neural_net_detail::Gemm>(impl_->model, x.data(), n_rows, 0, nullptr, y, d1, d2);
    }
    internal::copy_to_view(y, out_view, n_out_view);
}

void NeuralNet::predict(const std::vector<double>& x,
                        double** out_view, int* n_out_view) const {
    if (static_cast<int>(x.size()) != get_n_inputs())
        IMP_THROW("NeuralNet::predict: got " << x.size() << " values for one sample, expected "
                                             << get_n_inputs(),
                  IMP::ValueException);
    predict(x, 1, out_view, n_out_view);
}

// ---------------------------------------------------------------- derivatives

void NeuralNet::predict_derivatives(const double* in_x, int n_rows, int n_cols,
                                    const double* in_v, int n_rows_v, int n_cols_v,
                                    int order,
                                    double** out_y, int* n_out_y1, int* n_out_y2,
                                    double** out_dy_dv, int* n_out_dy_dv1, int* n_out_dy_dv2,
                                    double** out_d2y_dv2, int* n_out_d2y_dv21,
                                    int* n_out_d2y_dv22) const {
    const int n_in = get_n_inputs(), n_out = get_n_outputs();
    if (order < 0 || order > 2)
        IMP_THROW("NeuralNet::predict_derivatives: order must be 0, 1 or 2, not " << order,
                  IMP::ValueException);
    neural_net_detail::check_batch("predict_derivatives", "X", n_rows, n_cols, n_in);
    if (order >= 1) {
        neural_net_detail::check_batch("predict_derivatives", "V", n_rows_v, n_cols_v, n_in);
        if (n_rows_v != n_rows || (n_rows > 0 && in_v == nullptr))
            IMP_THROW("NeuralNet::predict_derivatives: X has " << n_rows << " rows but V has "
                                                               << n_rows_v,
                      IMP::ValueException);
    }
    std::vector<double> y, d1, d2;
    mc::model_predict<neural_net_detail::Gemm>(impl_->model, in_x, n_rows, order, order >= 1 ? in_v : nullptr,
                            y, d1, d2);
    neural_net_detail::publish_matrix(y, n_rows, n_out, out_y, n_out_y1, n_out_y2);
    neural_net_detail::publish_matrix(d1, order >= 1 ? n_rows : 0, n_out, out_dy_dv, n_out_dy_dv1, n_out_dy_dv2);
    neural_net_detail::publish_matrix(d2, order >= 2 ? n_rows : 0, n_out, out_d2y_dv2, n_out_d2y_dv21,
                   n_out_d2y_dv22);
}

void NeuralNet::jacobian(const std::vector<double>& x, double** out_matrix,
                         int* n_out_rows, int* n_out_cols) const {
    const int n_in = get_n_inputs(), n_out = get_n_outputs();
    if (static_cast<int>(x.size()) != n_in)
        IMP_THROW("NeuralNet::jacobian: got " << x.size() << " values, expected " << n_in,
                  IMP::ValueException);
    // One row per input direction: row j carries x and e_j.
    const std::size_t ni = static_cast<std::size_t>(n_in);
    std::vector<double> X(ni * ni), V(ni * ni, 0.0);
    for (std::size_t j = 0; j < ni; ++j) {
        std::copy(x.begin(), x.end(), X.begin() + static_cast<std::ptrdiff_t>(j * ni));
        V[j * ni + j] = 1.0;
    }
    std::vector<double> y, d1, d2;
    mc::model_predict<neural_net_detail::Gemm>(impl_->model, X.data(), n_in, 1, V.data(), y, d1, d2);
    // d1 is n_in x n_out (row j = column j of J); transpose to n_out x n_in.
    std::vector<double> J(static_cast<std::size_t>(n_out) * ni);
    for (std::size_t j = 0; j < ni; ++j)
        for (int k = 0; k < n_out; ++k)
            J[static_cast<std::size_t>(k) * ni + j] = d1[j * static_cast<std::size_t>(n_out) + k];
    neural_net_detail::publish_matrix(J, n_out, n_in, out_matrix, n_out_rows, n_out_cols);
}

void NeuralNet::hessian(const std::vector<double>& x, int output, double** out_matrix,
                        int* n_out_rows, int* n_out_cols) const {
    const int n_in = get_n_inputs(), n_out = get_n_outputs();
    if (static_cast<int>(x.size()) != n_in)
        IMP_THROW("NeuralNet::hessian: got " << x.size() << " values, expected " << n_in,
                  IMP::ValueException);
    if (output < 0 || output >= n_out)
        IMP_THROW("NeuralNet::hessian: output " << output << " out of range [0, " << n_out << ")",
                  IMP::ValueException);
    // Polarisation: H_ij = (q(e_i + e_j) - q(e_i) - q(e_j)) / 2, q(v) = v^T H v.
    // Rows 0..n_in-1 are the unit directions, then one row per pair i < j.
    const std::size_t ni = static_cast<std::size_t>(n_in);
    const std::size_t rows = ni + ni * (ni - 1) / 2;
    std::vector<double> X(rows * ni), V(rows * ni, 0.0);
    for (std::size_t r = 0; r < rows; ++r)
        std::copy(x.begin(), x.end(), X.begin() + static_cast<std::ptrdiff_t>(r * ni));
    for (std::size_t j = 0; j < ni; ++j) V[j * ni + j] = 1.0;
    {
        std::size_t r = ni;
        for (std::size_t i = 0; i < ni; ++i)
            for (std::size_t j = i + 1; j < ni; ++j, ++r) {
                V[r * ni + i] = 1.0;
                V[r * ni + j] = 1.0;
            }
    }
    std::vector<double> y, d1, d2;
    mc::model_predict<neural_net_detail::Gemm>(impl_->model, X.data(), static_cast<int>(rows), 2, V.data(),
                            y, d1, d2);
    auto q = [&](std::size_t r) { return d2[r * static_cast<std::size_t>(n_out) +
                                            static_cast<std::size_t>(output)]; };
    std::vector<double> H(ni * ni);
    for (std::size_t i = 0; i < ni; ++i) H[i * ni + i] = q(i);
    {
        std::size_t r = ni;
        for (std::size_t i = 0; i < ni; ++i)
            for (std::size_t j = i + 1; j < ni; ++j, ++r) {
                const double h = 0.5 * (q(r) - q(i) - q(j));
                H[i * ni + j] = h;
                H[j * ni + i] = h;
            }
    }
    neural_net_detail::publish_matrix(H, n_in, n_in, out_matrix, n_out_rows, n_out_cols);
}

void NeuralNet::backward(const double* in_x, int n_rows, int n_cols,
                         const double* in_dy, int n_rows_y, int n_cols_y,
                         double** out_dparams, int* n_out_dparams,
                         double** out_dx, int* n_out_dx1, int* n_out_dx2) const {
    backward_derivatives(in_x, n_rows, n_cols, nullptr, 0, get_n_inputs(),
                         in_dy, n_rows_y, n_cols_y, nullptr, 0, get_n_outputs(),
                         nullptr, 0, get_n_outputs(), out_dparams, n_out_dparams,
                         out_dx, n_out_dx1, n_out_dx2, nullptr, nullptr, nullptr);
}

void NeuralNet::backward_derivatives(const double* in_x, int n_rows, int n_cols,
                                     const double* in_v, int n_rows_v, int n_cols_v,
                                     const double* in_dy, int n_rows_y, int n_cols_y,
                                     const double* in_dy1, int n_rows_y1, int n_cols_y1,
                                     const double* in_dy2, int n_rows_y2, int n_cols_y2,
                                     double** out_dparams, int* n_out_dparams,
                                     double** out_dx, int* n_out_dx1, int* n_out_dx2,
                                     double** out_dv, int* n_out_dv1, int* n_out_dv2) const {
    const int n_in = get_n_inputs(), n_out = get_n_outputs();
    const char* who = "backward";
    neural_net_detail::check_batch(who, "X", n_rows, n_cols, n_in);
    neural_net_detail::check_batch(who, "dY", n_rows_y, n_cols_y, n_out);
    if (n_rows_y != n_rows)
        IMP_THROW("NeuralNet::backward: X has " << n_rows << " rows but dY has " << n_rows_y,
                  IMP::ValueException);
    const bool have1 = in_dy1 != nullptr && n_rows_y1 > 0;
    const bool have2 = in_dy2 != nullptr && n_rows_y2 > 0;
    if (have1) {
        neural_net_detail::check_batch(who, "dY1", n_rows_y1, n_cols_y1, n_out);
        if (n_rows_y1 != n_rows)
            IMP_THROW("NeuralNet::backward: dY1 has " << n_rows_y1 << " rows, X has " << n_rows,
                      IMP::ValueException);
    }
    if (have2) {
        neural_net_detail::check_batch(who, "dY2", n_rows_y2, n_cols_y2, n_out);
        if (n_rows_y2 != n_rows)
            IMP_THROW("NeuralNet::backward: dY2 has " << n_rows_y2 << " rows, X has " << n_rows,
                      IMP::ValueException);
    }
    if (have1 || have2) {
        neural_net_detail::check_batch(who, "V", n_rows_v, n_cols_v, n_in);
        if (n_rows_v != n_rows || in_v == nullptr)
            IMP_THROW("NeuralNet::backward: derivative adjoints are given, so V must have "
                      << n_rows << " rows (it has " << n_rows_v << ")",
                      IMP::ValueException);
    }
    std::vector<double> dparams, dx, dv;
    // model_backward needs a dY0 pointer even for zero rows.
    const double dummy = 0.0;
    mc::model_backward<neural_net_detail::Gemm>(impl_->model, in_x, n_rows, (have1 || have2) ? in_v : nullptr,
                             in_dy ? in_dy : &dummy, have1 ? in_dy1 : nullptr,
                             have2 ? in_dy2 : nullptr, dparams, dx, dv);
    neural_net_detail::publish_vector(dparams, out_dparams, n_out_dparams);
    neural_net_detail::publish_matrix(dx, n_rows, n_in, out_dx, n_out_dx1, n_out_dx2);
    neural_net_detail::publish_matrix(dv, n_rows, n_in, out_dv, n_out_dv1, n_out_dv2);
}

void NeuralNet::get_parameters(double** out_view, int* n_out_view) const {
    std::vector<double> p;
    mc::flatten(impl_->model.layers, p);
    neural_net_detail::publish_vector(p, out_view, n_out_view);
}

void NeuralNet::set_parameters(const double* in_params, int n_params) {
    const int want = get_n_parameters();
    if (n_params != want || (want > 0 && in_params == nullptr))
        IMP_THROW("NeuralNet::set_parameters: got " << n_params << " values, expected "
                                                    << want,
                  IMP::ValueException);
    // Copy on write: a copy of this network keeps the parameters it had.
    if (impl_.use_count() > 1) impl_ = std::make_shared<Impl>(*impl_);
    mc::unflatten(impl_->model.layers, in_params, static_cast<std::size_t>(n_params));
    impl_->flatten_for_door();
}

// ---------------------------------------------------------------- int8 / FP4 / ternary

struct QuantizedNeuralNet::Impl {
    std::string format;
    bool quantize_activations = false;
    internal::mlpquant::QuantModel int8;
    internal::mlpfp4::Fp4Model fp4;
    internal::mlpternary::TModel ternary;
    //! fp4's layers packed for the kernels, once (points into `fp4`)
    internal::mlpfp4::kern::Prepared packed;
    //! the ternary layers in the kernel layout, once (points into `ternary`)
    internal::mlpternary::Prepared tpacked;
    bool is_int8() const { return format == "int8"; }
    bool is_ternary() const { return format.compare(0, 7, "ternary") == 0; }
    void prepare() {
        if (is_ternary())
            tpacked = internal::mlpternary::prepare(ternary);
        else if (!is_int8())
            packed = internal::mlpfp4::kern::prepare(fp4);
    }
    Impl() = default;
    Impl(const Impl& o)
        : format(o.format), quantize_activations(o.quantize_activations), int8(o.int8), fp4(o.fp4),
          ternary(o.ternary) {
        prepare();
    }
    Impl& operator=(const Impl&) = delete;
};

QuantizedNeuralNet::QuantizedNeuralNet(const NeuralNet& net, const std::string& format,
                                       bool quantize_activations)
    : impl_(std::make_shared<Impl>()) {
    internal::mlpternary::Scale tsc;
    internal::mlpternary::Storage tst;
    const bool ternary = internal::mlpternary::parse_format(format, tsc, tst);
    if (format != "int8" && format != "fp4" && format != "mxfp4" && format != "nvfp4" && !ternary)
        IMP_THROW("QuantizedNeuralNet: unknown format '"
                          << format
                          << "' (int8, fp4, mxfp4, nvfp4, ternary, ternary_row, ternary_tq1 or ternary_tq1_row)",
                  IMP::ValueException);
    const internal::MlpModel m =
            internal::model_from_msgpack(net.to_msgpack(), "QuantizedNeuralNet");
    impl_->format = format;
    if (ternary) {
        impl_->quantize_activations = true;
        impl_->ternary = internal::mlpternary::quantize(m, tsc, tst);
        impl_->prepare();
    } else if (impl_->is_int8()) {
        impl_->quantize_activations = true;
        impl_->int8 = internal::mlpquant::quantize(m);
    } else {
        impl_->quantize_activations = quantize_activations;
        try {
            impl_->fp4 = internal::mlpfp4::quantize(m, internal::mlpfp4::format_from_string(format),
                                                    quantize_activations);
            impl_->prepare();
        } catch (const std::exception& e) {
            IMP_THROW("QuantizedNeuralNet: " << e.what(), IMP::ValueException);
        }
    }
}

QuantizedNeuralNet::~QuantizedNeuralNet() {}

QuantizedNeuralNet QuantizedNeuralNet::from_msgpack(const MsgpackBytes& document) {
    QuantizedNeuralNet q;
    q.impl_ = std::make_shared<Impl>();
    q.impl_->format = internal::quantized_from_msgpack(document, q.impl_->quantize_activations,
                                                       q.impl_->int8, q.impl_->fp4, q.impl_->ternary);
    if (q.impl_->is_int8()) q.impl_->quantize_activations = true;
    q.impl_->prepare();
    return q;
}

MsgpackBytes QuantizedNeuralNet::to_msgpack() const {
    if (impl_->is_ternary()) return internal::ternary_to_msgpack(impl_->ternary);
    return internal::quantized_to_msgpack(impl_->format, impl_->quantize_activations, impl_->int8,
                                          impl_->fp4);
}

std::string QuantizedNeuralNet::get_format() const { return impl_->format; }
bool QuantizedNeuralNet::get_quantize_activations() const { return impl_->quantize_activations; }
int QuantizedNeuralNet::get_n_inputs() const {
    if (impl_->is_ternary()) return impl_->ternary.n_inputs();
    return impl_->is_int8() ? impl_->int8.n_inputs() : impl_->fp4.n_inputs();
}
int QuantizedNeuralNet::get_n_outputs() const {
    if (impl_->is_ternary()) return impl_->ternary.n_outputs();
    return impl_->is_int8() ? impl_->int8.n_outputs() : impl_->fp4.n_outputs();
}
int QuantizedNeuralNet::get_n_layers() const {
    if (impl_->is_ternary()) return static_cast<int>(impl_->ternary.layers.size());
    return static_cast<int>(impl_->is_int8() ? impl_->int8.layers.size() : impl_->fp4.layers.size());
}
int QuantizedNeuralNet::get_n_weights() const {
    if (impl_->is_ternary()) return static_cast<int>(impl_->ternary.n_weights());
    if (!impl_->is_int8()) return static_cast<int>(impl_->fp4.n_weights());
    std::size_t n = 0;
    for (const auto& l : impl_->int8.layers) n += l.weight.size();
    return static_cast<int>(n);
}
int QuantizedNeuralNet::get_weight_bytes() const {
    if (impl_->is_ternary()) return static_cast<int>(impl_->ternary.weight_bytes());
    if (impl_->is_int8())  // one byte a weight and a double scale a layer
        return static_cast<int>(impl_->int8.weight_bytes() + sizeof(double) * impl_->int8.layers.size());
    return static_cast<int>(impl_->fp4.weight_bytes());
}
double QuantizedNeuralNet::get_bits_per_weight() const {
    const int n = get_n_weights();
    return n > 0 ? 8.0 * get_weight_bytes() / n : 0.0;
}
std::string QuantizedNeuralNet::get_kernel_name() {
    return internal::mlpfp4::kern::kernel_name();
}

void QuantizedNeuralNet::predict(const std::vector<double>& x, int n_rows,
                                 double** out_view, int* n_out_view) const {
    if (n_rows < 0)
        IMP_THROW("QuantizedNeuralNet::predict: n_rows must be >= 0", IMP::ValueException);
    if (x.size() != static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(get_n_inputs()))
        IMP_THROW("QuantizedNeuralNet::predict: x must be n_rows * n_inputs long",
                  IMP::ValueException);
    std::vector<double> y;
    if (impl_->is_ternary())
        internal::mlpternary::predict<neural_net_detail::Gemm>(impl_->tpacked, x.data(), n_rows, y);
    else if (impl_->is_int8())
        internal::mlpquant::predict(impl_->int8, x.data(), n_rows, y);
    else
        internal::mlpfp4::kern::predict<neural_net_detail::Gemm>(impl_->packed, x.data(), n_rows, y);
    internal::copy_to_view(y, out_view, n_out_view);
}

IMPBFF_END_NAMESPACE

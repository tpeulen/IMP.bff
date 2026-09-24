/**
 *  \file IMP/bff/NeuralNet.h
 *  \brief A dense network: evaluated, differentiated and imported.
 *
 * The kernels are bff's own `internal/MlpCore.h`; this header is the face
 * that bff and its bindings use. It adds three things to them:
 *
 * - **Where the forward pass runs.** `predict()` offers the whole batch to
 *   whatever `IMP/bff/ComputeBackend.h` has loaded, and runs it on the CPU
 *   when nothing is loaded, when the backend declines, or when the batch is
 *   too small to be worth the trip. Only predict() is offered; every
 *   derivative runs on the CPU.
 * - **The network as a differentiable building block.** predict_derivatives()
 *   carries a directional Taylor expansion through the forward pass (`y`,
 *   `J v`, `v^T H v`), jacobian() / hessian() give the full matrices at one
 *   sample, backward() and backward_derivatives() are the exact reverse
 *   passes for any loss the caller computes on those, and get_parameters() /
 *   set_parameters() are the flat vector an outside optimiser (L-BFGS, Adam)
 *   works on. That is what a physics-informed fit or a network inside a
 *   larger model needs; no tape or autodiff library is involved.
 * - **Networks trained elsewhere.** from_onnx() reads the dense-MLP subset of
 *   ONNX that PyTorch (either exporter), Keras, JAX and skl2onnx write, and
 *   from_safetensors() a PyTorch `state_dict`. Both are import formats only.
 *
 * Every network document bff *stores* is **msgpack** (MsgpackBytes below):
 * the `bff.neural_net` map this constructor reads and to_msgpack() writes,
 * what train_neural_net() returns, the action policy a model search takes
 * and the network nested in a `bff.hmm_surrogate`. There is no JSON reader
 * and no ONNX writer; the one encoder and decoder are
 * `internal/NetworkDocument.h`. An imported network becomes a native one by
 * `NeuralNet(net.to_msgpack())`.
 *
 * The API is tttrlib's `NeuralNet` (removed there in 10858cf19), ported with
 * bff's names: `get_` prefixes, managed numpy views for arrays, msgpack in
 * place of JSON, IMP::ValueException for every refusal.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_NEURALNET_H
#define IMPBFF_NEURALNET_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A msgpack document as raw bytes: bff's native format for networks.
/*! Binary-safe in C++ (a `std::string` holds any byte, including zero); in
    Python it is `bytes` both ways -- a bytes, bytearray or memoryview goes
    in, `bytes` comes out, and a `str` is refused with a TypeError. Build or
    read one with the `msgpack` package: `msgpack.packb(doc,
    use_bin_type=True)` / `msgpack.unpackb(data, raw=False)`. */
typedef std::string MsgpackBytes;

//! An ONNX model (`.onnx` file contents) as raw bytes; an import format.
/*! Binary-safe in C++; in Python `bytes`, `bytearray` or `memoryview` --
    `open(path, "rb").read()`. A `str` is refused with a TypeError. */
typedef std::string OnnxBytes;

//! A safetensors file's contents as raw bytes; an import format.
/*! As OnnxBytes: bytes-like in Python, a `str` refused with a TypeError. */
typedef std::string SafetensorsBytes;

//! A dense multilayer perceptron: evaluated in batches, and differentiable.
/*!
    Constructed from a `bff.neural_net` msgpack document -- the one
    train_neural_net() returns and the one a scikit-learn `MLPRegressor`
    converts to: a map `{"format": "bff.neural_net", "version": 1,
    "x_scaler", "y_scaler", "layers": [{"n_in", "n_out", "activation",
    "weight" (row-major n_out x n_in), "bias"}, ...]}` -- or imported with
    from_onnx() / from_safetensors().

    **The batch is the unit.** One `predict()` call crosses the binding once
    and, if an accelerator is loaded, crosses the plugin boundary once, however
    deep the network. Calling it per row would repeat the mistake that once
    made the quenching suite forty times slower. The derivative entry points
    take batches for the same reason.

    **Units.** Every input and output is in the network's physical
    (unscaled) units: the stored `x_scaler` / `y_scaler` are applied and
    undone inside, and every derivative includes them in the chain rule.

    **Arrays.** Batches come in as `(n_rows, n_inputs)` arrays (IN_ARRAY2 in
    Python: any array-like, converted to C-contiguous float64). Results are
    managed numpy views; in C++ each `out_*` buffer is `malloc`-ed and the
    caller frees it with `std::free`.

    Copies share their parameters until one of them calls set_parameters(),
    which gives that copy its own (copy on write).
*/
class IMPBFFEXPORT NeuralNet {
public:
    //! Build from a `bff.neural_net` msgpack document.
    /*! \throws IMP::ValueException if the bytes are not msgpack, not a
        `bff.neural_net` map, or if the layers do not chain. */
    explicit NeuralNet(const MsgpackBytes& document);
    ~NeuralNet();

    //! \name Import
    //! @{

    //! Read an ONNX model holding a dense network.
    /*! The MLP subset every exporter emits: `Gemm` (or `MatMul` + `Add`)
        with constant weights, `Relu` / `Tanh` / `Sigmoid` / `Softplus` /
        `Sin` between layers, the `Sigmoid` + `Mul` pattern for SiLU, and
        pass-through reshapes -- PyTorch (both exporters), Keras, JAX,
        scikit-learn via skl2onnx. No ONNX or protobuf library is involved;
        the wire format is read directly. The model carries no scalers (an
        ONNX graph normalises inside or not at all) and a float32 model
        evaluates as the float32 model it is.
        \throws IMP::ValueException naming the unsupported op or the defect */
    static NeuralNet from_onnx(const OnnxBytes& model);
    //! from_onnx() on the contents of the file at `path`.
    /*! \throws IMP::IOException if the file cannot be read */
    static NeuralNet from_onnx_file(const std::string& path);

    //! Read a safetensors file holding a PyTorch-style `state_dict`.
    /*! `<prefix>.weight` of shape `(n_out, n_in)` with optional
        `<prefix>.bias`, layers ordered by the first integer in the prefix;
        activations from the file's `__metadata__` (`"activations":
        "tanh,tanh,identity"` per layer, or `"activation": "tanh"` for the
        hidden layers), else `hidden_activation`; the output layer is linear.
        dtypes F32 and F64.
        \throws IMP::ValueException on a malformed file or unknown activation */
    static NeuralNet from_safetensors(const SafetensorsBytes& data,
                                      const std::string& hidden_activation = "tanh");
    //! from_safetensors() on the contents of the file at `path`.
    /*! \throws IMP::IOException if the file cannot be read */
    static NeuralNet from_safetensors_file(const std::string& path,
                                           const std::string& hidden_activation = "tanh");

    //! The network as a `bff.neural_net` msgpack document.
    /*! `NeuralNet(net.to_msgpack())` is the same network, bit for bit; this
        is how an imported network is stored natively or handed on as an
        action policy. Reflects set_parameters(). */
    MsgpackBytes to_msgpack() const;
    //! @}

    //! \name Shape
    //! @{
    //! Inputs the network takes.
    int get_n_inputs() const;
    //! Outputs it produces.
    int get_n_outputs() const;
    //! Layers it has.
    int get_n_layers() const;
    //! Weights plus biases: the length of get_parameters().
    int get_n_parameters() const;
    //! Activation of layer `layer` by its document name (`relu`, `tanh`,
    //! `logistic`, `identity`, `softplus`, `silu`, `sin`).
    std::string get_layer_activation(int layer) const;
    //! Weights of layer `layer` as an `(n_out, n_in)` view.
    void get_layer_weights(int layer, double** out_matrix, int* n_out_rows,
                           int* n_out_cols) const;
    //! Bias of layer `layer`, `n_out` long.
    void get_layer_bias(int layer, double** out_view, int* n_out_view) const;
    //! @}

    //! Where the last predict() ran: `cpu`, or the backend's name.
    /*! A loaded accelerator may still decline a batch as too small, so this
        is a fact about the last call rather than about the machine. */
    std::string get_last_backend() const;

    //! \name Evaluation
    //! @{

    //! Evaluate a batch.
    /*!
        \param[in] x the batch, row-major, `n_rows * get_n_inputs()` long
        \param[in] n_rows rows in the batch
        \param[out] out_view,n_out_view `n_rows * get_n_outputs()`, a managed
                    numpy view; in C++ free it with `std::free`
    */
    void predict(const std::vector<double>& x, int n_rows,
                 double** out_view, int* n_out_view) const;

    //! Evaluate one sample: predict() with `n_rows = 1`.
    /*! \param[in] x `get_n_inputs()` values
        \param[out] out_view,n_out_view `get_n_outputs()` values */
    void predict(const std::vector<double>& x,
                 double** out_view, int* n_out_view) const;
    //! @}

    //! \name Derivatives
    //! Every derivative runs on the CPU, through `internal/MlpCore.h`.
    //! @{

    //! Values and directional derivatives of a batch.
    /*!
        Each row of `x` is expanded along the direction in the same row of
        `v`: `y` is the value, `dy_dv` the directional derivative `J v`, and
        `d2y_dv2` the directional second derivative `v^T H v` (per output).
        A Laplacian is the sum of `d2y_dv2` over the unit directions; a full
        Jacobian or Hessian at one sample is jacobian() / hessian().

        In Python: `predict_derivatives(X, V=None, order=2)` returns
        `(y, dy_dv, d2y_dv2)`, each `(n_rows, n_outputs)`, with the orders
        above `order` as `None`.

        \param[in] in_x `(n_rows, n_inputs)` samples
        \param[in] in_v `(n_rows, n_inputs)` directions; may have 0 rows
                   (and be null) when `order == 0`
        \param[in] order 0: values only; 1: also `J v`; 2: also `v^T H v`
        \param[out] out_y,out_dy_dv,out_d2y_dv2 `(n_rows, n_outputs)` each;
                    the ones above `order` are `(0, n_outputs)`
        \throws IMP::ValueException on a bad order or shape
    */
    void predict_derivatives(const double* in_x, int n_rows, int n_cols,
                             const double* in_v, int n_rows_v, int n_cols_v,
                             int order,
                             double** out_y, int* n_out_y1, int* n_out_y2,
                             double** out_dy_dv, int* n_out_dy_dv1, int* n_out_dy_dv2,
                             double** out_d2y_dv2, int* n_out_d2y_dv21,
                             int* n_out_d2y_dv22) const;

    //! Jacobian `dy/dx` at one sample, `(n_outputs, n_inputs)`.
    void jacobian(const std::vector<double>& x, double** out_matrix,
                  int* n_out_rows, int* n_out_cols) const;

    //! Hessian `d2 y_k / dx dx` of output `output` at one sample,
    //! `(n_inputs, n_inputs)`; exactly symmetric.
    void hessian(const std::vector<double>& x, int output, double** out_matrix,
                 int* n_out_rows, int* n_out_cols) const;

    //! Reverse pass for an arbitrary loss on the outputs.
    /*!
        Given `dL/dy` for every sample and output, returns `dL/dparams` and
        `dL/dx`. This is the entry point for using the network as one term of
        a larger differentiable model: the caller's solver produces the
        adjoint of the network output, this returns the adjoint of the
        weights. Nothing about the loss is assumed.

        \param[in] in_x `(n_rows, n_inputs)` samples
        \param[in] in_dy `(n_rows, n_outputs)` adjoint of the outputs
        \param[out] out_dparams `get_n_parameters()` long, in
                    get_parameters() layout
        \param[out] out_dx `(n_rows, n_inputs)`
    */
    void backward(const double* in_x, int n_rows, int n_cols,
                  const double* in_dy, int n_rows_y, int n_cols_y,
                  double** out_dparams, int* n_out_dparams,
                  double** out_dx, int* n_out_dx1, int* n_out_dx2) const;

    //! Reverse pass for a loss that also depends on `J v` and `v^T H v`.
    /*!
        The adjoints `dy`, `dy1` (of `dy_dv`) and `dy2` (of `d2y_dv2`) are
        each `(n_rows, n_outputs)`; a `dy1` or `dy2` with no rows means zero.
        The Taylor order of the underlying pass is 2 if `dy2` is given, else
        1 if `dy1` is given, else 0. This is what a physics-informed loss --
        a PDE residual in `y`, its gradient and its Laplacian -- needs to
        train the weights by gradient descent or L-BFGS.

        In Python: `backward_derivatives(X, V, dY, dY1=None, dY2=None)`
        returns `(dparams, dx, dv)`.

        \param[out] out_dparams `get_n_parameters()` long
        \param[out] out_dx,out_dv `(n_rows, n_inputs)` each; `dv` is zero
                    for an order-0 pass
    */
    void backward_derivatives(const double* in_x, int n_rows, int n_cols,
                              const double* in_v, int n_rows_v, int n_cols_v,
                              const double* in_dy, int n_rows_y, int n_cols_y,
                              const double* in_dy1, int n_rows_y1, int n_cols_y1,
                              const double* in_dy2, int n_rows_y2, int n_cols_y2,
                              double** out_dparams, int* n_out_dparams,
                              double** out_dx, int* n_out_dx1, int* n_out_dx2,
                              double** out_dv, int* n_out_dv1, int* n_out_dv2) const;

    //! All weights and biases as one flat vector.
    /*! Layers in order, each as its row-major weight matrix followed by its
        bias: the layout backward() reports gradients in, and the one an
        optimiser works on. */
    void get_parameters(double** out_view, int* n_out_view) const;

    //! Overwrite the parameters from a flat vector in get_parameters() layout.
    /*! The layer shapes are unchanged; later predict() calls (and a loaded
        accelerator) see the new values.
        \throws IMP::ValueException unless `n_params == get_n_parameters()` */
    void set_parameters(const double* in_params, int n_params);
    //! @}

private:
    struct Impl;
    explicit NeuralNet(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

//! A NeuralNet's forward pass with quantised weights, for deployment.
/*!
    Built from a NeuralNet in one of four formats (`format`):

    - `"int8"` (default): dynamic-range int8 -- weights quantised once per
      tensor, symmetric; each layer's input requantised per sample; int32
      accumulation (`internal/MlpQuant.h`). **A reconstruction**: tttrlib's
      `MlpQuant.h` was lost before it was committed; this is rebuilt from
      its surviving description, not the original code.
    - `"fp4"`: FP4 E2M1 weights, one float32 scale per output row.
    - `"mxfp4"`: OCP MX v1.0 MXFP4 -- E2M1, blocks of 32 along the input
      dimension, one E8M0 (power-of-two) scale a block.
    - `"nvfp4"`: NVIDIA NVFP4 -- E2M1, blocks of 16, one FP8 E4M3 scale a
      block and one float32 scale a tensor.

    The exact recipes are in `internal/MlpFp4.h`. The FP4 weights are held
    packed (two codes a byte, element 2i in the low nibble -- ONNX
    FLOAT4E2M1 / PyTorch float4_e2m1fn_x2 order) with their scales, and
    predict() runs integer-SIMD dot products on the codes
    (`internal/MlpFp4Kernels.h`, after llama.cpp's MXFP4/NVFP4 kernels):
    without `quantize_activations` each layer's input is quantised to int8
    per block of 32 (16 for nvfp4) -- W4A8; with it, to FP4 in the same
    format per row -- W4A4, the arithmetic of NVFP4 GEMMs on Blackwell.
    CPUs have no FP4 unit: nothing is emulated in float, but the products
    are int8 products of the E2M1 values' doubles, scaled after. For
    `"int8"` the activations are always quantised and `quantize_activations`
    is ignored (get_quantize_activations() returns true).

    Biases, activations and scalers stay double. It is an inference path
    only -- no derivatives, no training, no accelerator; the source network
    is not changed. to_msgpack() / from_msgpack() store it as a
    `bff.quantized_neural_net` msgpack document (codes and scales as bin
    fields), bit-exact.
*/
class IMPBFFEXPORT QuantizedNeuralNet {
public:
    //! Quantise `net` (its current parameters) in `format`.
    /*! \throws IMP::ValueException for a format other than "int8", "fp4",
                "mxfp4", "nvfp4" */
    explicit QuantizedNeuralNet(const NeuralNet& net, const std::string& format = "int8",
                                bool quantize_activations = false);
    ~QuantizedNeuralNet();

    //! Load a `bff.quantized_neural_net` msgpack document.
    /*! \throws IMP::ValueException on a malformed document */
    static QuantizedNeuralNet from_msgpack(const MsgpackBytes& document);
    //! The `bff.quantized_neural_net` msgpack document (bit-exact round trip).
    MsgpackBytes to_msgpack() const;

    //! "int8", "fp4", "mxfp4" or "nvfp4".
    std::string get_format() const;
    //! Whether each layer's input is quantised in the weight format (W4A4).
    bool get_quantize_activations() const;
    //! Inputs the network takes.
    int get_n_inputs() const;
    //! Outputs it produces.
    int get_n_outputs() const;
    //! Layers it has.
    int get_n_layers() const;
    //! Weights (the number of multiply-adds a sample).
    int get_n_weights() const;
    //! Bytes the quantised weights occupy, scales included: int8 one a
    //! weight plus an 8-byte scale a layer; FP4 half a byte a weight (rows
    //! padded to 32) plus the scales. A double takes eight a weight.
    int get_weight_bytes() const;
    //! `8 * get_weight_bytes() / get_n_weights()`: about 4.25 for mxfp4,
    //! 4.5 for nvfp4, 4 + 32 / n_in for fp4, 8 for int8.
    double get_bits_per_weight() const;
    //! The FP4 kernel variant this build compiled: "neon-dotprod", "neon",
    //! "avx2" or "generic".
    static std::string get_kernel_name();

    //! Evaluate a batch, as NeuralNet::predict() does.
    /*!
        \param[in] x the batch, row-major, `n_rows * get_n_inputs()` long
        \param[in] n_rows rows in the batch
        \param[out] out_view,n_out_view `n_rows * get_n_outputs()`
    */
    void predict(const std::vector<double>& x, int n_rows,
                 double** out_view, int* n_out_view) const;

private:
    struct Impl;
    QuantizedNeuralNet() {}
    std::shared_ptr<Impl> impl_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_NEURALNET_H

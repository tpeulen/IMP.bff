/**
 *  \file IMP/bff/NeuralNet.h
 *  \brief Evaluating a dense network, on the CPU or through the compute door.
 *
 * The kernels are bff's own `internal/MlpCore.h`; this header is the face
 * that bff and its bindings use. What it adds is where the arithmetic runs:
 * `predict()` offers the whole forward pass to whatever
 * `IMP/bff/ComputeBackend.h` has loaded, and runs it on the CPU when nothing is
 * loaded, when the backend declines, or when the batch is too small to be
 * worth the trip.
 *
 * Every network document in bff is **msgpack** (MsgpackBytes below): the
 * `bff.neural_net` map this constructor reads, what train_neural_net()
 * writes, the action policy a model search takes and the network nested in
 * a `bff.hmm_surrogate`. There is no JSON reader; the one encoder and
 * decoder are `internal/NetworkDocument.h`.
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

//! A dense multilayer perceptron, evaluated in batches.
/*!
    Constructed from a `bff.neural_net` msgpack document -- the one
    train_neural_net() returns and the one a scikit-learn `MLPRegressor`
    converts to: a map `{"format": "bff.neural_net", "version": 1,
    "x_scaler", "y_scaler", "layers": [{"n_in", "n_out", "activation",
    "weight" (row-major n_out x n_in), "bias"}, ...]}`.

    **The batch is the unit.** One `predict()` call crosses the binding once
    and, if an accelerator is loaded, crosses the plugin boundary once, however
    deep the network. Calling it per row would repeat the mistake that once
    made the quenching suite forty times slower.
*/
class IMPBFFEXPORT NeuralNet {
public:
    //! Build from a `bff.neural_net` msgpack document.
    /*! \throws IMP::ValueException if the bytes are not msgpack, not a
        `bff.neural_net` map, or if the layers do not chain. */
    explicit NeuralNet(const MsgpackBytes& document);
    ~NeuralNet();

    //! Inputs the network takes.
    int get_n_inputs() const;
    //! Outputs it produces.
    int get_n_outputs() const;
    //! Layers it has.
    int get_n_layers() const;

    //! Where the last predict() ran: `cpu`, or the backend's name.
    /*! A loaded accelerator may still decline a batch as too small, so this
        is a fact about the last call rather than about the machine. */
    std::string get_last_backend() const;

    //! Evaluate a batch.
    /*!
        \param[in] x the batch, row-major, `n_rows * get_n_inputs()` long
        \param[in] n_rows rows in the batch
        \param[out] out_view,n_out_view `n_rows * get_n_outputs()`, a managed
                    numpy view; in C++ free it with `std::free`
    */
    void predict(const std::vector<double>& x, int n_rows,
                 double** out_view, int* n_out_view) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_NEURALNET_H

"""
=================================================
A neural network as a differentiable building block
=================================================

``IMP.bff.NeuralNet`` is a small dense network -- the kind that replaces an
expensive fit with a learned map (the HMM surrogate is built on it) or that
parametrises an unknown function inside a physical model. This example shows
the second use, which is what its derivative API exists for:

* ``backward(X, dL/dy)`` turns the adjoint of the outputs into the adjoint of
  the weights and of the inputs, for **any** loss the caller computes. That is
  the entry point for putting the network inside a larger differentiable model.
* ``jacobian`` / ``hessian`` give ``dy/dx`` and ``d²y/dx²`` at a sample, and
  ``predict_derivatives`` the directional forms ``J v`` and ``vᵀ H v`` for a
  whole batch -- what a physics residual needs (see the heat-equation and
  Burgers examples).
* ``get_parameters()`` / ``set_parameters()`` are the flat weight vector an
  outside optimiser works on.

Every derivative is exact -- reverse mode for the weights, a Taylor expansion
carried through the forward pass for the input derivatives -- and none of it
needs a tape or an autodiff library. The kernels are bff's header-only C++
(``internal/MlpCore.h``); the network came to IMP.bff from tttrlib.

We start where every network tutorial starts (XOR, then a sine), then check
each derivative against finite differences, and finish with a *Sobolev* fit:
training the network on a function **and its derivative** at once, which
plain regression cannot do.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import IMP.bff

rng = np.random.default_rng(0)


def predict(net, X):
    """``(n_rows, n_outputs)`` from a 2-D batch (``predict`` is flat)."""
    X = np.atleast_2d(X)
    return np.asarray(net.predict(X.ravel().tolist(), X.shape[0])).reshape(
        X.shape[0], net.get_n_outputs())


def train(X, Y, options):
    """``train_neural_net`` returns the network document; load it."""
    return IMP.bff.NeuralNet(IMP.bff.train_neural_net_arrays(X, Y, options).get_network())


def fd_grad(f, x, h=1e-6):
    """Central-difference gradient of a scalar function, for the checks below."""
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


# %%
# XOR, the classic
# ----------------
# Not linearly separable, so a hidden layer is the whole point. Trained with
# ``train_neural_net`` (Adam, mini-batches, sklearn's defaults) as a regression onto
# {0, 1}; the four corners plus jitter make a small training set.
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_xor = np.array([[0], [1], [1], [0]], dtype=float)
Xtr = np.repeat(X_xor, 60, axis=0) + rng.normal(scale=0.08, size=(240, 2))
Ytr = np.repeat(Y_xor, 60, axis=0)

opt = IMP.bff.NeuralNetTrainOptions()
opt.hidden_layer_sizes = [8]
opt.activation = "tanh"
opt.max_iter = 400
opt.batch_size = 32
opt.learning_rate = 1e-2
opt.early_stopping = False
opt.seed = 1
xor = train(Xtr, Ytr, opt)
print("XOR:", np.round(predict(xor, X_xor).ravel(), 3))

# %%
# A sine, and its derivative from the network
# -------------------------------------------
# ``jacobian`` returns ``dy/dx``; for a 1-D input that is the slope. The
# network was never told about the derivative, so this is a check of how well
# a regression fit extrapolates to the *derivative* of the target -- usually
# not very, which motivates the Sobolev fit at the end.
xs = rng.uniform(-np.pi, np.pi, size=(400, 1))
ys = np.sin(xs)
opt.hidden_layer_sizes = [16, 16]
opt.max_iter = 300
opt.learning_rate = 3e-3
sine = train(xs, ys, opt)

xg = np.linspace(-np.pi, np.pi, 200)[:, None]
slope = np.array([sine.jacobian(list(x))[0, 0] for x in xg])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(xg, np.sin(xg), "k-", label="sin x")
ax[0].plot(xg, predict(sine, xg), "C0--", label="network")
ax[0].set_title("value")
ax[1].plot(xg, np.cos(xg), "k-", label="cos x")
ax[1].plot(xg, slope, "C0--", label="network dy/dx (jacobian)")
ax[1].set_title("derivative, from the network")
for a in ax:
    a.legend(fontsize=8)
    a.set_xlabel("x")
plt.tight_layout()

# %%
# The derivatives are exact
# -------------------------
# ``backward`` for an arbitrary loss ``L = <W, y>``: its ``dL/dparams`` and
# ``dL/dx`` against central differences. The stored input/output scalers are
# part of the chain rule; ``backward`` handles them.
net = sine
Xc = rng.uniform(-3, 3, size=(5, 1))
W = rng.normal(size=(5, 1))
dparams, dx = net.backward(Xc, W)


def loss_of_params(p):
    net.set_parameters(p)
    return float(np.sum(W * predict(net, Xc)))


p0 = net.get_parameters()
fd_p = fd_grad(loss_of_params, p0)
net.set_parameters(p0)
fd_x = fd_grad(lambda xf: float(np.sum(W * predict(net, xf.reshape(-1, 1)))), Xc.ravel())
print("max |dL/dparams - FD| = %.2e   max |dL/dx - FD| = %.2e"
      % (np.abs(dparams - fd_p).max(), np.abs(dx.ravel() - fd_x).max()))

# %%
# ``hessian`` against a finite difference of ``jacobian``, at a few points:
x0 = np.array([0.7])
H = net.hessian(list(x0), 0)[0, 0]
H_fd = (net.jacobian(list(x0 + 1e-4))[0, 0] - net.jacobian(list(x0 - 1e-4))[0, 0]) / 2e-4
print("d2y/dx2 at 0.7: exact %.6f, FD %.6f" % (H, H_fd))

# %%
# Sobolev training: fit the function *and* its derivative
# -------------------------------------------------------
# A network trained on values alone gets the derivative wrong wherever the
# data are sparse. When the derivative is known -- from physics, or because it
# is the quantity that matters -- it can be trained on. The loss is
#
#   L = mean (y - sin x)² + mean (dy/dx - cos x)²
#
# and its gradient with respect to the weights comes from
# ``backward_derivatives``: the adjoint of ``y`` is ``2 (y - sin x)/n``, the
# adjoint of ``J v`` (with ``v = 1``) is ``2 (dy/dx - cos x)/n``. L-BFGS from
# SciPy does the rest, on the flat ``get_parameters()`` vector. Only 12 training
# points -- the derivative constraint is what makes that enough.
x_few = np.linspace(-np.pi, np.pi, 12)[:, None]
ones = np.ones_like(x_few)


def sobolev(p, with_derivative=True):
    net.set_parameters(p)
    y, dy, _ = net.predict_derivatives(x_few, ones, order=1)
    r0 = y - np.sin(x_few)
    r1 = dy - np.cos(x_few)
    loss = np.mean(r0 ** 2) + (np.mean(r1 ** 2) if with_derivative else 0.0)
    g, _, _ = net.backward_derivatives(x_few, ones, 2 * r0 / r0.size,
                                       dY1=2 * r1 / r1.size if with_derivative else None)
    return loss, g


start = net.get_parameters()
res_plain = minimize(sobolev, start, args=(False,), jac=True, method="L-BFGS-B",
                     options={"maxiter": 500})
plain = predict(net, xg).ravel(), np.array([net.jacobian(list(x))[0, 0] for x in xg])
res_sob = minimize(sobolev, start, jac=True, method="L-BFGS-B", options={"maxiter": 500})
sob = predict(net, xg).ravel(), np.array([net.jacobian(list(x))[0, 0] for x in xg])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(xg, np.sin(xg), "k-", lw=1, label="sin x")
ax[0].plot(xg, plain[0], "C1--", label="values only (12 points)")
ax[0].plot(xg, sob[0], "C0-", label="values + derivative")
ax[0].plot(x_few, np.sin(x_few), "ko", ms=3)
ax[1].plot(xg, np.cos(xg), "k-", lw=1, label="cos x")
ax[1].plot(xg, plain[1], "C1--", label="values only")
ax[1].plot(xg, sob[1], "C0-", label="values + derivative")
ax[0].set_title("fit"), ax[1].set_title("dy/dx")
for a in ax:
    a.legend(fontsize=8)
    a.set_xlabel("x")
plt.tight_layout()
print("max |dy/dx - cos| on the grid: values-only %.3f, Sobolev %.3f"
      % (np.abs(plain[1] - np.cos(xg).ravel()).max(), np.abs(sob[1] - np.cos(xg).ravel()).max()))
plt.show()

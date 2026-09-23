---
okf_version: "0.2"
---

# Free-energy landscapes from photons (`FRETLandscapeModel`)

The method is Dingeldein & Covino, *A differentiable photon-by-photon
likelihood for continuous free-energy landscapes and diffusion coefficients
from single-molecule FRET*, arXiv:2608.21061 (2026). The donor-acceptor
distance `x(t)` diffuses on a landscape `u(x)` (kT) with diffusion
coefficient `D`. Each photon is scored by its colour and its waiting time, and
the hidden path is integrated out exactly. The output is a continuous
landscape, `D`, the channel brightnesses and backgrounds, and Laplace error
bars on each.

It lives in imp.bff by the lead's placement decision (2026-09-23). The unknown
is a landscape over a distance coordinate, the consumer is a bff model, and
when input and consumer disagree the consumer wins. The photons come in as
plain arrays (times, channels, CSR trace offsets). There is no tttrlib data
type in the interface.

## The pieces

| Paper | Here |
|---|---|
| SqRA generator `Q`, Eq. 11 | `sqra_generator`, `sqra_stationary_distribution` (`FRETLandscapeGrid.h`) |
| Symmetrised killed generator, Eq. 12 | `sqra_symmetric_diagonal`: its off-diagonal is the constant `D/h^2` |
| Its diagonalisation | `symmetric_tridiagonal_eigen`: the QL kernel formerly private to `DiffusionSolverKrylov.cpp`, now `internal/TridiagonalEigen.h` |
| Natural cubic spline, Eq. 15 | `NaturalCubicSpline` |
| Brightness `lambda_c(x)`, Eq. 4 | `FRETLandscapeModel` constructor: `donor_fraction`, `acceptor_fraction` per channel; `a_c`, `beta_c` in theta |
| Normalised forward filter, Eq. 14 | `log_likelihood`, `trace_log_likelihoods` |
| Autodiff gradient | `log_likelihood_gradient`: adjoint sweep, exact |
| Priors, Eqs. 18-19 and the Gamma prior | `set_roughness_weight`, `set_anchor_sigma`, `set_background_prior`; `log_prior`, `prior_precision` |
| L-BFGS on the log posterior | `fit`, `FRETLandscapeFitOptions`, `FRETLandscapeFit` |
| Initial guess, Sec. III C | `initial_guess`, `FRETLandscapeInitialGuessOptions` |
| Euler-Maruyama + Poisson photons, Sec. III F | `simulate` returns `FRETLandscapePhotons` |
| Empirical Fisher and Laplace, Eqs. 20-22 | `trace_scores`, `laplace` returns `FRETLandscapeLaplace` |
| Smoothing and FFBS, App. D | `posterior_marginals`, `posterior_mean_trajectory`, `sample_trajectory` |

theta is `[mu_1..mu_K, log D, log a_c..., log beta_c...]`. `pack_parameters`
builds it.

## Design decisions

* **Eigen-coordinates, batched.** The likelihood uses one decomposition
  `A = Psi diag(nu) Psi^T` per theta. Each photon step is
  `w <- Psi^T (lambda_c o Psi (e^{nu tau} o w))`. Traces are sorted by length
  and processed in batches (`set_batch_size`, default 16), so a step is a
  GEMM. A trace that has already ended is padded with an identity step, which
  is exact.
* **Exact gradient without differentiating the eigenvectors.** The derivative
  of `exp(A tau)` along `dA` is `Psi (G o Psi^T dA Psi) Psi^T`, with
  `G_kl = (e_k - e_l)/(nu_k - nu_l)` (Daleckii-Krein). The backward sweep adds
  `(lhat o e) w^T - lhat (e o w)^T` into one `M x M` matrix; the division by
  `nu_k - nu_l` happens once at the end. Eigenvalue pairs closer than
  `1e-5 / tau_max` use a series form instead. Only three diagonals of
  `Psi W Psi^T` are ever needed, because only `diag(A)` depends on `u` and
  `lambda`, and the off-diagonal depends only on `D`. Cost: about 3.5x one
  likelihood evaluation.
* **The optimiser is tttrlib's.** `fit` drives `tttrlib/i_lbfgs.h`
  (header-only, std-only includes), included under `IMP_BFF_HAS_TTTRLIB`. This
  was the owner's ruling (2026-09-23): no ported copy in bff. **Without
  tttrlib, `fit` throws.** The likelihood, gradient, priors, simulation,
  Laplace and smoothing do not need tttrlib.
* **Preconditioning is what makes that L-BFGS usable.** The i_lbfgs history
  is fixed at 7, and the parameters differ in scale by orders of magnitude.
  So each block of `patience` iterations is run in whitened coordinates,
  `theta = theta0 + U^-1 z`, where `U^T U` is the damped Laplace precision
  (empirical Fisher + prior precision + `precondition_damping * I`). Measured
  on 237k photons, M = 80: without preconditioning, 300 iterations were still
  ~200 nats short; with it, the MAP took 75 iterations. Without the damping, a
  fit started on a landscape with 40 kT empty walls takes enormous steps along
  the unconstrained wall knots and backtracks for most of an hour.
* **The patience criterion is evaluated between L-BFGS blocks,** and the
  history restarts at each block. i_lbfgs has no callback to stop from
  inside.
* **The initial guess generalises the paper's `E_obs` inversion.** Each bin is
  given the grid distance that maximises the multinomial colour likelihood
  `sum_c n_c log(lambda_c / lambda_tot)`. For two channels this is the
  paper's mapping of `E_obs` through `lambda_r/(lambda_r+lambda_g)`. Amplitudes
  are refined from the count rates under the estimated landscape (two passes).

## Validation

Tests are in `test/landscape/`, 18 of them, about 2 s in total:

* the SqRA columns sum to zero, and the Boltzmann weights are the stationary
  distribution and satisfy detailed balance to 1e-12;
* the forward filter matches a brute-force `scipy.linalg.expm` recursion
  to 1e-10;
* with M = 2, it matches a hand-computed two-state Markov-modulated Poisson
  likelihood (closed-form 2x2 exponentials) to 1e-11;
* the gradient matches central differences for every parameter, on a small
  grid and on a stiff, paper-scale one (`D/h^2 ~ 200`, 30 photons/ms);
* the prior gradient and precision match finite differences;
* the smoothing marginals and the lag-one FFBS joint match brute force;
* the empirical Fisher (`trace_scores`) sums to the gradient.

A separate check, not a test: at the MAP, the empirical-Fisher `sigma_logD`
was 0.41 against 0.37 from the observed Hessian (finite differences of the
exact gradient). The diagonal ratios were 0.7-1.07.

**Two-well reproduction (paper Fig. 2)** (2026-09-23; the machine was at load ~100, so the timings are pessimistic). Run with
`examples/spectroscopy/fret_landscape_two_well.py`. Truth:
`u = 4.5 (z^2-1)^2 - 0.5 z`, `z = (x-5.9)/1.4`, D = 1.5 nm^2/ms, and the
Table I photophysics. The landscape comparison covers the region where
`u* - min u* < 6 kT`, with the offset aligned on the true Boltzmann weights.

| run | photons | D (nm^2/ms) | within 1 sigma / 2 sigma | barrier (kT), true 5.00 | photophysics |
|---|---|---|---|---|---|
| default: 60 traces, M = 80, dt = 5e-4 | 355k | 2.80 +- 1.15 (1.1 sigma) | 98% / 100% | 5.60 +- 0.99 | all within 1% |
| 300 traces, M = 100, dt = 1e-4 | 1.78M | 1.71 +- 0.24 (0.9 sigma) | 98% / 100% | 5.50 +- 0.73 | all within 1.5% |

Both fits stopped on convergence after 150 iterations: 5 min (default) and
33 min (300 traces). Both MAPs lie above the truth in log posterior, by 32 and
28 nats. The paper reports `D = 1.6 +- 0.1` for its 300-trace data set; its
grid is 200 points and its dt is 5e-6 ms. Our sigma_D of 0.24 is wider than
that, but the empirical Fisher agrees with the observed Hessian here (see
above), so the width belongs to this data set and discretisation. It is not
an artefact of the Fisher estimate. The largest landscape errors (2.0-2.4 kT)
sit at the populated region's edge, on the steep walls, where the band is
also widest.

## Where to pick this up

* **Speed.** At M = 100 and 1.5M photons, one likelihood evaluation takes
  about 2.2 s and a gradient about 7.7 s, single-threaded. `built_with_openmp()`
  is false in the local build, so the batch loop's `omp parallel for` is inert.
  Enabling OpenMP is the first factor. The second is to store `Y = Psi(e o w)`
  from the forward pass rather than recompute it in the backward pass: one GEMM
  of five, at the cost of twice the memory.
* **No graph node yet.** `FitObjective`/`FitMinimizer` are least-squares only;
  a scalar-objective node wrapping `FRETLandscapeModel` is the missing link to
  the bff model graph.
* **Not done from the paper:** the curvature-weight scan by held-out
  likelihood (Fig. S1; `trace_log_likelihoods` has what it needs), the
  experimental-design scans (Fig. 4: `laplace` at the truth over a range of
  `k_tot`), transition-path sampling (App. B), the three-state landscape
  (Fig. 5), and dark gaps before the first and after the last photon
  (dropped, as in Eq. 14).
* **Bias to watch.** The simulator is Euler-Maruyama at the caller's `dt`
  (the paper uses 5e-6 ms), and SqRA has its own grid error. Neither has been
  swept for its effect on the recovered `D`.

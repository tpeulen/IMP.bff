"""PRD-155 steps 2 and 4: the weighted sum of p(R) with its derivatives, and the softplus response
floor. Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen; the observation term this file
first tested moved out of the decay posterior into the factor graph the same day (external FRET
measurements are factors there, in the docking restraint's vocabulary --
test_bayesian_decay_factor_graph.py).

**The weighted sum.** Many summaries of the distance distribution p(R) are weighted sums
`sum_j p_j w_j`: the mean R/R0 (weights R/R0), the mean FRET efficiency (weights
1/(1 + (R/R0)^6)). `bayesian_decay_weighted_sum` gives one with its exact gradient and Hessian in
the free coordinates of p(R); the mean-R/R0 tilt of the Tierney-Kadane moments uses it.

**The floor.** The instrument response (IRF: the measured shape of an infinitely short light
pulse) is shifted by a fraction of a time bin with an FFT phase ramp; that rings, and small
negative values appear. `hard` clamps them at zero (derivative zero there); `softplus` replaces the
clamp by a smooth function so the derivative is defined everywhere.

What is checked, and what makes each check fail:
* the mean-R/R0 tilt, now computed through `bayesian_decay_weighted_sum`, is bit-identical to
  the code it replaced (copied below as it stood at 47512f2f9) -- any reordering fails it;
* the log posterior is `loglik + logprior` exactly, the split `bayesian_decay_evaluate` reports;
* the weighted sum's gradient equals central differences; a gradient perturbed by 1e-3 of its
  scale, run through the same comparison, must fail it (the control);
* the softplus floor's shift tangent equals central differences of the basis at zero shift,
  the case where the hard clamp's choice is round-off; the hard clamp is run the same way and
  reported (it is the case the change exists for).
"""
import json
import os
import tempfile

import numpy as np
import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/bayesian/BayesianDecayPosterior.h>
#include <IMP/bff/fit/FitStatistics.h>
#include <cstdio>
#include <fstream>
#include <random>
using namespace IMP::bff;

static void num(double v) { if (std::isfinite(v)) std::printf("%.17g", v); else std::printf("null"); }
static void vec(const char* name, const std::vector<double>& v) {
  std::printf(", \"%s\": [", name);
  for (std::size_t i = 0; i < v.size(); ++i) { if (i) std::printf(","); num(v[i]); }
  std::printf("]");
}

// bayesian_decay_mean_rel_tilt as it stood at 47512f2f9, before PRD-155 factored it out
static BayesianDecayTilt old_tilt(const BayesianDecayExperiment& f, const BayesianDecayTensors& e,
                                  const std::vector<double>& th, double power, double eps) {
    const std::size_t dim = f.dim;
    BayesianDecayTilt T;
    T.grad.assign(dim, 0.0);
    T.hess.assign(dim * dim, 0.0);
    const std::vector<double> p = bayesian_decay_distribution(e, bayesian_decay_unpack(f, th));
    const std::vector<double>& rel = f["rel"].d;
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(f, "c");
    const BayesianDecayArray& Q = f["Q_c"];
    const std::size_t nR = p.size(), nc = e.spl->shape[1], nz = Q.shape[1];
    std::vector<double> M(nR * nz, 0.0);
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t a = 0; a < nc; ++a) {
        const double s_ = e.spl->d[j * nc + a];
        if (s_ != 0.0) for (std::size_t b = 0; b < nz; ++b) M[j * nz + b] += s_ * Q.d[a * nz + b];
    }
    double fv = 0.0;
    for (std::size_t j = 0; j < nR; ++j) fv += p[j] * rel[j];
    T.f = fv;
    std::vector<double> gj(nR), G(nz, 0.0), P(nz, 0.0);
    for (std::size_t j = 0; j < nR; ++j) gj[j] = p[j] * (rel[j] - fv);
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b) {
        G[b] += gj[j] * M[j * nz + b];
        P[b] += p[j] * M[j * nz + b];
    }
    const double d = fv + eps, k1 = power / d;
    T.value = power * std::log(d);
    for (std::size_t b = 0; b < nz; ++b) T.grad[vc.off + b] = k1 * G[b];
    std::vector<double> W(nz * nz, 0.0);
    for (std::size_t j = 0; j < nR; ++j) {
        if (gj[j] == 0.0) continue;
        for (std::size_t a = 0; a < nz; ++a) {
            const double t = gj[j] * M[j * nz + a];
            for (std::size_t b = 0; b < nz; ++b) W[a * nz + b] += t * M[j * nz + b];
        }
    }
    const double k2 = power / (d * d);
    for (std::size_t a = 0; a < nz; ++a) for (std::size_t b = 0; b < nz; ++b) {
        const double d2f = W[a * nz + b] - G[a] * P[b] - P[a] * G[b];
        T.hess[(vc.off + a) * dim + vc.off + b] = k1 * d2f - k2 * G[a] * G[b];
    }
    return T;
}

static std::vector<double> read_theta(const std::string& dir, std::size_t dim) {
  std::vector<double> t(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(t.data()), std::streamsize(dim * sizeof(double)));
  return t;
}

static double summary(const BayesianDecayExperiment& ex, const BayesianDecayTensors& E, const std::vector<double>& th,
                      const std::vector<double>& w) {
  return bayesian_decay_weighted_sum(ex, E, th, w, false).f;
}

int main(int, char** argv) {
  const std::string plain_dir = argv[1], soft_dir = argv[2];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(plain_dir, ex);
  const std::size_t dim = ex.dim;
  const std::vector<double> truth = read_theta(plain_dir, dim);
  const BayesianDecayTensors E(ex);
  std::printf("{\"dim\": %zu", dim);

  // 1. the tilt, old against new, bit for bit
  {
    const BayesianDecayTilt a = old_tilt(ex, E, truth, 2.0, 1e-6), b = bayesian_decay_mean_rel_tilt(ex, E, truth, 2.0, 1e-6);
    int same = a.f == b.f && a.value == b.value && a.grad == b.grad && a.hess == b.hess;
    std::printf(", \"tilt_identical\": %d", same);
  }
  // 2. the log posterior and its split
  {
    const BayesianDecayValues v = bayesian_decay_unpack(ex, truth);
    const double before = bayesian_decay_log_likelihood(ex, bayesian_decay_expected_counts(ex, E, v)) + bayesian_decay_log_prior(ex, truth, false).lp;
    const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, truth);
    std::printf(", \"plain_identical\": %d, \"split_exact\": %d", int(before == bayesian_decay_log_posterior(ex, E, truth)),
                int(pt.logpost == pt.loglik + pt.logprior));
  }
  // 3. the weighted sum's gradient against central differences (FRET-efficiency weights)
  {
    std::vector<double> w(ex["rel"].size());
    for (std::size_t j = 0; j < w.size(); ++j) { const double r = ex["rel"].d[j]; w[j] = 1.0 / (1.0 + std::pow(r, 6)); }
    const BayesianDecayWeightedSum S = bayesian_decay_weighted_sum(ex, E, truth, w, false);
    std::vector<double> g(S.dz.size()), gfd(S.dz.size());
    for (std::size_t b = 0; b < S.dz.size(); ++b) {
      const double h = 1e-6;
      std::vector<double> tp = truth, tm = truth;
      tp[S.off + b] += h; tm[S.off + b] -= h;
      gfd[b] = (bayesian_decay_weighted_sum(ex, E, tp, w, false).f - bayesian_decay_weighted_sum(ex, E, tm, w, false).f) / (2.0 * h);
      g[b] = S.dz[b];
    }
    vec("ws_grad", g); vec("ws_grad_fd", gfd);
  }
  // 5. the response floor at zero shift: tangent against central differences, softplus and hard
  {
    BayesianDecayExperiment es;
    bayesian_decay_experiment_load(soft_dir, es);
    const std::vector<double>& measured = es["response_0"].d;
    const double h = 1e-4;
    for (int mode = 0; mode < 2; ++mode) {
      BayesianResponseOptions opt;
      opt.floor = mode == 0 ? BayesianResponseFloor::softplus : BayesianResponseFloor::hard;
      opt.softplus_scale = es.softplus_scale;
      const BayesianResponseBasis b0 = bayesian_response_basis(es.kernel(), measured, 0.05, 0.0, true, opt);
      const BayesianResponseBasis bp = bayesian_response_basis(es.kernel(), measured, 0.05, h, false, opt);
      const BayesianResponseBasis bm = bayesian_response_basis(es.kernel(), measured, 0.05, -h, false, opt);
      double err = 0.0, scale = 0.0;
      for (std::size_t i = 0; i < b0.B.size(); ++i) {
        const double fd = (bp.B[i] - bm.B[i]) / (2.0 * h);
        err = std::max(err, std::fabs(fd - b0.dB_shift[i]));
        scale = std::max(scale, std::fabs(fd));
      }
      std::printf(", \"%s_tangent_err\": %.17g, \"%s_tangent_scale\": %.17g", mode == 0 ? "softplus" : "hard", err,
                  mode == 0 ? "softplus" : "hard", scale);
    }
  }
  std::printf("}\n");
  return 0;
}
"""


def _edit_manifest(directory, **changes):
    path = os.path.join(directory, "manifest.json")
    with open(path) as fh:
        m = json.load(fh)
    m.update(changes)
    with open(path, "w") as fh:
        json.dump(m, fh)


@pytest.fixture(scope="module")
def result():
    root = tempfile.mkdtemp()
    plain, soft = (os.path.join(root, n) for n in ("plain", "soft"))
    for d in (plain, soft):
        syn.write(d)
    _edit_manifest(soft, response_floor="softplus")
    return run_driver(DRIVER, args=[plain, soft], timeout=1800)


def test_the_tilt_is_unchanged_bit_for_bit(result):
    assert result["tilt_identical"] == 1


def test_the_log_posterior_is_likelihood_plus_prior(result):
    assert result["plain_identical"] == 1
    assert result["split_exact"] == 1


def _grad_close(g, gfd):
    g, gfd = np.asarray(g), np.asarray(gfd)
    return np.max(np.abs(g - gfd)) <= 1e-5 * max(1.0, np.max(np.abs(gfd)))


def test_the_weighted_sum_gradient_is_exact_and_the_control_fails(result):
    g, gfd = np.asarray(result["ws_grad"]), np.asarray(result["ws_grad_fd"])
    assert np.max(np.abs(gfd)) > 1e-4          # the summary does depend on the coordinates
    assert _grad_close(g, gfd)
    perturbed = g + 1e-3 * np.max(np.abs(g)) * np.sign(np.random.default_rng(0).normal(size=g.size))
    assert not _grad_close(perturbed, gfd)


def test_the_softplus_floor_has_an_exact_tangent_at_zero_shift(result):
    r = result
    print({k: r[k] for k in r if "tangent" in k})
    assert r["softplus_tangent_err"] <= 1e-6 * r["softplus_tangent_scale"]

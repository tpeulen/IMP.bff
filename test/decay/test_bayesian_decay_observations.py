"""PRD-155 steps 2-4: observations of a weighted sum of p(R), the log posterior split, and the
softplus response floor. Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.

**What an observation is.** A FRET experiment measured elsewhere (e.g. an ensemble efficiency
from intensities) reports one number about the distance distribution p(R): a weighted sum
`sum_j p_j w_j`. With `w_j = 1 / (1 + (R_j/R0)^6)` it is the mean FRET efficiency; with
`w_j = R_j/R0` the mean distance in units of the Foerster radius R0. Its measurement error
`sigma` makes it one Gaussian likelihood factor next to the TCSPC histograms (the counts of
photons per arrival-time bin) -- the generic replacement for an exact-equality constraint.

**What the floor is.** The instrument response (IRF: the measured shape of an infinitely short
light pulse) is shifted by a fraction of a time bin with an FFT phase ramp; that rings, and
small negative values appear. `hard` clamps them at zero (derivative zero there); `softplus`
replaces the clamp by a smooth function so the derivative is defined everywhere.

What is checked, and what makes each check fail:
* the mean-R/R0 tilt, now computed through `bayesian_decay_weighted_sum`, is bit-identical to
  the code it replaced (copied below as it stood at 47512f2f9) -- any reordering fails it;
* without observations the log posterior is bit-identical to `loglik + logprior` as before;
* the observations' analytic gradient equals central differences; a gradient perturbed by
  1e-3 of its scale, run through the same comparison, must fail it (the control);
* `logpost == loglik + logprior + logobs`, and `bayesian_decay_log_posterior` and
  `bayesian_decay_log_posterior_and_gradient` agree with `bayesian_decay_evaluate`;
* a fit with a tight observation moves the summary to the observed value, a fit without it
  does not get there (so the check can fail); both fits' Poisson deviance per degree of
  freedom and runs test are reported (the observation must not wreck the fit to the counts);
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
  const std::string plain_dir = argv[1], obs_dir = argv[2], soft_dir = argv[3];
  const double target_shift = std::atof(argv[4]);
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
  // 2. no observations: the log posterior is what it was
  {
    const BayesianDecayValues v = bayesian_decay_unpack(ex, truth);
    const double before = bayesian_decay_log_likelihood(ex, bayesian_decay_expected_counts(ex, E, v)) + bayesian_decay_log_prior(ex, truth, false).lp;
    const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, truth);
    std::printf(", \"plain_identical\": %d, \"plain_logobs\": %.17g", int(before == bayesian_decay_log_posterior(ex, E, truth)), pt.logobs);
  }
  // 3. with observations: gradient against central differences, the split, the three doors agree
  BayesianDecayExperiment eo;
  bayesian_decay_experiment_load(obs_dir, eo);
  const BayesianDecayTensors Eo(eo);
  {
    std::vector<double> g(dim, 0.0), gfd(dim, 0.0);
    bayesian_decay_log_observations(eo, Eo, truth, g.data());
    for (std::size_t i = 0; i < dim; ++i) {
      const double h = 1e-6 * std::max(1.0, std::fabs(truth[i]));
      std::vector<double> tp = truth, tm = truth;
      tp[i] += h; tm[i] -= h;
      gfd[i] = (bayesian_decay_log_observations(eo, Eo, tp) - bayesian_decay_log_observations(eo, Eo, tm)) / (2.0 * h);
    }
    vec("obs_grad", g); vec("obs_grad_fd", gfd);
    const BayesianDecayPoint pt = bayesian_decay_evaluate(eo, Eo, truth);
    std::vector<double> g2;
    const double lp2 = bayesian_decay_log_posterior_and_gradient(eo, Eo, truth, g2);
    std::printf(", \"n_obs\": %zu, \"logpost\": %.17g, \"loglik\": %.17g, \"logprior\": %.17g, \"logobs\": %.17g, \"logpost_direct\": %.17g, \"logpost_and_grad\": %.17g",
                eo.observations.size(), pt.logpost, pt.loglik, pt.logprior, pt.logobs, bayesian_decay_log_posterior(eo, Eo, truth), lp2);
    double gdiff = 0.0;
    for (std::size_t i = 0; i < dim; ++i) gdiff = std::max(gdiff, std::fabs(pt.grad[i] - g2[i]));
    std::printf(", \"grad_doors_maxdiff\": %.17g", gdiff);
  }
  // 4. fits on data drawn from the model: without the observation, and with it at a shifted target
  {
    const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, truth));
    std::mt19937_64 rng(3);
    std::vector<double> y(lam.size());
    for (std::size_t i = 0; i < lam.size(); ++i) y[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
    ex.arrays["y"].d = y;
    const std::vector<double> w = bayesian_decay_named_weights("fret_efficiency", ex["rel"].d);
    const BayesianDecayFit F0 = bayesian_decay_fit_node(ex, E, truth, 1.0, 400);
    const double f0 = summary(ex, E, F0.theta, w);
    BayesianDecayExperiment e1 = ex;            //: the same data, plus the observation
    BayesianDecayObservation ob; ob.name = "ext_E"; ob.weights = w; ob.value = f0 + target_shift; ob.sigma = 1e-3;
    e1.observations = {ob};
    const BayesianDecayTensors E1(e1);
    const BayesianDecayFit F1 = bayesian_decay_fit_node(e1, E1, F0.theta, 1.0, 400);
    const double f1 = summary(e1, E1, F1.theta, w);
    const PoissonGoodnessOfFit G0 = poisson_goodness_of_fit(y, F0.pt.lam, ex["mask"].d, ex.data_keys.size(), double(dim));
    const PoissonGoodnessOfFit G1 = poisson_goodness_of_fit(y, F1.pt.lam, ex["mask"].d, ex.data_keys.size(), double(dim));
    double runs0 = 1.0, runs1 = 1.0;
    for (const auto& h : G0.histograms) runs0 = std::min(runs0, h.runs_p);
    for (const auto& h : G1.histograms) runs1 = std::min(runs1, h.runs_p);
    std::printf(", \"fit\": {\"converged0\": %d, \"converged1\": %d, \"f0\": %.17g, \"f1\": %.17g, \"target\": %.17g, \"sigma\": %.17g,"
                " \"dpd0\": %.17g, \"dpd1\": %.17g, \"dof\": %.17g, \"z0\": %.17g, \"z1\": %.17g, \"runs_min0\": %.17g, \"runs_min1\": %.17g,"
                " \"logobs1\": %.17g}",
                int(F0.converged), int(F1.converged), f0, f1, ob.value, ob.sigma, G0.deviance_per_dof, G1.deviance_per_dof, G0.dof,
                G0.z, G1.z, runs0, runs1, F1.pt.logobs);
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
    plain, obs, soft = (os.path.join(root, n) for n in ("plain", "obs", "soft"))
    for d in (plain, obs, soft):
        syn.write(d)
    rel = np.linspace(0.4, 2.0, syn.NR)
    _edit_manifest(obs, observations=[
        dict(name="ext_E", weights="fret_efficiency", value=0.4, sigma=0.05),
        dict(name="ext_rel", weights=list(rel ** 2), value=1.2, sigma=0.1),
    ])
    _edit_manifest(soft, response_floor="softplus")
    return run_driver(DRIVER, args=[plain, obs, soft, "0.05"], timeout=1800)


def test_the_tilt_is_unchanged_bit_for_bit(result):
    assert result["tilt_identical"] == 1


def test_without_observations_the_log_posterior_is_unchanged(result):
    assert result["plain_identical"] == 1
    assert result["plain_logobs"] == 0.0


def _grad_close(g, gfd):
    g, gfd = np.asarray(g), np.asarray(gfd)
    return np.max(np.abs(g - gfd)) <= 1e-5 * max(1.0, np.max(np.abs(gfd)))


def test_the_observation_gradient_is_exact_and_the_control_fails(result):
    g, gfd = np.asarray(result["obs_grad"]), np.asarray(result["obs_grad_fd"])
    assert np.max(np.abs(gfd)) > 1e-3          # the observations do depend on theta
    assert _grad_close(g, gfd)
    perturbed = g + 1e-3 * np.max(np.abs(g)) * np.sign(np.random.default_rng(0).normal(size=g.size))
    assert not _grad_close(perturbed, gfd)


def test_the_log_posterior_splits_and_the_doors_agree(result):
    r = result
    assert r["n_obs"] == 2
    assert r["logobs"] != 0.0
    assert abs(r["logpost"] - (r["loglik"] + r["logprior"] + r["logobs"])) <= 1e-12 * abs(r["logpost"])
    assert abs(r["logpost"] - r["logpost_direct"]) <= 1e-9 * abs(r["logpost"])
    assert abs(r["logpost"] - r["logpost_and_grad"]) <= 1e-9 * abs(r["logpost"])
    assert r["grad_doors_maxdiff"] <= 1e-6


def test_an_observation_moves_the_fit_to_the_measured_value(result):
    f = result["fit"]
    assert f["converged0"] == 1 and f["converged1"] == 1
    # without the observation the fit sits 0.05 from the target: 50 sigma, so the check can fail
    assert abs(f["f0"] - f["target"]) > 10 * f["sigma"]
    assert abs(f["f1"] - f["target"]) < 3 * f["sigma"]
    # the counts are still fitted: deviance per dof within the Poisson reference, no residual structure
    print(json.dumps(f, indent=1))
    assert abs(f["z0"]) < 3 and abs(f["z1"]) < 3
    assert f["runs_min0"] > 0.001 and f["runs_min1"] > 0.001


def test_the_softplus_floor_has_an_exact_tangent_at_zero_shift(result):
    r = result
    print({k: r[k] for k in r if "tangent" in k})
    assert r["softplus_tangent_err"] <= 1e-6 * r["softplus_tangent_scale"]

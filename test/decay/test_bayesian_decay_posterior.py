"""PRD-142 step 4: the posterior of a Bayesian decay model, and its fit.

`BayesianDecayPosterior.h` adds the prior (per-variable families, the spectrum's
continuity prior, the P-spline on log p(R/R0)) with its gradient and Hessian, the
Poisson likelihood, Fisher scoring to the mode through `DampedNewton`, the Laplace
evidence and p(R/R0) with its delta-method band. On the synthetic experiment of
`bayesian_decay_synthetic.py`:

* the prior's gradient and Hessian, and the posterior's gradient, are the
  derivatives of what they differentiate (central differences);
* counts drawn from the model at a known point and fitted from a perturbed start:
  the fit converges, its evidence is finite, its covariance positive definite, and
  its expected counts pass the goodness-of-fit test (|z| < 3 overall and per
  histogram);
* counts with a structure no decay model makes -- a 20 % ripple of 16 channels --
  must FAIL that test, or the test above is not evidence of anything. (Counts
  drawn with every grid lifetime 20 % longer were tried first and are NOT a
  control here: the free lifetime spectrum re-weights the grid and absorbs the
  stretch, z +2.1.)

Written 2026-09-14 (ucfret prompt 432).
"""

import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecayPosterior.h>
#include <IMP/bff/BayesianGoodnessOfFit.h>
#include <cstdio>
#include <random>
using namespace IMP::bff;

static std::vector<double> simulate(const BayesianDecayExperiment& ex, const std::vector<double>& th, std::uint64_t seed) {
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, th));
  std::mt19937_64 rng(seed);
  std::vector<double> y(lam.size());
  for (std::size_t i = 0; i < y.size(); ++i) y[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
  return y;
}

int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::size_t dim = ex.dim;
  std::vector<double> truth(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(truth.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);

  // derivatives of the prior and of the posterior
  const BayesianDecayPrior P = bayesian_decay_log_prior(ex, truth, true);
  double g_err = 0, g_pk = 0, H_err = 0, H_pk = 0, post_err = 0, post_pk = 0;
  ex.arrays["y"].d = simulate(ex, truth, 7);
  const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, truth);
  for (std::size_t c = 0; c < dim; ++c) {
    const double h = 1e-5;
    std::vector<double> tp = truth, tm = truth; tp[c] += h; tm[c] -= h;
    const BayesianDecayPrior Pp = bayesian_decay_log_prior(ex, tp, true), Pm = bayesian_decay_log_prior(ex, tm, true);
    const double fd = (Pp.lp - Pm.lp) / (2 * h);
    g_pk = std::max(g_pk, std::fabs(P.g[c])); g_err = std::max(g_err, std::fabs(fd - P.g[c]));
    for (std::size_t r = 0; r < dim; ++r) {
      const double fh = (Pp.g[r] - Pm.g[r]) / (2 * h);
      H_pk = std::max(H_pk, std::fabs(P.H[r * dim + c])); H_err = std::max(H_err, std::fabs(fh - P.H[r * dim + c]));
    }
    const double hp = 1e-6;
    std::vector<double> sp = truth, sm = truth; sp[c] += hp; sm[c] -= hp;
    const double fpost = (bayesian_decay_log_posterior(ex, E, sp) - bayesian_decay_log_posterior(ex, E, sm)) / (2 * hp);
    post_pk = std::max(post_pk, std::fabs(pt.grad[c])); post_err = std::max(post_err, std::fabs(fpost - pt.grad[c]));
  }

  // the fit, on data from the model and on data from a model with longer lifetimes
  std::mt19937_64 rng(11); std::normal_distribution<double> N(0.0, 0.05);
  std::vector<double> start = truth; for (double& t : start) t += N(rng);
  auto fit_report = [&](const char* name, const std::vector<double>& y) {
    ex.arrays["y"].d = y;
    const BayesianDecayFit F = bayesian_decay_fit_node(ex, E, start, 1.0, 1000);
    const std::vector<double> S = bayesian_decay_covariance(F.pt, dim);
    const BayesianGoodnessOfFit G = bayesian_goodness_of_fit(ex["y"].d, F.pt.lam, ex["mask"].d, ex.data_keys.size(), double(dim));
    double zmax = 0; for (auto& h : G.histograms) zmax = std::max(zmax, std::fabs(h.z));
    std::printf("\"%s\": {\"converged\": %s, \"iterations\": %d, \"evidence_finite\": %s, \"covariance_pd\": %s, \"z\": %.4f, \"z_max\": %.4f, \"dpd\": %.4f}",
                name, F.converged ? "true" : "false", F.iterations, std::isfinite(F.evidence) ? "true" : "false",
                S.empty() ? "false" : "true", G.z, zmax, G.deviance_per_dof);
  };
  std::printf("{\"prior_gradient\": %.6e, \"prior_hessian\": %.6e, \"posterior_gradient\": %.6e, ",
              g_err / g_pk, H_err / H_pk, post_err / post_pk);
  fit_report("model", simulate(ex, truth, 3));
  std::printf(", ");
  //: the same expected counts with a ripple no sum of periodic decays produces
  {
    const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, truth));
    const std::size_t n = ex.n_bins;
    std::mt19937_64 rr(3);
    std::vector<double> y(lam.size());
    for (std::size_t j = 0; j < y.size(); ++j) {
      const double m = std::max(lam[j], 1e-12) * (1.0 + 0.2 * std::sin(2.0 * 3.14159265358979323846 * double(j % n) / 16.0));
      y[j] = double(std::poisson_distribution<long long>(m)(rr));
    }
    fit_report("ripple", y);
  }
  std::printf("}\n");
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp])


def test_prior_and_posterior_derivatives(result):
    assert result["prior_gradient"] < 1e-7, result
    assert result["prior_hessian"] < 1e-6, result
    assert result["posterior_gradient"] < 1e-5, result


def test_data_from_the_model_are_fitted(result):
    r = result["model"]
    assert r["converged"] and r["evidence_finite"] and r["covariance_pd"], r
    assert abs(r["z"]) < 3 and r["z_max"] < 3, r


def test_data_no_decay_model_makes_are_rejected(result):
    assert result["ripple"]["z"] > 3, result["ripple"]

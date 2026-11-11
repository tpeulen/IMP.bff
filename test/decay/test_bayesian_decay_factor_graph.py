"""PRD-155 steps 5-7: scored factors, one generic fitting of a factor graph, and the Bayesian decay
model as factors. Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.

**The idea.** A posterior is a product of factors: each dataset's likelihood, each prior, each
external measurement. `InferenceFactorGraph` already holds that structure; with a score on each
factor (its log density, gradient and information over its scope) the graph *is* the log
posterior, and one generic fitting (`inference_fitting`: damped Newton on the summed gradient and
information, then the Laplace evidence) finds the mode of any model built this way. The Bayesian
decay model -- TCSPC histograms (photon counts per arrival-time bin after each laser pulse) turned
into the distance distribution p(R) -- is then a graph with a `counts` likelihood, a `prior`, and
one likelihood per external observation of a weighted sum of p(R) (e.g. an ensemble FRET
efficiency).

What is checked, and what makes each check fail:
* a toy graph with a closed-form posterior (two Gaussian measurements of one value and a Gaussian
  prior): the fit is the closed-form mode, its Laplace evidence the closed-form evidence (exact
  for a Gaussian), the per-kind split sums to the total -- a wrong sign, a missing factor or a
  wrong information fails it;
* a factor without a score is refused, not skipped;
* without observations the decay graph's score at one point equals `bayesian_decay_evaluate`
  there (log posterior, gradient, information, and the split into likelihood / prior), and fitting
  the graph equals `bayesian_decay_fit_node` from the same start;
* external FRET measurements are factors in the docking restraint's vocabulary (`RDAMeanE`,
  `RDAMean`, `Efficiency`, asymmetric errors): each one's model value equals
  `distance_sample_statistics` of p(R) (the docking statistic), its gradient equals central
  differences of its score, and the graph's total is the counts and prior plus these factors;
* the fit is a good fit: Poisson deviance per degree of freedom against its reference (z) and the
  runs test on the residual signs, from the counts factor's own predict() and data -- and every
  observation is met within its errors.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from bayesian_cxx import ROOT, run_driver
import bayesian_decay_synthetic as syn

PRELUDE = r"""
#include "@ROOT@/src/spectroscopy/States.cpp"
#include "@ROOT@/src/fit/InferenceFactorGraph.cpp"
#include "@ROOT@/test/decay/registry_stub.h"
#include "@ROOT@/src/fit/InferenceFitting.cpp"
#include "@ROOT@/src/bayesian/BayesianDecayFactorGraph.cpp"
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
"""

PRELUDE = PRELUDE.replace("@ROOT@", ROOT)

TOY = PRELUDE + r"""
// x ~ N(m0, s0^2) (prior); y1 ~ N(x, s1^2), y2 ~ N(x, s2^2) (two measurements)
static IMP::Pointer<InferenceFactorScore> gauss(double m, double s) {
  return new InferenceFunctionScore([m, s](const std::vector<double>& x, bool d) {
    const double r = (x[0] - m) / s, lp = -0.5 * r * r - std::log(s) - 0.5 * std::log(2.0 * M_PI);
    if (!d) return std::vector<double>{lp};
    return std::vector<double>{lp, -r / s, 1.0 / (s * s)};
  });
}
int main() {
  const double m0 = 1.0, s0 = 2.0, y1 = 3.0, s1 = 0.5, y2 = 2.0, s2 = 1.0;
  InferenceFactorGraph g;
  g.add_variable("x", "x", 0);
  g.add_factor("prior", INFERENCE_FACTOR_PRIOR, {"x"});
  g.add_factor("y1", INFERENCE_FACTOR_LIKELIHOOD, {"x"}, 0, 1);
  g.add_factor("y2", INFERENCE_FACTOR_LIKELIHOOD, {"x"}, 1, 1);
  int refused = 0;
  try { g.eval({0.0}); } catch (const std::domain_error&) { refused = 1; }
  g.set_factor_score("prior", gauss(m0, s0));
  g.set_factor_score("y1", gauss(y1, s1));
  g.set_factor_score("y2", gauss(y2, s2));
  const InferenceFit F = inference_fitting(g, {-5.0});
  // closed form: precision-weighted mean; evidence = N(y1, y2 | marginal), done by the
  // Gaussian identity log Z = log int prod = sum of logs at the mode + 1/2 log(2 pi / P)
  const double P = 1 / (s0 * s0) + 1 / (s1 * s1) + 1 / (s2 * s2);
  const double mode = (m0 / (s0 * s0) + y1 / (s1 * s1) + y2 / (s2 * s2)) / P;
  auto lg = [](double x, double m, double s) { const double r = (x - m) / s; return -0.5 * r * r - std::log(s) - 0.5 * std::log(2.0 * M_PI); };
  const double logZ = lg(mode, m0, s0) + lg(mode, y1, s1) + lg(mode, y2, s2) + 0.5 * std::log(2.0 * M_PI / P);
  std::printf("{\"refused\": %d, \"mode\": %.17g, \"mode_exact\": %.17g, \"evidence\": %.17g, \"evidence_exact\": %.17g,"
              " \"converged\": %d, \"total\": %.17g, \"prior\": %.17g, \"lik\": %.17g, \"hyper\": %.17g, \"n_factors\": %zu}\n",
              refused, F.theta[0], mode, F.laplace_evidence, logZ, int(F.converged), F.at_mode.log_density,
              F.at_mode.log_prior, F.at_mode.log_likelihood, F.at_mode.log_hyper, F.at_mode.factor_keys.size());
}
"""

DECAY = PRELUDE + r"""
static std::vector<double> read_theta(const std::string& dir, std::size_t dim) {
  std::vector<double> t(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(t.data()), std::streamsize(dim * sizeof(double)));
  return t;
}
int main(int, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::vector<double> truth = read_theta(dir, ex.dim);
  const BayesianDecayTensors E(ex);
  const InferenceFactorGraph g = bayesian_decay_factor_graph(dir);
  std::printf("{\"dim\": %zu, \"graph_dim\": %u, \"n_factors\": %u", ex.dim, g.get_parameter_dimension(), g.get_number_of_factors());
  // 1. one point: the graph's score against bayesian_decay_evaluate, and each observation factor
  {
    const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, truth);
    const InferenceScore s = g.eval(truth, true);
    double obs_total = 0.0, fd_err = 0.0, fd_scale = 0.0, stat_err = 0.0;
    int n_obs = 0;
    for (std::size_t k = 0; k < s.factor_keys.size(); ++k) {
      const std::string& key = s.factor_keys[k];
      if (key.rfind("observation:", 0) != 0) continue;
      ++n_obs;
      obs_total += s.factor_log_densities[k];
      InferenceFactorScore* o = g.get_factor_score(key);
      const std::vector<double> x = g.get_factor_values(key, truth);
      const std::vector<double> r = o->eval(x, true);
      for (std::size_t a = 0; a < x.size(); ++a) {
        const double h = 1e-6;
        std::vector<double> xp = x, xm = x;
        xp[a] += h; xm[a] -= h;
        const double fd = (o->eval(xp, false)[0] - o->eval(xm, false)[0]) / (2.0 * h);
        fd_err = std::max(fd_err, std::fabs(fd - r[1 + a]));
        fd_scale = std::max(fd_scale, std::fabs(fd));
      }
      // the model value is the docking statistic of p(R)
      const std::vector<double> p = bayesian_decay_distribution(E, bayesian_decay_unpack(ex, truth));
      std::vector<double> R(ex["rel"].size());
      const double R0 = std::atof(argv[2]);
      for (std::size_t j = 0; j < R.size(); ++j) R[j] = ex["rel"].d[j] * R0;
      const std::vector<double> st = distance_sample_statistics(R, p, R0);
      const double m = o->predict(x)[0];
      stat_err = std::max(stat_err, std::min({std::fabs(m - st[0]), std::fabs(m - st[1]), std::fabs(m - st[2])}));
    }
    double dg = 0.0, dA = 0.0, sg = 0.0, sA = 0.0;
    for (std::size_t i = 0; i < pt.grad.size(); ++i) { dg = std::max(dg, std::fabs(pt.grad[i] - s.gradient[i])); sg = std::max(sg, std::fabs(pt.grad[i])); }
    for (std::size_t i = 0; i < pt.A.size(); ++i) { dA = std::max(dA, std::fabs(pt.A[i] - s.information[i])); sA = std::max(sA, std::fabs(pt.A[i])); }
    std::printf(", \"point\": {\"logpost\": %.17g, \"score\": %.17g, \"loglik\": %.17g, \"logprior\": %.17g, \"obs_total\": %.17g,"
                " \"n_obs\": %d, \"obs_fd_err\": %.17g, \"obs_fd_scale\": %.17g, \"obs_stat_err\": %.17g,"
                " \"s_lik\": %.17g, \"s_prior\": %.17g, \"grad_maxdiff\": %.17g, \"grad_scale\": %.17g, \"A_maxdiff\": %.17g, \"A_scale\": %.17g}",
                pt.logpost, s.log_density, pt.loglik, pt.logprior, obs_total, n_obs, fd_err, fd_scale, stat_err,
                s.log_likelihood, s.log_prior, dg, sg, dA, sA);
  }
  // 2. the same fit two ways
  {
    const BayesianDecayFit F = bayesian_decay_fit_node(ex, E, truth, 1.0, 300, 1e-6);
    const InferenceFit G = inference_fitting(g, truth, 300, 1e-6);
    double dth = 0.0;
    for (std::size_t i = 0; i < F.theta.size(); ++i) dth = std::max(dth, std::fabs(F.theta[i] - G.theta[i]));
    // the fit statistic from the counts factor's own predict() and data
    InferenceFactorScore* counts = g.get_factor_score("counts");
    const std::vector<double> m = counts->predict(g.get_factor_values("counts", G.theta));
    const PoissonGoodnessOfFit Q = poisson_goodness_of_fit(counts->get_data(), m, ex["mask"].d, ex.data_keys.size(), double(ex.dim));
    double zmax = 0.0, runs_min = 1.0;
    for (const auto& h : Q.histograms) { zmax = std::max(zmax, std::fabs(h.z)); runs_min = std::min(runs_min, h.runs_p); }
    std::vector<double> obs_pred, obs_data;
    for (const std::string& k : g.get_factor_keys()) {
      if (k.rfind("observation:", 0) != 0) continue;
      InferenceFactorScore* o = g.get_factor_score(k);
      obs_pred.push_back(o->predict(g.get_factor_values(k, G.theta))[0]);
      obs_data.push_back(o->get_data()[0]);
    }
    std::printf(", \"fit\": {\"node_converged\": %d, \"graph_converged\": %d, \"theta_maxdiff\": %.17g, \"node_logpost\": %.17g,"
                " \"graph_logpost\": %.17g, \"node_evidence\": %.17g, \"graph_evidence\": %.17g, \"node_it\": %d, \"graph_it\": %d,"
                " \"loglik\": %.17g, \"logprior\": %.17g, \"g_lik\": %.17g, \"g_prior\": %.17g,"
                " \"deviance_per_dof\": %.17g, \"dof\": %.17g, \"z_max\": %.17g, \"runs_min\": %.17g",
                int(F.converged), int(G.converged), dth, F.pt.logpost, G.at_mode.log_density, F.evidence, G.laplace_evidence,
                F.iterations, G.iterations, F.pt.loglik, F.pt.logprior, G.at_mode.log_likelihood,
                G.at_mode.log_prior, Q.deviance_per_dof, Q.dof, zmax, runs_min);
    vec("obs_pred", obs_pred); vec("obs_data", obs_data);
    std::printf("}");
  }
  std::printf("}\n");
}
"""

DRAW = PRELUDE + r"""
int main(int, char** argv) {
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(argv[1], ex);
  std::vector<double> t(ex.dim);
  std::ifstream(std::string(argv[1]) + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(t.data()), std::streamsize(ex.dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, t));
  std::mt19937_64 rng(3);
  std::vector<double> y(lam.size());
  for (std::size_t i = 0; i < lam.size(); ++i) y[i] = double(std::poisson_distribution<long long>(std::max(lam[i], 1e-12))(rng));
  std::ofstream(std::string(argv[1]) + "/y.bin", std::ios::binary).write(reinterpret_cast<const char*>(y.data()), std::streamsize(y.size() * sizeof(double)));
  std::printf("{\"n\": %zu}\n", y.size());
}
"""


def experiment(directory, observations=None):
    """The synthetic experiment, its counts drawn from the model at its own parameters."""
    syn.write(directory)
    run_driver(DRAW, args=[directory])
    if observations:
        path = os.path.join(directory, "manifest.json")
        with open(path) as fh:
            m = json.load(fh)
        m["observations"] = observations
        with open(path, "w") as fh:
            json.dump(m, fh)


def test_a_gaussian_toy_graph_is_fitted_exactly():
    r = run_driver(TOY)
    assert r["refused"] == 1                       # a factor without a score is refused
    assert r["converged"] == 1 and r["n_factors"] == 3
    assert abs(r["mode"] - r["mode_exact"]) < 1e-10
    assert abs(r["evidence"] - r["evidence_exact"]) < 1e-10
    assert abs(r["total"] - (r["prior"] + r["lik"] + r["hyper"])) < 1e-12 and r["hyper"] == 0.0


R0 = 52.0
#: an ensemble efficiency, an efficiency-averaged distance and a mean distance, as docking reads them
OBSERVATIONS = [
    dict(name="ext_E", distance_type="Efficiency", distance=0.45, error_neg=0.01, error_pos=0.01, forster_radius=R0),
    dict(name="ext_RE", distance_type="RDAMeanE", distance=54.0, error_neg=1.5, error_pos=2.5, forster_radius=R0),
    dict(name="ext_R", distance_type="RDAMean", distance=57.0, error_neg=3.0, error_pos=3.0, forster_radius=R0),
]


@pytest.fixture(scope="module", params=["plain", "observed"])
def decay(request):
    d = tempfile.mkdtemp()
    experiment(d, None if request.param == "plain" else OBSERVATIONS)
    r = run_driver(DECAY, args=[d, R0], timeout=1800)
    r["case"] = request.param
    return r


def test_the_decay_graph_scores_what_the_model_scores(decay):
    p = decay["point"]
    assert decay["graph_dim"] == decay["dim"]
    observed = decay["case"] == "observed"
    assert decay["n_factors"] == 2 + (len(OBSERVATIONS) if observed else 0) and p["n_obs"] == decay["n_factors"] - 2
    assert abs(p["score"] - (p["logpost"] + p["obs_total"])) <= 1e-12 * abs(p["logpost"])
    assert abs(p["s_lik"] - (p["loglik"] + p["obs_total"])) <= 1e-12 * abs(p["loglik"])
    assert p["s_prior"] == p["logprior"]
    if not observed:
        assert p["grad_maxdiff"] <= 1e-12 * p["grad_scale"]
        assert p["A_maxdiff"] <= 1e-12 * p["A_scale"]
    else:
        assert p["obs_stat_err"] < 1e-12            # the model value is the docking statistic
        assert p["obs_fd_scale"] > 1e-3 and p["obs_fd_err"] <= 1e-5 * p["obs_fd_scale"]


def test_fitting_the_graph_equals_the_decay_fit(decay):
    f = decay["fit"]
    print(json.dumps(dict(case=decay["case"], **f)))
    assert f["node_converged"] == 1 and f["graph_converged"] == 1
    if decay["case"] == "plain":
        assert f["theta_maxdiff"] < 1e-8
        assert abs(f["graph_logpost"] - f["node_logpost"]) <= 1e-10 * abs(f["node_logpost"])
        assert abs(f["graph_evidence"] - f["node_evidence"]) <= 1e-8 * abs(f["node_evidence"])
        assert abs(f["g_lik"] - f["loglik"]) <= 1e-10 * abs(f["loglik"])
        assert abs(f["g_prior"] - f["logprior"]) <= 1e-10 * max(1.0, abs(f["logprior"]))


def test_the_fit_is_good_and_meets_the_observations(decay):
    f = decay["fit"]
    assert f["z_max"] < 3 and f["runs_min"] > 0.001, f
    if decay["case"] == "observed":
        for o, pred in zip(OBSERVATIONS, f["obs_pred"]):
            err = o["error_neg"] if pred < o["distance"] else o["error_pos"]
            assert abs(pred - o["distance"]) < 3 * err, (o, pred)

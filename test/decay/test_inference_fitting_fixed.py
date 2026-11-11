"""PRD-156 step 2: `inference_fitting` holds named variables fixed. Written 2026-11-11 by
claude-code/Opus 5.5 for tpeulen.

**Why.** A hyperparameter -- here the weight lambda of the P-spline roughness prior on the distance
distribution p(R) -- is an ordinary variable of the factor graph. It must not be maximised with the
rest: the prior p(c | lambda) grows as lambda^{rank/2} toward the penalty's null space, so a joint
mode runs to large lambda (PRD-143 A4.6). What a user does instead is hold lambda, fit the rest, and
read the Laplace evidence over the free coordinates, which is then log p(y | lambda); a loop over
lambda (user code, not bff) weights by it.

What is checked, and what would make each check fail:
* a Gaussian toy with two variables, one held: the mode of the free one and the evidence over it
  equal the closed form of the conditional (exact for a Gaussian, 1e-10). Fitting both would move
  the held one and fail; an evidence over both coordinates would fail by 1/2 log(2 pi / P_y);
* an unknown variable and "everything fixed" are refused;
* the Bayesian decay graph with `log10_lam` a free variable but held at log10 lambda = 1 fits the
  same mode as the graph whose lambda is a fixed value of 1: the log posteriors differ by the held
  coordinate's own prior term to 1e-9 nats, and the evidence minus that term equals that graph's
  evidence to 1e-6 nats. The control holds lambda at 0 instead and must differ from the lambda = 1
  evidence by more than 1e-3. The coordinates agree to 1e-5, not tighter: first run (2026-11-11)
  declared 1e-7 and measured 1.2e-6 at tolerance 1e-9 and 4.1e-6 at 1e-13 -- a direction this flat
  is not pinned by the stopping rule, while the log posterior agreed to 4e-11 at both; the
  threshold was changed after that measurement and is recorded here.
* `fixed` empty: the result field `fixed` is empty and the decay fit is the PRD-155 one (that test,
  `test_bayesian_decay_factor_graph.py`, compares it with `bayesian_decay_fit_node`).
"""
import math
import os
import tempfile

import numpy as np
import pytest

from bayesian_cxx import ROOT, run_driver
import bayesian_decay_synthetic as syn

PRELUDE = r"""
#include "@ROOT@/src/spectroscopy/States.cpp"
#include "@ROOT@/src/fit/InferenceFactorGraph.cpp"
#include "@ROOT@/src/fit/InferenceFitting.cpp"
#include "@ROOT@/src/bayesian/BayesianDecayFactorGraph.cpp"
#include <cstdio>
#include <fstream>
using namespace IMP::bff;
""".replace("@ROOT@", ROOT)

TOY = PRELUDE + r"""
static IMP::Pointer<InferenceFactorScore> gauss1(double m, double s) {
  return new InferenceFunctionScore([m, s](const std::vector<double>& x, bool d) {
    const double r = (x[0] - m) / s, lp = -0.5 * r * r - std::log(s) - 0.5 * std::log(2.0 * M_PI);
    if (!d) return std::vector<double>{lp};
    return std::vector<double>{lp, -r / s, 1.0 / (s * s)};
  });
}
// z ~ N(x + y, s): gradient and information over the scope (x, y)
static IMP::Pointer<InferenceFactorScore> sum_obs(double z, double s) {
  return new InferenceFunctionScore([z, s](const std::vector<double>& v, bool d) {
    const double r = (v[0] + v[1] - z) / s, lp = -0.5 * r * r - std::log(s) - 0.5 * std::log(2.0 * M_PI);
    if (!d) return std::vector<double>{lp};
    const double g = -r / s, I = 1.0 / (s * s);
    return std::vector<double>{lp, g, g, I, I, I, I};
  });
}
int main() {
  const double z = 2.5, s = 0.5, y0 = 0.7;
  InferenceFactorGraph g;
  g.add_variable("x", "x", 0);
  g.add_variable("y", "y", 1);
  g.add_factor("px", INFERENCE_FACTOR_PRIOR, {"x"});
  g.add_factor("py", INFERENCE_FACTOR_HYPER, {"y"});
  g.add_factor("z", INFERENCE_FACTOR_LIKELIHOOD, {"x", "y"}, 0, 1);
  g.set_factor_score("px", gauss1(0.0, 1.0));
  g.set_factor_score("py", gauss1(0.0, 1.0));
  g.set_factor_score("z", sum_obs(z, s));
  const InferenceFit F = inference_fitting(g, {-3.0, y0}, 300, 1e-12, 1.0, {"y"});
  const InferenceFit B = inference_fitting(g, {-3.0, y0}, 300, 1e-12);
  // conditional on y = y0: x ~ N(0, 1) * N(z - y0, s^2) in x
  const double P = 1.0 + 1.0 / (s * s);
  const double mode = ((z - y0) / (s * s)) / P;
  auto lg = [](double v, double m, double sd) { const double r = (v - m) / sd; return -0.5 * r * r - std::log(sd) - 0.5 * std::log(2.0 * M_PI); };
  const double logZ = lg(mode, 0.0, 1.0) + lg(y0, 0.0, 1.0) + lg(mode + y0, z, s) + 0.5 * std::log(2.0 * M_PI / P);
  int unknown = 0, all = 0;
  try { inference_fitting(g, {0.0, 0.0}, 10, 1e-6, 1.0, {"nope"}); } catch (const std::invalid_argument&) { unknown = 1; }
  try { inference_fitting(g, {0.0, 0.0}, 10, 1e-6, 1.0, {"x", "y"}); } catch (const std::invalid_argument&) { all = 1; }
  std::printf("{\"x\": %.17g, \"y\": %.17g, \"mode\": %.17g, \"evidence\": %.17g, \"evidence_exact\": %.17g,"
              " \"both_y\": %.17g, \"both_evidence\": %.17g, \"n_fixed\": %zu, \"n_fixed_none\": %zu,"
              " \"converged\": %d, \"unknown_refused\": %d, \"all_refused\": %d}\n",
              F.theta[0], F.theta[1], mode, F.laplace_evidence, logZ, B.theta[1], B.laplace_evidence,
              F.fixed.size(), B.fixed.size(), int(F.converged), unknown, all);
}
"""

DECAY = PRELUDE + r"""
static std::vector<double> read(const std::string& path, std::size_t n) {
  std::vector<double> t(n);
  std::ifstream(path, std::ios::binary).read(reinterpret_cast<char*>(t.data()), std::streamsize(n * sizeof(double)));
  return t;
}
int main(int, char** argv) {
  const std::string held = argv[1], freed = argv[2];
  BayesianDecayExperiment eh, ef;
  bayesian_decay_experiment_load(held, eh);
  bayesian_decay_experiment_load(freed, ef);
  const std::size_t zl = ef.variable("log10_lam")->offset;
  const double lo = ef.variable("log10_lam")->lo, hi = ef.variable("log10_lam")->hi;
  auto z_of = [&](double log10_lam) { const double u = (log10_lam - lo) / (hi - lo); return std::log(u / (1.0 - u)); };
  const std::vector<double> th = read(held + "/theta.bin", eh.dim);
  // the free graph's start: the held graph's start with the lambda coordinate inserted
  auto start_at = [&](double log10_lam) {
    std::vector<double> t(th.begin(), th.begin() + std::ptrdiff_t(zl));
    t.push_back(z_of(log10_lam));
    t.insert(t.end(), th.begin() + std::ptrdiff_t(zl), th.end());
    return t;
  };
  const InferenceFactorGraph gh = bayesian_decay_factor_graph(held), gf = bayesian_decay_factor_graph(freed);
  const InferenceFit H = inference_fitting(gh, th, 300, 1e-9);
  const InferenceFit F1 = inference_fitting(gf, start_at(1.0), 300, 1e-9, 1.0, {"log10_lam"});
  const InferenceFit F0 = inference_fitting(gf, start_at(0.0), 300, 1e-9, 1.0, {"log10_lam"});
  double dth = 0.0;
  for (std::size_t i = 0, j = 0; i < F1.theta.size(); ++i) {
    if (i == zl) continue;
    dth = std::max(dth, std::fabs(F1.theta[i] - H.theta[j++]));
  }
  //: the held coordinate's own prior term (uniform on [lo, hi] through the logit, Jacobian included)
  auto softplus = [](double v) { return v > 30 ? v : std::log1p(std::exp(v)); };
  const double z1 = z_of(1.0), hyper1 = -softplus(-z1) - softplus(z1);
  std::printf("{\"held_converged\": %d, \"free_converged\": %d, \"theta_maxdiff\": %.17g,"
              " \"z_kept\": %.17g, \"z1\": %.17g,"
              " \"logpost_held\": %.17g, \"logpost_free\": %.17g, \"hyper1\": %.17g,"
              " \"evidence_held\": %.17g, \"evidence_free1\": %.17g, \"evidence_free0\": %.17g, \"laplace_ok\": %d}\n",
              int(H.converged), int(F1.converged), dth, F1.theta[zl], z1,
              H.at_mode.log_density, F1.at_mode.log_density, hyper1,
              H.laplace_evidence, F1.laplace_evidence, F0.laplace_evidence, int(F1.laplace_ok && H.laplace_ok));
}
"""


@pytest.fixture(scope="module")
def toy():
    return run_driver(TOY)


@pytest.fixture(scope="module")
def decay():
    held, freed = tempfile.mkdtemp(), tempfile.mkdtemp()
    syn.write(held)
    syn.write(freed, free_lambda=True)
    # the two fixtures must be the same data: lambda's status is the only difference
    assert np.array_equal(np.fromfile(os.path.join(held, "y.bin")), np.fromfile(os.path.join(freed, "y.bin")))
    return run_driver(DECAY, args=[held, freed], timeout=1800)


def test_toy_held_variable_stays_and_the_free_one_is_the_conditional_mode(toy):
    assert toy["converged"] == 1
    assert toy["y"] == 0.7                                   # untouched, bit for bit
    assert math.isclose(toy["x"], toy["mode"], rel_tol=0, abs_tol=1e-10)
    assert toy["n_fixed"] == 1 and toy["n_fixed_none"] == 0


def test_toy_evidence_is_over_the_free_coordinate_only(toy):
    assert math.isclose(toy["evidence"], toy["evidence_exact"], rel_tol=0, abs_tol=1e-10)
    # the control: fitting both moves y and gives a different evidence
    assert abs(toy["both_y"] - 0.7) > 1e-3
    assert abs(toy["both_evidence"] - toy["evidence"]) > 1e-3


def test_unknown_and_all_fixed_are_refused(toy):
    assert toy["unknown_refused"] == 1
    assert toy["all_refused"] == 1


def test_decay_lambda_held_as_a_variable_is_lambda_as_a_fixed_value(decay):
    assert decay["held_converged"] == 1 and decay["free_converged"] == 1 and decay["laplace_ok"] == 1
    assert decay["z_kept"] == decay["z1"]                    # the held coordinate never moved
    assert decay["theta_maxdiff"] < 1e-5, decay["theta_maxdiff"]
    # the log posteriors differ by exactly the held coordinate's own prior term
    assert math.isclose(decay["logpost_free"] - decay["logpost_held"], decay["hyper1"], rel_tol=0, abs_tol=1e-9)


def test_decay_evidence_with_lambda_held_is_log_p_y_given_lambda(decay):
    ev = decay["evidence_free1"] - decay["hyper1"]
    assert math.isclose(ev, decay["evidence_held"], rel_tol=0, abs_tol=1e-6), (ev, decay["evidence_held"])
    # the control: another lambda, another evidence
    assert abs(decay["evidence_free0"] - decay["evidence_free1"]) > 1e-3

"""PRD-156 step 3: a factor graph saved with its data, as msgpack, and loaded evaluable again.
Written 2026-11-11 by claude-code/Opus 5.5 for tpeulen ("improve save load, make date embedding
option (default off, use msgpack for data)").

**What a saved graph is.** `InferenceFactorGraph.save(path)` has always written the structure --
which variables, which factors, which variables each factor reads -- as JSON. The numbers of a
factor (its score: the data and model that give its log density) were not written, so a loaded
graph could not be evaluated. With `embed_data=True` each score is written as a registered type
and a document (msgpack, arrays as `bin`), and loading rebuilds it through the factory registered
for that type (registry capability `factor_score`). The Bayesian decay graph's scores -- the
Poisson likelihood of the photon-count histograms (`counts`), the prior, and one factor per
external FRET measurement -- share one experiment, so it is written once.

What is checked, and what would make each check fail:
* `embed_data=False` (the default) writes exactly `to_json()`'s text, as before;
* the decay graph with three FRET observations, saved with data and loaded **in another process**,
  scores the same log density, gradient and information at a point and fits to the same mode,
  log posterior, evidence and iteration count -- all bit for bit (printed with 17 digits and
  compared as equal strings of numbers). A score rebuilt from a wrong or reordered array fails;
* the experiment is written once, not once per factor: the file is smaller than twice one
  document (five factors share it);
* saving the loaded graph again gives the same bytes;
* refusals, each leaving the target graph as it was: a score that cannot be written (a closure
  without a type) names its factor; truncated bytes; a document whose score type is not
  registered names the type; data for a factor the structure does not have.
"""
import json
import os
import tempfile

import pytest

from bayesian_cxx import ROOT, run_driver
from test_bayesian_decay_factor_graph import DRAW, OBSERVATIONS, experiment  # noqa: F401

PRELUDE = r"""
#include "@ROOT@/src/spectroscopy/States.cpp"
#include "@ROOT@/src/fit/InferenceFactorGraph.cpp"
#include "@ROOT@/src/fit/InferenceFitting.cpp"
#include "@ROOT@/src/bayesian/BayesianDecayFactorGraph.cpp"
#include "@ROOT@/test/decay/registry_stub.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace IMP::bff;
static void vec(const char* name, const std::vector<double>& v) {
  std::printf(", \"%s\": [", name);
  for (std::size_t i = 0; i < v.size(); ++i) std::printf("%s%.17g", i ? "," : "", v[i]);
  std::printf("]");
}
static std::string slurp(const std::string& p) {
  std::ifstream in(p, std::ios::binary); std::ostringstream o; o << in.rdbuf(); return o.str();
}
static std::vector<double> read_theta(const std::string& dir, std::size_t dim) {
  std::vector<double> t(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(t.data()), std::streamsize(dim * sizeof(double)));
  return t;
}
//: what the graph scores at theta and where it fits to, every number printed exactly
static void report(const InferenceFactorGraph& g, const std::vector<double>& th) {
  const InferenceScore s = g.eval(th, true);
  std::printf("{\"log_density\": %.17g, \"log_likelihood\": %.17g, \"log_prior\": %.17g", s.log_density, s.log_likelihood, s.log_prior);
  vec("factor_log_densities", s.factor_log_densities);
  vec("gradient", s.gradient);
  vec("information", s.information);
  const InferenceFit F = inference_fitting(g, th, 300, 1e-6);
  std::printf(", \"fit_iterations\": %d, \"fit_logpost\": %.17g, \"fit_evidence\": %.17g", F.iterations, F.at_mode.log_density, F.laplace_evidence);
  vec("fit_theta", F.theta);
  std::printf("}");
}
""".replace("@ROOT@", ROOT)

SAVE = PRELUDE + r"""
int main(int, char** argv) {
  const std::string dir = argv[1], out = argv[2];
  const InferenceFactorGraph g = bayesian_decay_factor_graph(dir);
  g.save(out + "/plain.json");
  g.save(out + "/data.bff", true);
  const std::string doc = g.get_factor_score("counts")->get_score_document();
  std::printf("{\"plain_is_to_json\": %d, \"file_bytes\": %zu, \"document_bytes\": %zu, \"n_factors\": %u, \"report\": ",
              int(slurp(out + "/plain.json") == g.to_json()), slurp(out + "/data.bff").size(), doc.size(), g.get_number_of_factors());
  report(g, read_theta(dir, g.get_parameter_dimension()));
  std::printf("}\n");
}
"""

LOAD = PRELUDE + r"""
int main(int, char** argv) {
  const std::string dir = argv[1], out = argv[2];
  InferenceFactorGraph g;
  g.load(out + "/data.bff");
  g.save(out + "/again.bff", true);
  std::printf("{\"evaluable\": %d, \"resaved_identical\": %d, \"n_factors\": %u, \"report\": ",
              int(g.get_is_evaluable()), int(slurp(out + "/again.bff") == slurp(out + "/data.bff")), g.get_number_of_factors());
  report(g, read_theta(dir, g.get_parameter_dimension()));
  std::printf("}\n");
}
"""

REFUSE = PRELUDE + r"""
static int refused(const std::function<void()>& f, const char* must_name) {
  try { f(); } catch (const std::exception& e) { return std::string(e.what()).find(must_name) != std::string::npos ? 1 : 2; }
  return 0;
}
int main(int, char** argv) {
  const std::string out = argv[1];
  // a closure score without a type
  InferenceFactorGraph t;
  t.add_variable("x", "x", 0);
  t.add_factor("anonymous_factor", INFERENCE_FACTOR_PRIOR, {"x"});
  t.set_factor_score("anonymous_factor", new InferenceFunctionScore([](const std::vector<double>&, bool) { return std::vector<double>{0.0}; }));
  const int no_type = refused([&] { t.to_msgpack(true); }, "anonymous_factor");
  const int no_type_file = refused([&] { t.save(out + "/never.bff", true); }, "anonymous_factor");
  const int no_file_written = int(!std::ifstream(out + "/never.bff").good());
  // the same graph without data is fine
  const int structure_only = refused([&] { t.save(out + "/structure.bff", false); }, "") == 0 ? 1 : 0;
  // a good file, then broken copies of it
  const std::string good = slurp(out + "/data.bff");
  InferenceFactorGraph target;
  target.load(out + "/data.bff");
  const unsigned n_before = target.get_number_of_factors();
  const int truncated = refused([&] { target.from_msgpack(good.substr(0, good.size() / 2)); }, "from_msgpack");
  namespace mpk = internal::mpk;
  mpk::Value doc = mpk::parse(good);
  mpk::Value unknown = doc, missing = doc;
  for (auto& kv : unknown.m) if (kv.first == "data") for (auto& e : kv.second.a) for (auto& f : e.m) if (f.first == "type") f.second.s = "no_such_score_type";
  for (auto& kv : missing.m) if (kv.first == "data") for (auto& f : kv.second.a[0].m) if (f.first == "factors") { mpk::Value k; k.kind = mpk::Value::Str; k.s = "not_in_structure"; f.second.a.push_back(k); }
  std::string ub, mb;
  mpk::put_value(ub, unknown);
  mpk::put_value(mb, missing);
  const int unknown_type = refused([&] { target.from_msgpack(ub); }, "no_such_score_type");
  const int missing_factor = refused([&] { target.from_msgpack(mb); }, "not_in_structure");
  std::printf("{\"no_type\": %d, \"no_type_file\": %d, \"no_file_written\": %d, \"structure_only\": %d, \"truncated\": %d,"
              " \"unknown_type\": %d, \"missing_factor\": %d, \"unchanged\": %d, \"still_evaluable\": %d}\n",
              no_type, no_type_file, no_file_written, structure_only, truncated, unknown_type, missing_factor,
              int(target.get_number_of_factors() == n_before), int(target.get_is_evaluable()));
}
"""


@pytest.fixture(scope="module")
def saved():
    d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    experiment(d, OBSERVATIONS)
    s = run_driver(SAVE, args=[d, out], timeout=1800)
    loaded = run_driver(LOAD, args=[d, out], timeout=1800)   # another process
    return dict(dir=d, out=out, saved=s, loaded=loaded)


def test_without_data_the_file_is_the_json_of_before(saved):
    assert saved["saved"]["plain_is_to_json"] == 1


def test_loaded_in_another_process_the_graph_scores_and_fits_bit_for_bit(saved):
    s, l = saved["saved"]["report"], saved["loaded"]["report"]
    assert saved["loaded"]["evaluable"] == 1
    assert saved["loaded"]["n_factors"] == saved["saved"]["n_factors"] == 5   # counts, prior, three observations
    assert s.keys() == l.keys()
    for k in s:
        assert s[k] == l[k], k                    # exact: the same doubles, parsed from the same 17 digits


def test_the_experiment_is_written_once(saved):
    assert saved["saved"]["file_bytes"] < 2 * saved["saved"]["document_bytes"]


def test_saving_the_loaded_graph_again_gives_the_same_bytes(saved):
    assert saved["loaded"]["resaved_identical"] == 1


def test_refusals_name_the_cause_and_leave_the_graph_as_it_was(saved):
    r = run_driver(REFUSE, args=[saved["out"]], timeout=1800)
    assert r["no_type"] == 1 and r["no_type_file"] == 1     # 1: refused, naming the factor
    assert r["no_file_written"] == 1
    assert r["structure_only"] == 1
    assert r["truncated"] == 1
    assert r["unknown_type"] == 1
    assert r["missing_factor"] == 1
    assert r["unchanged"] == 1 and r["still_evaluable"] == 1

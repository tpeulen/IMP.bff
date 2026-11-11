/**\file BayesianDecayFactorGraph.cpp
 * The Bayesian decay model as an evaluable factor graph (PRD-155). Written
 * 2026-11-10 by claude-code/Opus 5.5 for tpeulen.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/bayesian/BayesianDecayFactorGraph.h>
#include <IMP/bff/spectroscopy/States.h>

#if __has_include("pocketfft/pocketfft_hdronly.h")
#include <IMP/bff/bayesian/BayesianDecayPosterior.h>
#include <IMP/bff/internal/Msgpack.h>
#define IMPBFF_HAS_BAYESIAN_DECAY 1
#endif

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

#ifdef IMPBFF_HAS_BAYESIAN_DECAY
namespace {

//: the registered type of every score of the decay graph: one document, the whole experiment
const char* const kScoreType = "bayesian_decay_experiment";
const char* const kDocumentFormat = "bff.bayesian_decay_experiment";

struct DecayContext {
  BayesianDecayExperiment ex;
  std::unique_ptr<BayesianDecayTensors> E;
  nlohmann::json manifest;
  //: the experiment as msgpack, built once on the first save and shared by every score
  std::once_flag document_once;
  MsgpackBytes document;
  const MsgpackBytes& get_document() {
    std::call_once(document_once, [this] {
      namespace mpk = internal::mpk;
      std::string o;
      mpk::put_map(o, 4);
      mpk::put_str(o, "format"); mpk::put_str(o, kDocumentFormat);
      mpk::put_str(o, "version"); mpk::put_int(o, 1);
      mpk::put_str(o, "manifest"); mpk::put_str(o, manifest.dump());
      //: the arrays the manifest lists, as read (derived arrays -- a basis built from its degree,
      //: transfer tensors -- are rebuilt by the loader from the manifest)
      const nlohmann::json& listed = manifest["arrays"];
      mpk::put_str(o, "arrays");
      mpk::put_map(o, listed.size());
      for (auto it = listed.begin(); it != listed.end(); ++it) {
        mpk::put_str(o, it.key());
        mpk::put_value(o, mpk::bin_of_doubles(ex[it.key()].d));
      }
      document = o;
    });
    return document;
  }
};

void make_writable(InferenceFunctionScore* s, std::shared_ptr<DecayContext> C) {
  s->set_score_document(kScoreType, [C]() { return C->get_document(); });
}

//! One external FRET measurement of the pair, in the docking restraint's vocabulary (fps.json:
//! `distance_type`, `distance`, `error_neg`, `error_pos`, `forster_radius`).
struct FretMeasurement {
  std::string name;
  int type = -1;          //: PROBE_PAIR_DISTANCE_MEAN, PROBE_PAIR_DISTANCE_E or PROBE_PAIR_EFFICIENCY
  double value = 0.0, error_neg = 0.0, error_pos = 0.0, forster_radius = 0.0;
};

std::vector<FretMeasurement> read_measurements(const nlohmann::json& m) {
  std::vector<FretMeasurement> out;
  if (!m.count("observations")) return out;
  const nlohmann::json& list = m["observations"];
  if (!list.is_array()) throw std::runtime_error("bayesian_decay_factor_graph: observations must be a list");
  for (const auto& o : list) {
    FretMeasurement f;
    f.name = o.at("name").get<std::string>();
    const std::string tn = o.at("distance_type").get<std::string>();
    f.type = probe_pair_distance_type(tn);
    if (f.type != PROBE_PAIR_DISTANCE_MEAN && f.type != PROBE_PAIR_DISTANCE_E && f.type != PROBE_PAIR_EFFICIENCY)
      throw std::runtime_error("bayesian_decay_factor_graph: observation " + f.name + ": distance_type " + tn +
                               " is not a function of p(R) (RDAMean, RDAMeanE or Efficiency)");
    f.value = o.at("distance").get<double>();
    f.error_neg = o.at("error_neg").get<double>();
    f.error_pos = o.at("error_pos").get<double>();
    f.forster_radius = o.at("forster_radius").get<double>();
    //: chi2_score scores a non-positive error as nothing; a measurement that silently counts nothing is refused
    if (!std::isfinite(f.value) || !(f.error_neg > 0.0) || !(f.error_pos > 0.0) || !(f.forster_radius > 0.0))
      throw std::runtime_error("bayesian_decay_factor_graph: observation " + f.name +
                               ": distance finite, error_neg, error_pos and forster_radius positive");
    for (const auto& prev : out)
      if (prev.name == f.name) throw std::runtime_error("bayesian_decay_factor_graph: observation " + f.name + " given twice");
    out.push_back(f);
  }
  return out;
}

std::vector<double> pack(double lp, const std::vector<double>& g, const std::vector<double>& A) {
  std::vector<double> r;
  r.reserve(1 + g.size() + A.size());
  r.push_back(lp);
  r.insert(r.end(), g.begin(), g.end());
  r.insert(r.end(), A.begin(), A.end());
  return r;
}

}  // namespace
#endif

#ifdef IMPBFF_HAS_BAYESIAN_DECAY
namespace {

//! The scored graph of a loaded experiment (`C->ex`, `C->manifest` set).
InferenceFactorGraph decay_graph(std::shared_ptr<DecayContext> C) {
  const nlohmann::json& manifest = C->manifest;
  const std::vector<FretMeasurement> observations = read_measurements(manifest);
  C->E.reset(new BayesianDecayTensors(C->ex));
  const BayesianDecayExperiment& ex = C->ex;
  const std::size_t dim = ex.dim;

  InferenceFactorGraph g;
  std::vector<const BayesianDecayVariable*> vars;
  for (const BayesianDecayVariable& v : ex.variables) vars.push_back(&v);
  std::sort(vars.begin(), vars.end(),
            [](const BayesianDecayVariable* a, const BayesianDecayVariable* b) { return a->offset < b->offset; });
  //: the counts and the prior read every coordinate: their scope, in theta order, must tile theta
  std::size_t next = 0;
  std::vector<std::string> all;
  for (const BayesianDecayVariable* v : vars) {
    if (v->offset != next)
      throw std::runtime_error("bayesian_decay_factor_graph: variables do not tile theta at " + v->name);
    g.add_variable(v->name, v->name, int(v->offset), -1, int(v->size), v->family);
    all.push_back(v->name);
    next += v->size;
  }
  if (next != dim) throw std::runtime_error("bayesian_decay_factor_graph: variables do not cover theta");

  const std::vector<double> y = ex["y"].d;
  std::size_t n_masked = 0;
  for (double w : ex["mask"].d) n_masked += w > 0.0 ? 1 : 0;

  IMP::Pointer<InferenceFactorScore> counts(new InferenceFunctionScore(
      [C](const std::vector<double>& th, bool derivs) {
        const BayesianDecayExperiment& f = C->ex;
        const BayesianDecayTensors& e = *C->E;
        const BayesianDecayValues v = bayesian_decay_unpack(f, th);
        if (!derivs) return std::vector<double>{bayesian_decay_log_likelihood(f, bayesian_decay_expected_counts(f, e, v))};
        //: bayesian_decay_evaluate's likelihood part, unchanged
        const std::size_t d = f.dim;
        const std::vector<double> a2 = bayesian_decay_amplitudes(f, e, v);
        const std::vector<double> Ja = bayesian_decay_amplitude_jacobian(f, e, v);
        std::vector<std::vector<std::size_t>> cols;
        std::vector<double> J;
        const std::vector<double> lam = bayesian_decay_expected_counts(f, e, v, &J, &Ja, &a2, &cols);
        const BayesianDecayArray& yy = f["y"], & mask = f["mask"];
        std::vector<double> grad(d, 0.0), A(d * d, 0.0);
        ::tttrlib::poisson_score_blocks(yy.d.data(), lam.data(), J.data(), mask.d.data(), lam.size(), d,
                                        ::tttrlib::POISSON_INFORMATION_EXPECTED, cols, f["y"].shape[1], grad.data(), A.data(),
                                        [](std::size_t n, const std::function<void(std::size_t)>& body) { internal::bayesian_parallel_for(n, body); });
        return pack(bayesian_decay_log_likelihood(f, lam), grad, A);
      },
      [C](const std::vector<double>& th) {
        return bayesian_decay_expected_counts(C->ex, *C->E, bayesian_decay_unpack(C->ex, th));
      },
      y, "bayesian_decay_counts"));
  make_writable(static_cast<InferenceFunctionScore*>(counts.get()), C);
  g.add_factor("counts", INFERENCE_FACTOR_LIKELIHOOD, all, -1, int(n_masked));
  g.set_factor_score("counts", counts);

  IMP::Pointer<InferenceFactorScore> prior(new InferenceFunctionScore(
      [C](const std::vector<double>& th, bool derivs) {
        const BayesianDecayPrior P = bayesian_decay_log_prior(C->ex, th, derivs);
        if (!derivs) return std::vector<double>{P.lp};
        std::vector<double> A(P.H.size());
        for (std::size_t k = 0; k < A.size(); ++k) A[k] = -P.H[k];   //: information = minus the Hessian
        return pack(P.lp, P.g, A);
      },
      InferenceFunctionScore::Predict(), std::vector<double>(), "bayesian_decay_prior"));
  make_writable(static_cast<InferenceFunctionScore*>(prior.get()), C);
  g.add_factor("prior", INFERENCE_FACTOR_PRIOR, all, -1, 0);
  g.set_factor_score("prior", prior);

  if (!observations.empty()) {
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(ex, "c");
    if (!vc.present) throw std::runtime_error("bayesian_decay_factor_graph: observations need the distribution variable c");
    const std::vector<double> rel = ex["rel"].d;
    for (const FretMeasurement& o : observations) {
      //: the model's statistic is the docking one, distance_sample_statistics of p(R) at R = rel R0,
      //: scored by chi2_score; its derivative goes through the weighted sum (<R> or <E>) of p(R)
      std::vector<double> R(rel.size()), w(rel.size());
      for (std::size_t j = 0; j < rel.size(); ++j) {
        R[j] = rel[j] * o.forster_radius;
        w[j] = o.type == PROBE_PAIR_DISTANCE_MEAN ? R[j] : fret_efficiency(R[j], o.forster_radius);
      }
      //: the weighted sum reads only c; the other coordinates are held at zero, which it ignores
      auto theta_of = [C, vc](const std::vector<double>& xc) {
        std::vector<double> th(C->ex.dim, 0.0);
        std::copy(xc.begin(), xc.end(), th.begin() + std::ptrdiff_t(vc.off));
        return th;
      };
      auto model_value = [C, o, R, theta_of](const std::vector<double>& xc) {
        const std::vector<double> p = bayesian_decay_distribution(*C->E, bayesian_decay_unpack(C->ex, theta_of(xc)));
        const std::vector<double> st = distance_sample_statistics(R, p, o.forster_radius);
        return o.type == PROBE_PAIR_DISTANCE_MEAN ? st[0] : o.type == PROBE_PAIR_DISTANCE_E ? st[1] : st[2];
      };
      IMP::Pointer<InferenceFactorScore> obs(new InferenceFunctionScore(
          [C, o, w, theta_of, model_value](const std::vector<double>& xc, bool derivs) {
            const double m = model_value(xc);
            //: log of the split normal with the two error bars, i.e. -chi2/2 and its normalisation
            const double lp = -0.5 * chi2_score(m, o.value, o.error_neg, o.error_pos)
                              - std::log(0.5 * (o.error_neg + o.error_pos)) - 0.5 * std::log(2.0 * 3.14159265358979323846);
            if (!derivs) return std::vector<double>{lp};
            const BayesianDecayWeightedSum S = bayesian_decay_weighted_sum(C->ex, *C->E, theta_of(xc), w, false);
            //: d m / d (weighted sum): 1 for <R> and <E>; for R_E = R0 (1/E - 1)^(1/6), -R0 / (6 E^2) (1/E - 1)^(-5/6)
            double dm = 1.0;
            if (o.type == PROBE_PAIR_DISTANCE_E) {
              const double E = S.f;
              dm = -o.forster_radius / (6.0 * E * E) * std::pow(1.0 / E - 1.0, -5.0 / 6.0);
            }
            const double sigma = m < o.value ? o.error_neg : o.error_pos;
            const double r = (m - o.value) / sigma;
            const std::size_t n = S.dz.size();
            std::vector<double> grad(n), A(n * n);
            for (std::size_t a = 0; a < n; ++a) {
              const double ga = dm * S.dz[a];
              grad[a] = -(r / sigma) * ga;
              for (std::size_t b = 0; b < n; ++b) A[a * n + b] = ga * dm * S.dz[b] / (sigma * sigma);
            }
            return pack(lp, grad, A);
          },
          [model_value](const std::vector<double>& xc) { return std::vector<double>{model_value(xc)}; },
          std::vector<double>{o.value}, "bayesian_decay_fret_observation"));
      make_writable(static_cast<InferenceFunctionScore*>(obs.get()), C);
      const std::string key = "observation:" + o.name;
      g.add_factor(key, INFERENCE_FACTOR_LIKELIHOOD, std::vector<std::string>{"c"}, -1, 1);
      g.set_factor_score(key, obs);
    }
  }
  return g;
}

const char* const kScoreEntry = R"JSON({
  "label": "Bayesian decay experiment",
  "summary": "The scores of a Bayesian decay factor graph (counts, prior, external FRET observations), rebuilt from the experiment they share.",
  "description": "Every score of bayesian_decay_factor_graph is written as this type with one document: the experiment's manifest and its float64 arrays as msgpack bin. Loading rebuilds the experiment once and returns the scores of the factors named. Written for PRD-156 (graph save/load with embedded data)."
})JSON";

std::map<std::string, IMP::Pointer<InferenceFactorScore> > rebuild_scores(const MsgpackBytes& document,
                                                                          const std::vector<std::string>& keys) {
  namespace mpk = internal::mpk;
  const mpk::Value doc = mpk::parse(document);
  if (doc.kind != mpk::Value::Map || doc.at("format").str() != kDocumentFormat || doc.at("version").integer() != 1)
    throw std::runtime_error("not a " + std::string(kDocumentFormat) + " version 1 document");
  std::shared_ptr<DecayContext> C(new DecayContext());
  C->manifest = nlohmann::json::parse(doc.at("manifest").str());
  const mpk::Value& arrays = doc.at("arrays");
  bayesian_decay_experiment_from_manifest(
      C->manifest,
      [&arrays](const std::string& name, const std::string&, std::vector<double>& out) {
        const std::vector<double> d = mpk::doubles_of_bin(arrays.at(name));
        if (d.size() != out.size())
          throw std::runtime_error("array " + name + " has " + std::to_string(d.size()) + " values, its shape " +
                                   std::to_string(out.size()));
        out = d;
      },
      C->ex);
  //: the document is the one read, byte for byte, so a graph saved again writes the same bytes
  std::call_once(C->document_once, [&C, &document] { C->document = document; });
  const InferenceFactorGraph g = decay_graph(C);
  std::map<std::string, IMP::Pointer<InferenceFactorScore> > out;
  for (const std::string& k : keys) {
    InferenceFactorScore* s = g.get_factor_score(k);
    if (s == nullptr) throw std::runtime_error("the experiment has no factor '" + k + "'");
    out[k] = s;
  }
  return out;
}

const bool registered_decay_scores = register_factor_score(kScoreType, kScoreEntry, rebuild_scores);

}  // namespace
#endif

InferenceFactorGraph bayesian_decay_factor_graph(const std::string& directory) {
#ifdef IMPBFF_HAS_BAYESIAN_DECAY
  std::shared_ptr<DecayContext> C(new DecayContext());
  C->manifest = bayesian_decay_experiment_load(directory, C->ex);
  return decay_graph(C);
#else
  (void)directory;
  throw std::domain_error("bayesian_decay_factor_graph: this build of IMP.bff has no Bayesian decay code (pocketfft missing)");
#endif
}

std::vector<double> bayesian_decay_distribution_at(const std::string& directory,
                                                   const std::vector<double>& theta) {
#ifdef IMPBFF_HAS_BAYESIAN_DECAY
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(directory, ex);
  if (theta.size() != ex.dim) throw std::invalid_argument("bayesian_decay_distribution_at: theta has the wrong length");
  const BayesianDecayTensors E(ex);
  return bayesian_decay_distribution(E, bayesian_decay_unpack(ex, theta));
#else
  (void)directory; (void)theta;
  throw std::domain_error("bayesian_decay_distribution_at: no Bayesian decay code in this build");
#endif
}

std::vector<double> bayesian_decay_distance_grid(const std::string& directory) {
#ifdef IMPBFF_HAS_BAYESIAN_DECAY
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(directory, ex);
  return ex["rel"].d;
#else
  (void)directory;
  throw std::domain_error("bayesian_decay_distance_grid: no Bayesian decay code in this build");
#endif
}

IMPBFF_END_NAMESPACE

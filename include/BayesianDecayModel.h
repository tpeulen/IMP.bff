/**
 * \file IMP/bff/BayesianDecayModel.h
 * \brief A polarised, multi-sample TCSPC experiment as data, and the model of
 *        its counts: amplitudes per physics channel, expected counts, and both
 *        Jacobians, analytic.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * Ported 2026-09-14 from ucfret's `investigation/pinn_pR_anisotropy/s89_cpp`
 * (PRD-142 step 3), where every quantity here was gated against the Python
 * prototype (`s88_laplace_posterior`, torch) on the CBM56 measurement:
 * amplitudes 9e-16, the amplitude Jacobian 6e-16, expected counts 2e-15, the
 * count Jacobian 4e-15, each relative to its peak.
 *
 * **The experiment is data.** What distinguishes one measurement from another --
 * which samples, which pulses and detectors, which species a sample holds and how
 * it is excited (a *scope*), which measured response each histogram was
 * convolved with, the free coordinates and their priors -- is read once from a
 * manifest into the tables below (`bayesian_decay_experiment_load`). The code holds
 * the physics:
 *
 * * a donor block `(x base + sum_j u_j S_R[j]) @ spectrum` times the donor's
 *   quantum yield and excitation, `u = (1 - x) p(R/R0)` the FRET population;
 * * an acceptor block, sensitised (`S_Ag`, with its rotational part `S_Ag_rot_r`)
 *   plus directly excited (`A_dir`), weighted by the acceptor's lifetimes `w_a`;
 * * polarised mixing per detector, `(2 - 3 l1)` for VV and `(-1 + 3 l2)` for VH,
 *   the rotational operator `S_rho` weighted by the scope's rotational
 *   distribution and fundamental anisotropy, the g factor dividing VH;
 * * the instrument: a scale per sample, scatter into the response column and a
 *   flat background, then the measured response's periodic basis
 *   (`BayesianMeasuredResponse.h`) and a soft floor on the counts.
 *
 * The transfer tensors (`E_base`, `E_S_R`, ...) are computed elsewhere and read as
 * arrays; building them is not this header's concern.
 */

#ifndef IMPBFF_BAYESIANDECAYMODEL_H
#define IMPBFF_BAYESIANDECAYMODEL_H

//! The FFT is pocketfft, reached through tttrlib (`"pocketfft/pocketfft_hdronly.h"`
//! with tttrlib's `thirdparty/` on the include path, as tttrlib's own sources
//! include it). Where it is not on the path -- the IMP module build today -- this
//! header declares nothing, so that `IMP/bff.h`, which includes every public header,
//! still compiles.
#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/BayesianMeasuredResponse.h>
#include <IMP/bff/BayesianTransferTensors.h>
#include <IMP/bff/BayesianPSpline.h>
#include <IMP/bff/BayesianTransforms.h>
#include <IMP/bff/BayesianFisherScoring.h>
#include <IMP/bff/PhotophysicsPolarisation.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/BayesianParallel.h>
#include <IMP/bff/internal/TCSPCInstrument.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

IMPBFF_BEGIN_NAMESPACE


//! \name A TCSPC experiment as data
//! @{

//! A named float64 array read from the manifest (row-major).
struct BayesianDecayArray {
  std::vector<std::size_t> shape;
  std::vector<double> d;
  std::size_t size() const { std::size_t n = 1; for (auto s : shape) n *= s; return n; }
};

//! A free coordinate block of theta: its transform and its prior.
struct BayesianDecayVariable {
  std::string name, transform, family;   //!< transform: identity/log/logit/alr/sum_to_zero; family: the prior's
  std::size_t offset = 0, size = 0, constrained_size = 0;
  double lo = 0.0, hi = 1.0;             //!< logit bounds
  std::vector<double> a, b;              //!< the prior's two parameters per element
};

/**
 * \brief What a sample holds and how a pulse excites it.
 *
 * `spectrum`: "" (no donor), "donor" or "reference" (a reference dye's own
 * spectrum). `don_ex`: the donor's excitation, "1", "EX_DR" or "EX_AG".
 * `fret`: the FRET population `u` exists; `sensitised`: acceptor emission by
 * transfer; `has_acc`: an acceptor block exists; `direct`: its direct excitation
 * as a product of "EX_AG", "n_da", "per_mol" ("" none); `rho`, `r0`: the names of
 * the rotational distribution and fundamental anisotropy used.
 */
struct BayesianDecayScope {
  std::string spectrum, don_ex, direct, rho, r0;
  bool fret = false, sensitised = false, has_acc = false;
};

//! One physics channel: a (sample, pulse, detector) contribution to one histogram.
struct BayesianDecayPart {
  std::size_t out = 0;          //!< the data histogram it adds to
  std::size_t amp_index = 0;    //!< its position in the amplitude vector (`keys` order)
  std::size_t response = 0, scope = 0;
  int colour = 0, pol = 0;      //!< 0 green / 1 red; 0 VV / 1 VH
  std::string channel, det, phys, sample;   //!< det may be a pulse's view of the physical detector `phys`
  double offset_ns = 0.0;       //!< the pulse delay of that view
};

//! A measured response and the names of the coordinates that prepare it.
struct BayesianDecayResponse {
  std::string sample, det, bg_var, shift_var, offset_var;   //!< offset_var "" when none
};

//! The P-spline prior on log p(R/R0) (Eilers & Marx 1996).
struct BayesianDecayPSpline {
  std::size_t n = 0;
  int order = 2;
  double tilt_sd = 3.0, quad_sd = 30.0, rank = 0.0;
  std::string family = "gaussian", space = "log", link = "softmax";
};

/**
 * \brief The experiment: axis, tables, arrays, and the periodic kernel on its grid.
 *
 * `arrays` holds every named array of the manifest -- the transfer tensors, the
 * spline, the data (`y`, `mask`), the measured responses (`response_<i>`) and
 * whatever else the manifest carries. `kernel()` is built at load and shared.
 */
struct BayesianDecayExperiment {
  std::size_t n_bins = 0, dim = 0;
  double dt = 0.0, period = 0.0, soft = 0.05, ref_spec_sd = 0.05, spec_smooth_logdet = 0.0;
  std::vector<std::string> keys, data_keys;
  std::vector<BayesianDecayScope> scopes;
  std::vector<BayesianDecayPart> parts;
  std::vector<BayesianDecayResponse> responses;
  std::vector<BayesianDecayVariable> variables;
  std::map<std::string, std::vector<double>> fixed_values;
  BayesianDecayPSpline pspline;
  std::map<std::string, BayesianDecayArray> arrays;
  std::shared_ptr<const BayesianPeriodicKernel> kernel_ptr;
  //! Coates pile-up on every histogram, computed from its own counts (manifest block
  //! `pile_up`); a DNL table per histogram is the optional array `linearization`
  //! (n_data x n_bins). Both in internal/TCSPCInstrument.h's order.
  bool pile_up = false;
  double repetition_rate_mhz = 0.0, dead_time_ns = 0.0, measurement_time_s = 0.0;

  const BayesianDecayArray& operator[](const std::string& name) const {
    auto it = arrays.find(name);
    if (it == arrays.end()) throw std::out_of_range("BayesianDecayExperiment: no array " + name);
    return it->second;
  }
  const BayesianDecayVariable* variable(const std::string& name) const {
    for (const auto& v : variables) if (v.name == name) return &v;
    return nullptr;
  }
  BayesianDecayAxis axis() const { return BayesianDecayAxis{n_bins, dt, std::size_t(std::llround(period / dt))}; }
  const BayesianPeriodicKernel& kernel() const { return *kernel_ptr; }
};

/**
 * \brief Build the transfer tensors an experiment's model reads, from its manifest's
 *        `transfer` block, instead of reading them as arrays (PRD-143 #18e).
 *
 * The block: `response_sigma` (ns), optional `response_position` (ns; default 0.10
 * of the window), `log10_tau_lo`, optional `log10_tau_hi` (default three windows),
 * `per_decade`, the (R0, tau_ref) pair `R0` and `tau_ref`, and the grids `rel`
 * (R/R0), `rho` (ns), `tau_a` (ns), `rho_a` (ns), and optionally `acceptor_maps`
 * (`faithful`, the default: the prototype's construction; `exact`: periodic, PRD-143
 * A4.3). Written: `tau_c`, `E_base`
 * (`K x Kint`, the interior identity), `E_S_R`, `E_S_rho`, `E_S_Ag`, `E_S_Ag_rot_r`,
 * `E_A_dir`, `E_A_dir_rot_r` -- in the shapes ucfret's emitter writes them. An
 * experiment that carries both the block and any of those arrays is refused: two
 * sources for one tensor is how a fixture ends up gating the wrong one.
 */
inline void bayesian_decay_build_transfer_tensors(const nlohmann::json& t, BayesianDecayExperiment& ex) {
  for (const char* name : {"tau_c", "E_base", "E_S_R", "E_S_rho", "E_S_Ag", "E_S_Ag_rot_r", "E_A_dir", "E_A_dir_rot_r"})
    if (ex.arrays.count(name)) throw std::runtime_error(std::string("transfer block and array ") + name + " both given");
  BayesianTransferBasisSpec spec;
  spec.axis = ex.axis();
  spec.response_sigma = t["response_sigma"].get<double>();
  if (t.count("response_position")) spec.response_position = t["response_position"].get<double>();
  spec.log10_tau_lo = t["log10_tau_lo"].get<double>();
  if (t.count("log10_tau_hi")) spec.log10_tau_hi = t["log10_tau_hi"].get<double>();
  spec.per_decade = t["per_decade"].get<int>();
  const double R0 = t["R0"].get<double>(), tau_ref = t["tau_ref"].get<double>();
  const auto rel = t["rel"].get<std::vector<double>>(), rho = t["rho"].get<std::vector<double>>();
  const auto tau_a = t["tau_a"].get<std::vector<double>>(), rho_a = t["rho_a"].get<std::vector<double>>();
  const BayesianTransferBasis tb = bayesian_transfer_basis(spec);
  const ::tttrlib::RidgeProjector P = bayesian_transfer_projector(tb.basis);
  const std::size_t K = tb.basis.K, nt = tb.kernel->tau().size();
  std::vector<double> k_fret(rel.size()), k_rot(rho.size());
  for (std::size_t j = 0; j < rel.size(); ++j) k_fret[j] = std::pow(1.0 / rel[j], 6) / tau_ref;   // (1/tau_ref)(R0/R)^6
  for (std::size_t j = 0; j < rho.size(); ++j) k_rot[j] = 1.0 / rho[j];
  (void)R0;   // R enters only as R/R0; R0 is recorded with tau_ref as the pair the rates assume
  auto put = [&](const char* name, std::vector<std::size_t> shape, std::vector<double> d) {
    BayesianDecayArray a; a.shape = std::move(shape); a.d = std::move(d); ex.arrays.emplace(name, std::move(a));
  };
  put("tau_c", {nt}, tb.kernel->tau());
  std::vector<double> base(K * nt, 0.0);
  for (std::size_t i = 0; i < nt; ++i) base[(i + 1) * nt + i] = 1.0;
  put("E_base", {K, nt}, std::move(base));
  put("E_S_R", {rel.size(), K, nt}, bayesian_transfer_rate_maps(tb, P, k_fret));
  put("E_S_rho", {rho.size(), K, nt}, bayesian_transfer_rate_maps(tb, P, k_rot));
  std::string how = t.count("acceptor_maps") ? t["acceptor_maps"].get<std::string>() : std::string("faithful");
  if (how != "faithful" && how != "exact") throw std::runtime_error("transfer.acceptor_maps must be 'faithful' or 'exact', not " + how);
  BayesianAcceptorMaps am = bayesian_transfer_acceptor_maps(tb, P, k_fret, tau_a, rho_a,
                                                            how == "exact" ? BayesianAcceptorConstruction::exact
                                                                           : BayesianAcceptorConstruction::faithful);
  put("E_S_Ag", {rel.size(), tau_a.size(), K, nt}, std::move(am.sensitised));
  put("E_S_Ag_rot_r", {rho_a.size(), rel.size(), tau_a.size(), K, nt}, std::move(am.sensitised_rot));
  put("E_A_dir", {tau_a.size(), K}, std::move(am.direct));
  put("E_A_dir_rot_r", {rho_a.size(), tau_a.size(), K}, std::move(am.direct_rot));
}

/**
 * \brief Read an experiment from `dir/manifest.json` and its raw arrays.
 *
 * The manifest names every array (`arrays: {name: {shape, dtype, file}}`, float64,
 * little-endian, one file each) and carries the tables: `axis`, `soft`,
 * `ref_spec_sd`, `keys`, `data_keys`, `scopes`, `parts`, `responses`, `variables`,
 * `fixed_values`, `pspline`, `spec_smooth`, `dim`, and optionally `transfer`, from
 * which the transfer tensors are built (`bayesian_decay_build_transfer_tensors`); a
 * `pspline.basis_degree` builds the spline matrix `spl` on the `rel` grid
 * (`bayesian_pspline_basis`). ucfret's
 * `s89_cpp/emit_cbm56_fixture.py` writes this format. Returns the parsed manifest
 * too, for a caller that stores more in it.
 */
inline nlohmann::json bayesian_decay_experiment_load(const std::string& dir, BayesianDecayExperiment& ex) {
  std::ifstream mi(dir + "/manifest.json");
  if (!mi) throw std::runtime_error("cannot open " + dir + "/manifest.json");
  nlohmann::json m;
  mi >> m;
  auto str = [](const nlohmann::json& j) { return j.is_null() ? std::string() : j.get<std::string>(); };
  ex.n_bins = m["axis"]["n"].get<std::size_t>();
  ex.dt = m["axis"]["dt"].get<double>();
  ex.period = m["axis"]["period"].get<double>();
  ex.soft = m["soft"].get<double>();
  ex.ref_spec_sd = m["ref_spec_sd"].get<double>();
  ex.dim = m["dim"].get<std::size_t>();
  ex.keys = m["keys"].get<std::vector<std::string>>();
  ex.data_keys = m["data_keys"].get<std::vector<std::string>>();
  for (auto& s : m["scopes"]) {
    BayesianDecayScope sc;
    sc.spectrum = str(s["spectrum"]); sc.don_ex = str(s["don_ex"]); sc.direct = str(s["direct"]);
    sc.rho = str(s["rho"]); sc.r0 = str(s["r0"]);
    sc.fret = s["fret"].get<bool>(); sc.sensitised = s["sensitised"].get<bool>(); sc.has_acc = s["has_acc"].get<bool>();
    ex.scopes.push_back(sc);
  }
  for (auto& p : m["parts"]) {
    BayesianDecayPart pt;
    pt.out = p["out"].get<std::size_t>(); pt.amp_index = p["amp_index"].get<std::size_t>();
    pt.response = p["response"].get<std::size_t>(); pt.scope = p["scope"].get<std::size_t>();
    pt.colour = p["colour"].get<int>(); pt.pol = p["pol"].get<int>();
    pt.channel = str(p["channel"]); pt.det = str(p["det"]); pt.phys = str(p["phys"]); pt.sample = str(p["sample"]);
    pt.offset_ns = p["offset_ns"].get<double>();
    ex.parts.push_back(pt);
  }
  for (auto& r : m["responses"])
    ex.responses.push_back({str(r["sample"]), str(r["det"]), str(r["bg_var"]), str(r["shift_var"]), str(r["offset_var"])});
  for (auto& v : m["variables"]) {
    BayesianDecayVariable var;
    var.name = str(v["name"]); var.transform = str(v["transform"]); var.family = str(v["family"]);
    var.offset = v["offset"].get<std::size_t>(); var.size = v["size"].get<std::size_t>();
    var.constrained_size = v["constrained_size"].get<std::size_t>();
    var.lo = v["lo"].get<double>(); var.hi = v["hi"].get<double>();
    var.a = v["a"].get<std::vector<double>>(); var.b = v["b"].get<std::vector<double>>();
    ex.variables.push_back(var);
  }
  for (auto it = m["fixed_values"].begin(); it != m["fixed_values"].end(); ++it)
    ex.fixed_values[it.key()] = it.value().get<std::vector<double>>();
  const auto& ps = m["pspline"];
  ex.pspline.n = ps["n"].get<std::size_t>(); ex.pspline.order = ps["order"].get<int>();
  ex.pspline.tilt_sd = ps["tilt_sd"].get<double>(); ex.pspline.quad_sd = ps["quad_sd"].get<double>();
  ex.pspline.rank = ps["rank"].get<double>(); ex.pspline.family = str(ps["family"]);
  ex.pspline.space = str(ps["space"]); ex.pspline.link = str(ps["link"]);
  if (m.count("spec_smooth")) ex.spec_smooth_logdet = m["spec_smooth"]["logdet"].get<double>();
  if (m.count("pile_up")) {
    const auto& pu = m["pile_up"];
    ex.pile_up = true;
    ex.repetition_rate_mhz = pu["repetition_rate_mhz"].get<double>();
    ex.dead_time_ns = pu["dead_time_ns"].get<double>();
    ex.measurement_time_s = pu["measurement_time_s"].get<double>();
  }
  for (auto it = m["arrays"].begin(); it != m["arrays"].end(); ++it) {
    const auto& spec = it.value();
    if (spec["dtype"].get<std::string>() != "float64") throw std::runtime_error("only float64 arrays: " + it.key());
    BayesianDecayArray arr;
    for (auto& s : spec["shape"]) arr.shape.push_back(s.get<std::size_t>());
    arr.d.resize(arr.size());
    std::ifstream in(dir + "/" + spec["file"].get<std::string>(), std::ios::binary);
    in.read(reinterpret_cast<char*>(arr.d.data()), std::streamsize(arr.d.size() * sizeof(double)));
    if (!in) throw std::runtime_error("short read of array " + it.key());
    ex.arrays.emplace(it.key(), std::move(arr));
  }
  if (m.count("transfer")) bayesian_decay_build_transfer_tensors(m["transfer"], ex);
  //: the B-spline basis of p(R/R0) built here when the manifest gives its degree instead of the matrix
  if (ps.count("basis_degree")) {
    if (ex.arrays.count("spl")) throw std::runtime_error("pspline.basis_degree and array spl both given");
    BayesianDecayArray spl;
    const std::size_t n_points = ex["rel"].size();
    spl.shape = {n_points, ex.pspline.n};
    spl.d = bayesian_pspline_basis(n_points, ex.pspline.n, ps["basis_degree"].get<int>());
    ex.arrays.emplace("spl", std::move(spl));
  }
  ex.kernel_ptr = std::make_shared<BayesianPeriodicKernel>(ex.axis(), ex["tau_c"].d);
  return m;
}

//! @}

//! \name The model of the counts
//! @{

//! The instrument stage's shapes in the basis's amplitude space: the response is
//! column 0, the flat background column K-1, both of unit sum by construction.
inline const internal::TCSPCInstrumentSettings& bayesian_decay_instrument_settings(std::size_t K) {
    static thread_local std::vector<double> e0, eK;
    static thread_local internal::TCSPCInstrumentSettings s;
    if (e0.size() != K) {
        e0.assign(K, 0.0); eK.assign(K, 0.0);
        e0[0] = 1.0; eK[K - 1] = 1.0;
        s = internal::TCSPCInstrumentSettings();
        s.response = e0.data(); s.flat = eK.data();
    }
    return s;
}

// ---------------------------------------------------------------------------
// the parameters: theta -> constrained values by name
// ---------------------------------------------------------------------------
using BayesianDecayValues = std::map<std::string, std::vector<double>>;

inline double bayesian_decay_scalar(const BayesianDecayValues& v, const std::string& name, double missing = 0.0) {
    auto it = v.find(name);
    return it == v.end() ? missing : it->second[0];
}

inline void bayesian_softmax_inplace(std::vector<double>& x) {
    double mx = -1e300;
    for (double t : x) mx = std::max(mx, t);
    double s = 0.0;
    for (double& t : x) { t = std::exp(t - mx); s += t; }
    for (double& t : x) t /= s;
}

inline BayesianDecayValues bayesian_decay_unpack(const BayesianDecayExperiment& f, const std::vector<double>& theta) {
    //: through BayesianTransforms.h; the sum-to-zero basis is the experiment's own
    //: (`Q_<name>`), since theta is defined on it
    BayesianDecayValues out;
    for (const BayesianDecayVariable& var : f.variables) {
        const std::string& name = var.name, & tr = var.transform;
        const double* z = theta.data() + var.offset;
        std::vector<double> x;
        if (tr == "identity") { x.resize(var.size); BayesianIdentityTransform<double>().to_constrained(z, var.size, x.data()); }
        else if (tr == "log") { x.resize(var.size); BayesianLogTransform<double>().to_constrained(z, var.size, x.data()); }
        else if (tr == "logit") { x.resize(var.size); BayesianLogitTransform<double>(var.lo, var.hi).to_constrained(z, var.size, x.data()); }
        else if (tr == "alr") { x.resize(var.size + 1); BayesianALRTransform<double>(var.size + 1).to_constrained(z, var.size, x.data()); }
        else if (tr == "sum_to_zero") {
            BayesianSumToZeroTransform<double> t;
            t.n = var.size + 1;
            t.Q = f["Q_" + name].d;
            x.resize(t.n);
            t.to_constrained(z, var.size, x.data());
        } else { throw std::runtime_error("bayesian_decay_unpack: transform " + tr + " of " + name); }
        out[name] = std::move(x);
    }
    for (const auto& kv : f.fixed_values)
        if (!out.count(kv.first)) out[kv.first] = kv.second;
    return out;
}

// ---------------------------------------------------------------------------
// the physics: species vectors per scope, channel mixing, instrument
// ---------------------------------------------------------------------------
struct BayesianDecayTensors {
    std::size_t K, Kint, nR, nA, nrho, nrho_a;
    const BayesianDecayArray* base, *S_R, *S_Ag, *S_Ag_rot_r, *A_dir, *A_dir_rot_r, *S_rho, *spl, *tau_c;
    explicit BayesianDecayTensors(const BayesianDecayExperiment& f)
        : base(&f["E_base"]), S_R(&f["E_S_R"]), S_Ag(&f["E_S_Ag"]), S_Ag_rot_r(&f["E_S_Ag_rot_r"]),
          A_dir(&f["E_A_dir"]), A_dir_rot_r(&f["E_A_dir_rot_r"]), S_rho(&f["E_S_rho"]), spl(&f["spl"]), tau_c(&f["tau_c"]) {
        K = base->shape[0]; Kint = base->shape[1]; nR = S_R->shape[0]; nA = S_Ag->shape[1];
        nrho = S_rho->shape[0]; nrho_a = S_Ag_rot_r->shape[0];
    }
};

//! p(R/R0) = softmax(spl @ c)
inline std::vector<double> bayesian_decay_distribution(const BayesianDecayTensors& e, const BayesianDecayValues& v) {
    const std::vector<double>& c = v.at("c");
    const std::size_t nR = e.spl->shape[0], nc = e.spl->shape[1];
    std::vector<double> q(nR, 0.0);
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t i = 0; i < nc; ++i) q[j] += e.spl->d[j * nc + i] * c[i];
    bayesian_softmax_inplace(q);
    return q;
}

inline std::vector<double> bayesian_decay_donor_spectrum(const BayesianDecayValues& v) { std::vector<double> s = v.at("spec_eps"); bayesian_softmax_inplace(s); return s; }

inline std::vector<double> bayesian_decay_reference_spectrum(const BayesianDecayTensors& e, const BayesianDecayValues& v, double sd) {
    if (v.count("spec_ref_eps")) { std::vector<double> s = v.at("spec_ref_eps"); bayesian_softmax_inplace(s); return s; }
    const double mu = v.at("log10_tau_ref")[0];
    std::vector<double> w(e.Kint);
    double s = 0.0;
    for (std::size_t i = 0; i < e.Kint; ++i) { const double u = std::log10(e.tau_c->d[i]); w[i] = std::exp(-0.5 * std::pow((u - mu) / sd, 2)); s += w[i]; }
    for (double& t : w) t /= s;
    return w;
}

//! the bayesian_decay_amplitudes of every physics channel (manifest `keys` order), K each,
//! instrument applied -- physics_amplitudes + instrument_amplitudes
inline std::vector<double> bayesian_decay_amplitudes(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const BayesianDecayValues& v) {
    const std::size_t K = e.K, Kint = e.Kint, nR = e.nR, nA = e.nA;
    const double ref_sd = f.ref_spec_sd;
    const std::vector<double> p = bayesian_decay_distribution(e, v), spec = bayesian_decay_donor_spectrum(v);
    const std::vector<double>& w_a = v.at("w_a"), &w_rho_a = v.at("w_rho_a");
    const double x = bayesian_decay_scalar(v, "x_d0"), QY_D = bayesian_decay_scalar(v, "QY_D"), QY_A = bayesian_decay_scalar(v, "QY_A");
    const double EX_AG = bayesian_decay_scalar(v, "EX_AG"), EX_DR = bayesian_decay_scalar(v, "EX_DR", 0.0);
    auto ex_of = [&](const std::string& s) { return s == "1" ? 1.0 : s == "EX_DR" ? EX_DR : s == "EX_AG" ? EX_AG : 0.0; };
    // a_dir = w_a @ A_dir; a_dir_r = w_a @ (w_rho_a . A_dir_rot_r)
    std::vector<double> a_dir(K, 0.0), a_dir_r(K, 0.0);
    for (std::size_t l = 0; l < nA; ++l) for (std::size_t k = 0; k < K; ++k) {
        a_dir[k] += w_a[l] * e.A_dir->d[l * K + k];
        double t = 0.0;
        for (std::size_t r = 0; r < e.nrho_a; ++r) t += w_rho_a[r] * e.A_dir_rot_r->d[(r * nA + l) * K + k];
        a_dir_r[k] += w_a[l] * t;
    }
    double per_mol = 0.0; for (double t : spec) per_mol += t;
    auto matvec = [&](const BayesianDecayArray& M, std::size_t row0, const std::vector<double>& s) {   // (K, Kint) block @ s
        std::vector<double> o(K, 0.0);
        for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) o[k] += M.d[row0 + k * Kint + i] * s[i];
        return o;
    };
    auto rot_op = [&](const std::string& rho_name) {               // Srho = sum_r w_r S_rho[r]  (K, Kint)
        const std::vector<double>& w = v.at(rho_name);
        std::vector<double> S(K * Kint, 0.0);
        for (std::size_t r = 0; r < e.nrho; ++r) for (std::size_t j = 0; j < K * Kint; ++j) S[j] += w[r] * e.S_rho->d[r * K * Kint + j];
        return S;
    };
    const std::vector<double> Srho_d = rot_op("w_rho");
    std::vector<double> Srho_ref;
    if (v.count("w_rho_ref")) Srho_ref = rot_op("w_rho_ref");

    struct Species { std::vector<double> don, acc, accr; };
    //: the scopes are independent: in parallel
    const std::size_t n_scopes = f.scopes.size();
    std::vector<Species> sp(n_scopes);
    internal::bayesian_parallel_for(n_scopes, [&](std::size_t si_) {
        const BayesianDecayScope& sc = f.scopes[si_];
        Species S{std::vector<double>(K, 0.0), std::vector<double>(K, 0.0), std::vector<double>(K, 0.0)};
        const bool fret = sc.fret, sens = sc.sensitised;
        std::vector<double> u(nR, 0.0); double n_da = 0.0;
        if (fret) for (std::size_t j = 0; j < nR; ++j) { u[j] = (1.0 - x) * p[j]; n_da += u[j]; }
        if (!sc.spectrum.empty()) {
            const bool is_ref = sc.spectrum == "reference";
            const std::vector<double> s_use = is_ref ? bayesian_decay_reference_spectrum(e, v, ref_sd) : spec;
            const double exd = ex_of(sc.don_ex);
            std::vector<double> q = matvec(*e.base, 0, s_use);
            if (fret) {
                for (double& t : q) t *= x;
                for (std::size_t j = 0; j < nR; ++j) {
                    const std::vector<double> qj = matvec(*e.S_R, j * K * Kint, s_use);
                    for (std::size_t k = 0; k < K; ++k) q[k] += u[j] * qj[k];
                }
            }
            for (std::size_t k = 0; k < K; ++k) S.don[k] = QY_D * exd * q[k];
        }
        if (sc.has_acc) {
            std::vector<double> qa(K, 0.0), qar(K, 0.0);
            if (sens) {
                //: sum_{j,l} u_j w_l S_Ag[j,l] @ spec and the same with sum_r w_rho_a[r],
                //: without a temporary per (j, l, r)
                for (std::size_t j = 0; j < nR; ++j) for (std::size_t l = 0; l < nA; ++l) {
                    if (u[j] == 0.0) continue;
                    const double ujl = u[j] * w_a[l];
                    const double* M0 = e.S_Ag->d.data() + (j * nA + l) * K * Kint;
                    for (std::size_t k = 0; k < K; ++k) {
                        double t = 0.0; const double* row = M0 + k * Kint;
                        for (std::size_t i = 0; i < Kint; ++i) t += row[i] * spec[i];
                        qa[k] += ujl * t;
                    }
                    for (std::size_t r = 0; r < e.nrho_a; ++r) {
                        const double wr = w_rho_a[r] * ujl;
                        const double* Mr = e.S_Ag_rot_r->d.data() + ((r * nR + j) * nA + l) * K * Kint;
                        for (std::size_t k = 0; k < K; ++k) {
                            double t = 0.0; const double* row = Mr + k * Kint;
                            for (std::size_t i = 0; i < Kint; ++i) t += row[i] * spec[i];
                            qar[k] += wr * t;
                        }
                    }
                }
            }
            double direct = 0.0;
            if (!sc.direct.empty()) {
                const std::string d = sc.direct;
                direct = d == "EX_AG*n_da*per_mol" ? EX_AG * n_da * per_mol : d == "n_da*per_mol" ? n_da * per_mol
                       : d == "EX_AG" ? EX_AG : d == "1" ? 1.0 : 0.0;
            }
            for (std::size_t k = 0; k < K; ++k) {
                S.acc[k] = QY_A * (qa[k] + a_dir[k] * direct);
                S.accr[k] = QY_A * (qar[k] + a_dir_r[k] * direct);
            }
        }
        sp[si_] = std::move(S);
    });

    // channel mixing and the instrument, per physics channel in `keys` order
    const std::vector<std::string>& keys = f.keys;
    std::vector<double> out(keys.size() * K, 0.0);
    std::map<std::string, const BayesianDecayPart*> part_of;
    for (auto& pt : f.parts) part_of[pt.channel] = &pt;
    for (std::size_t ic = 0; ic < keys.size(); ++ic) {
        const BayesianDecayPart& pt = *part_of.at(keys[ic]);
        const BayesianDecayScope& sc = f.scopes[pt.scope];
        const Species& S = sp[pt.scope];
        const bool vh = pt.pol == 1, red = pt.colour == 1;
        //: the polarised channel weight through PhotophysicsPolarisation.h: 2 - 3 l1 (VV), -1 + 3 l2 (VH)
        const double sign = anisotropy_weight(vh ? POL_VH : POL_VV, bayesian_decay_scalar(v, "l1"), bayesian_decay_scalar(v, "l2"));
        const double gf = vh ? (red && v.count("g_r") ? bayesian_decay_scalar(v, "g_r") : bayesian_decay_scalar(v, "g")) : 1.0;
        const bool is_ref = sc.r0 == "r0_ref";
        const std::vector<double>& Srho = is_ref ? Srho_ref : Srho_d;
        const double r0 = bayesian_decay_scalar(v, sc.r0), r0_a = bayesian_decay_scalar(v, "r0_a");
        std::vector<double> mix(K);
        const double Gc = red ? bayesian_decay_scalar(v, "G_RED") : bayesian_decay_scalar(v, "G_GREEN");
        const double CD = red ? bayesian_decay_scalar(v, "C_RD") : bayesian_decay_scalar(v, "C_GD"), CA = red ? bayesian_decay_scalar(v, "C_RA") : bayesian_decay_scalar(v, "C_GA");
        //: donor and acceptor each mixed by PhotophysicsPolarisation.h's mix_polarised
        //: ((iso + w r0 aniso) / g_channel, g applied once below)
        std::vector<double> rot(K, 0.0), pol_d(K), pol_a(K);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t i = 0; i < Kint; ++i) rot[k] += Srho[k * Kint + i] * S.don[1 + i];
        mix_polarised(S.don.data(), rot.data(), K, r0, sign, 1.0, pol_d.data());
        mix_polarised(S.acc.data(), S.accr.data(), K, r0_a, sign, 1.0, pol_a.data());
        for (std::size_t k = 0; k < K; ++k) mix[k] = Gc * (CD * pol_d[k] + CA * pol_a[k]) / gf;
        // the instrument stage (internal/TCSPCInstrument.h) in the basis's amplitude space:
        // every column sums to one, so the response and the flat background are unit
        // vectors -- the same stage TCSPCDecay applies in channel space
        const std::string samp = pt.sample, chan = pt.channel;
        const internal::TCSPCInstrumentSettings ins = bayesian_decay_instrument_settings(K);
        internal::TCSPCInstrumentParameters<double> ip;
        ip.scale = std::exp(bayesian_decay_scalar(v, "log_scale_" + samp));
        ip.scatter = bayesian_decay_scalar(v, "scat_" + chan);
        ip.background = bayesian_decay_scalar(v, "bkg_" + chan);
        internal::tcspc_instrument_components(mix.data(), K, ins, ip, mix.data());
        for (std::size_t k = 0; k < K; ++k) out[ic * K + k] = mix[k];
    }
    return out;
}

// ---------------------------------------------------------------------------
// the analytic amplitude Jacobian: d a2 / d theta, (n_keys * K, dim)
//
// The port of s88.amplitude_jacobian, table-driven: the donor block is
// (vv base + sum_j u_j S_R[j]) @ spectrum times QY_D and the scope's donor
// excitation; the acceptor block is the sensitised sums plus a DIRECT term
// whose partials in the spectrum, u and EX_AG follow from which of
// EX_AG * n_da * per_mol it contains; then the mixing and the instrument,
// both linear.  A coordinate held by the graph has no column.
// ---------------------------------------------------------------------------
struct BayesianDecayVariableInfo { std::size_t off = 0, size = 0; std::string transform; double lo = 0, hi = 1; bool present = false; };

inline BayesianDecayVariableInfo bayesian_decay_variable_info(const BayesianDecayExperiment& f, const std::string& name) {
    BayesianDecayVariableInfo vi;
    const BayesianDecayVariable* var = f.variable(name);
    if (var) { vi.off = var->offset; vi.size = var->size; vi.transform = var->transform; vi.lo = var->lo; vi.hi = var->hi; vi.present = true; }
    return vi;
}

//! d constrained / d z of a bayesian_decay_scalar coordinate, at its constrained value x
inline double bayesian_decay_dxdz(const BayesianDecayVariableInfo& vi, double x) {
    if (vi.transform == "log") return x;
    if (vi.transform == "logit") { const double sg = (x - vi.lo) / (vi.hi - vi.lo); return (vi.hi - vi.lo) * sg * (1.0 - sg); }
    return 1.0;
}

inline std::vector<double> bayesian_decay_amplitude_jacobian(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const BayesianDecayValues& v, bool drop_softmax_norm = false) {
    const std::size_t K = e.K, Kint = e.Kint, nR = e.nR, nA = e.nA, nra = e.nrho_a;
    const std::size_t dim = f.dim, nkeys = f.keys.size();
    const double ref_sd = f.ref_spec_sd;
    auto VI = [&](const std::string& n) { return bayesian_decay_variable_info(f, n); };
    const std::vector<double> p = bayesian_decay_distribution(e, v), spec = bayesian_decay_donor_spectrum(v);
    const std::vector<double>& w_a = v.at("w_a"), &w_rho_a = v.at("w_rho_a");
    const double x = bayesian_decay_scalar(v, "x_d0"), QY_D = bayesian_decay_scalar(v, "QY_D"), QY_A = bayesian_decay_scalar(v, "QY_A");
    const double EX_AG = bayesian_decay_scalar(v, "EX_AG"), EX_DR = bayesian_decay_scalar(v, "EX_DR", 0.0);
    double per_mol = 0.0; for (double t : spec) per_mol += t;
    auto ex_of = [&](const std::string& s_) { return s_ == "1" ? 1.0 : s_ == "EX_DR" ? EX_DR : s_ == "EX_AG" ? EX_AG : 0.0; };
    auto M = [&](std::size_t r, std::size_t c) { return r * dim + c; };

    // spectrum derivatives: (Kint, dim)
    std::vector<double> dspec(Kint * dim, 0.0), dspecr(Kint * dim, 0.0);
    const BayesianDecayVariableInfo v_eps = VI("spec_eps");
    if (v_eps.present) for (std::size_t i = 0; i < Kint; ++i) for (std::size_t j = 0; j < Kint; ++j)
        dspec[i * dim + v_eps.off + j] = (i == j ? spec[i] : 0.0) - (drop_softmax_norm ? 0.0 : spec[i] * spec[j]);
    std::vector<double> spec_ref;
    const BayesianDecayVariableInfo v_tau = VI("log10_tau_ref");
    if (v.count("log10_tau_ref") || v.count("spec_ref_eps")) {
        spec_ref = bayesian_decay_reference_spectrum(e, v, ref_sd);
        if (v_tau.present) {
            const double mu = bayesian_decay_scalar(v, "log10_tau_ref");
            std::vector<double> d(Kint); double sd_ = 0.0;
            for (std::size_t i = 0; i < Kint; ++i) { d[i] = (std::log10(e.tau_c->d[i]) - mu) / (ref_sd * ref_sd); sd_ += spec_ref[i] * d[i]; }
            for (std::size_t i = 0; i < Kint; ++i) dspecr[i * dim + v_tau.off] = spec_ref[i] * (d[i] - sd_) * bayesian_decay_dxdz(v_tau, mu);
        }
    }
    // p, dp/dc through the spline and the sum-to-zero basis: (nR, dim)
    const BayesianDecayVariableInfo v_c = VI("c");
    const BayesianDecayArray& Q = f["Q_c"];
    const std::size_t nc = e.spl->shape[1], nz = Q.shape[1];
    std::vector<double> dp(nR * dim, 0.0);
    {
        std::vector<double> BQ(nR * nz, 0.0);                     // spl @ Q
        for (std::size_t j = 0; j < nR; ++j) for (std::size_t a = 0; a < nc; ++a) {
            const double s_ = e.spl->d[j * nc + a];
            if (s_ == 0.0) continue;
            for (std::size_t b = 0; b < nz; ++b) BQ[j * nz + b] += s_ * Q.d[a * nz + b];
        }
        std::vector<double> pBQ(nz, 0.0);                          // p^T spl Q
        for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b) pBQ[b] += p[j] * BQ[j * nz + b];
        for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b)
            dp[j * dim + v_c.off + b] = p[j] * BQ[j * nz + b] - p[j] * pBQ[b];
    }
    const BayesianDecayVariableInfo v_x = VI("x_d0");
    const double dx = v_x.present ? bayesian_decay_dxdz(v_x, x) : 0.0;
    // ALR Jacobians: (n, n-1)
    auto alr_jac = [&](const std::string& n) {
        const std::vector<double>& w = v.at(n); const std::size_t nn = w.size();
        std::vector<double> J(nn * (nn - 1));
        for (std::size_t i = 0; i < nn; ++i) for (std::size_t j = 0; j + 1 < nn; ++j) J[i * (nn - 1) + j] = (i == j ? w[i] : 0.0) - w[i] * w[j];
        return J;
    };
    const BayesianDecayVariableInfo v_wa = VI("w_a"), v_wra = VI("w_rho_a");
    const std::vector<double> Jwa = alr_jac("w_a"), Jwra = alr_jac("w_rho_a");
    // contracted tensors
    std::vector<double> MR(nR * K, 0.0), MA(nR * nA * K, 0.0), MAr(nra * nR * nA * K, 0.0), base_spec(K, 0.0);
    for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) base_spec[k] += e.base->d[k * Kint + i] * spec[i];
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t k = 0; k < K; ++k) {
        double t = 0.0; for (std::size_t i = 0; i < Kint; ++i) t += e.S_R->d[(j * K + k) * Kint + i] * spec[i]; MR[j * K + k] = t;
    }
    internal::bayesian_parallel_for(nR, [&](std::size_t j) {  // in parallel over distances
        for (std::size_t l = 0; l < nA; ++l) for (std::size_t k = 0; k < K; ++k) {
            double t = 0.0; const double* row = e.S_Ag->d.data() + ((j * nA + l) * K + k) * Kint;
            for (std::size_t i = 0; i < Kint; ++i) t += row[i] * spec[i];
            MA[(j * nA + l) * K + k] = t;
            for (std::size_t r = 0; r < nra; ++r) {
                double tr = 0.0; const double* rr = e.S_Ag_rot_r->d.data() + (((r * nR + j) * nA + l) * K + k) * Kint;
                for (std::size_t i = 0; i < Kint; ++i) tr += rr[i] * spec[i];
                MAr[((r * nR + j) * nA + l) * K + k] = tr;
            }
        }
    });
    std::vector<double> a_dir(K, 0.0), a_dir_r(K, 0.0), A_dir_rot(nA * K, 0.0), dadirr_dwra(K * nra, 0.0);
    for (std::size_t l = 0; l < nA; ++l) for (std::size_t k = 0; k < K; ++k) {
        a_dir[k] += w_a[l] * e.A_dir->d[l * K + k];
        for (std::size_t r = 0; r < nra; ++r) {
            const double t = e.A_dir_rot_r->d[(r * nA + l) * K + k];
            A_dir_rot[l * K + k] += w_rho_a[r] * t;
            dadirr_dwra[k * nra + r] += w_a[l] * t;
        }
    }
    for (std::size_t l = 0; l < nA; ++l) for (std::size_t k = 0; k < K; ++k) a_dir_r[k] += w_a[l] * A_dir_rot[l * K + k];

    struct Blocks { std::vector<double> Dd, Da, Dr, don, acc, accr; };
    const std::size_t n_scopes = f.scopes.size();
    std::vector<Blocks> blk(n_scopes);
    internal::bayesian_parallel_for(n_scopes, [&](std::size_t si_) {
        const BayesianDecayScope& sc = f.scopes[si_];
        Blocks B{std::vector<double>(K * dim, 0.0), std::vector<double>(K * dim, 0.0), std::vector<double>(K * dim, 0.0),
                 std::vector<double>(K, 0.0), std::vector<double>(K, 0.0), std::vector<double>(K, 0.0)};
        const bool fret = sc.fret, sens = sc.sensitised;
        std::vector<double> u(nR, 0.0), du_dx(nR, 0.0); double n_da = 0.0;
        if (fret) for (std::size_t j = 0; j < nR; ++j) { u[j] = (1.0 - x) * p[j]; n_da += u[j]; du_dx[j] = -p[j] * dx; }
        // --- donor ---
        if (!sc.spectrum.empty()) {
            const bool is_ref = sc.spectrum == "reference";
            const std::vector<double>& S_use = is_ref ? spec_ref : spec;
            const std::vector<double>& dS = is_ref ? dspecr : dspec;
            const std::string exs = sc.don_ex; const double exd = ex_of(exs);
            // Mdon = vv base + sum_j u_j S_R[j]   (K, Kint)
            std::vector<double> Mdon(K * Kint);
            const double vv = fret ? x : 1.0;
            for (std::size_t k = 0; k < K * Kint; ++k) Mdon[k] = vv * e.base->d[k];
            if (fret) for (std::size_t j = 0; j < nR; ++j) { if (u[j] == 0.0) continue;
                for (std::size_t k = 0; k < K * Kint; ++k) Mdon[k] += u[j] * e.S_R->d[j * K * Kint + k]; }
            std::vector<double> q(K, 0.0);
            for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) q[k] += Mdon[k * Kint + i] * S_use[i];
            for (std::size_t k = 0; k < K; ++k) B.don[k] = QY_D * exd * q[k];
            for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) {
                const double mk = QY_D * exd * Mdon[k * Kint + i];
                if (mk == 0.0) continue;
                for (std::size_t c = 0; c < dim; ++c) B.Dd[M(k, c)] += mk * dS[i * dim + c];
            }
            const std::vector<double> qspec_base = [&] { std::vector<double> t(K, 0.0);
                for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) t[k] += e.base->d[k * Kint + i] * S_use[i]; return t; }();
            if (fret) {
                // through u (c and x) and vv = x; MR contracted with the donor spectrum (fret scopes carry it)
                for (std::size_t k = 0; k < K; ++k) {
                    for (std::size_t j = 0; j < nR; ++j) {
                        const double mr = QY_D * exd * MR[j * K + k];
                        for (std::size_t b = 0; b < nz; ++b) B.Dd[M(k, v_c.off + b)] += mr * (1.0 - x) * dp[j * dim + v_c.off + b];
                        if (v_x.present) B.Dd[M(k, v_x.off)] += mr * du_dx[j];
                    }
                    if (v_x.present) B.Dd[M(k, v_x.off)] += QY_D * exd * qspec_base[k] * dx;
                }
            }
            const BayesianDecayVariableInfo v_qd = VI("QY_D");
            if (v_qd.present) for (std::size_t k = 0; k < K; ++k) B.Dd[M(k, v_qd.off)] += exd * q[k] * bayesian_decay_dxdz(v_qd, QY_D);
            if (exs == "EX_DR") { const BayesianDecayVariableInfo v_ed = VI("EX_DR");
                if (v_ed.present) for (std::size_t k = 0; k < K; ++k) B.Dd[M(k, v_ed.off)] += QY_D * q[k] * bayesian_decay_dxdz(v_ed, EX_DR); }
        }
        // --- acceptor ---
        if (sc.has_acc) {
            std::string ds = sc.direct;
            const bool has_n = ds.find("n_da") != std::string::npos, has_pm = ds.find("per_mol") != std::string::npos;
            const bool has_ex = ds.find("EX_AG") != std::string::npos;
            const double direct = ds.empty() ? 0.0 : (has_ex ? EX_AG : 1.0) * (has_n ? n_da : 1.0) * (has_pm ? per_mol : 1.0);
            const double d_direct_dspec = has_pm ? (has_ex ? EX_AG : 1.0) * (has_n ? n_da : 1.0) : 0.0;
            const double d_direct_du = has_n ? (has_ex ? EX_AG : 1.0) * (has_pm ? per_mol : 1.0) : 0.0;
            const double d_direct_dex = has_ex ? (has_n ? n_da : 1.0) * (has_pm ? per_mol : 1.0) : 0.0;
            std::vector<double> qa(K, 0.0), qar(K, 0.0);
            if (sens) for (std::size_t j = 0; j < nR; ++j) { if (u[j] == 0.0) continue;
                for (std::size_t l = 0; l < nA; ++l) for (std::size_t k = 0; k < K; ++k) {
                    qa[k] += u[j] * w_a[l] * MA[(j * nA + l) * K + k];
                    for (std::size_t r = 0; r < nra; ++r) qar[k] += w_rho_a[r] * u[j] * w_a[l] * MAr[((r * nR + j) * nA + l) * K + k];
                } }
            for (std::size_t k = 0; k < K; ++k) { B.acc[k] = QY_A * (qa[k] + a_dir[k] * direct); B.accr[k] = QY_A * (qar[k] + a_dir_r[k] * direct); }
            // spectrum: sensitised matrices Qa, Qar (K, Kint) and the direct term's per_mol
            if (v_eps.present) {
                std::vector<double> Qa(K * Kint, 0.0), Qar(K * Kint, 0.0);
                if (sens) for (std::size_t j = 0; j < nR; ++j) { if (u[j] == 0.0) continue;
                    for (std::size_t l = 0; l < nA; ++l) {  // saxpy per block, r outside
                        const double ujl = u[j] * w_a[l];
                        const double* M0 = e.S_Ag->d.data() + (j * nA + l) * K * Kint;
                        for (std::size_t kk = 0; kk < K * Kint; ++kk) Qa[kk] += ujl * M0[kk];
                        for (std::size_t r = 0; r < nra; ++r) {
                            const double wr = w_rho_a[r] * ujl;
                            const double* Mr = e.S_Ag_rot_r->d.data() + ((r * nR + j) * nA + l) * K * Kint;
                            for (std::size_t kk = 0; kk < K * Kint; ++kk) Qar[kk] += wr * Mr[kk];
                        }
                    } }
                for (std::size_t k = 0; k < K; ++k) for (std::size_t i = 0; i < Kint; ++i) {
                    const double ma = QY_A * (Qa[k * Kint + i] + a_dir[k] * d_direct_dspec);
                    const double mr = QY_A * (Qar[k * Kint + i] + a_dir_r[k] * d_direct_dspec);
                    for (std::size_t c = v_eps.off; c < v_eps.off + v_eps.size; ++c) {
                        B.Da[M(k, c)] += ma * dspec[i * dim + c]; B.Dr[M(k, c)] += mr * dspec[i * dim + c];
                    }
                }
            }
            // u (through c and x)
            if (fret) for (std::size_t k = 0; k < K; ++k) for (std::size_t j = 0; j < nR; ++j) {
                double dqa = a_dir[k] * d_direct_du, dqr = a_dir_r[k] * d_direct_du;
                if (sens) for (std::size_t l = 0; l < nA; ++l) {
                    dqa += w_a[l] * MA[(j * nA + l) * K + k];
                    for (std::size_t r = 0; r < nra; ++r) dqr += w_rho_a[r] * w_a[l] * MAr[((r * nR + j) * nA + l) * K + k];
                }
                dqa *= QY_A; dqr *= QY_A;
                for (std::size_t b = 0; b < nz; ++b) {
                    const double dudc = (1.0 - x) * dp[j * dim + v_c.off + b];
                    B.Da[M(k, v_c.off + b)] += dqa * dudc; B.Dr[M(k, v_c.off + b)] += dqr * dudc;
                }
                if (v_x.present) { B.Da[M(k, v_x.off)] += dqa * du_dx[j]; B.Dr[M(k, v_x.off)] += dqr * du_dx[j]; }
            }
            // w_a
            if (v_wa.present) {
                const std::size_t nw = nA - 1;
                for (std::size_t k = 0; k < K; ++k) for (std::size_t l = 0; l < nA; ++l) {
                    double dqa = direct * e.A_dir->d[l * K + k], dqr = direct * A_dir_rot[l * K + k];
                    if (sens) for (std::size_t j = 0; j < nR; ++j) { if (u[j] == 0.0) continue;
                        dqa += u[j] * MA[(j * nA + l) * K + k];
                        for (std::size_t r = 0; r < nra; ++r) dqr += u[j] * w_rho_a[r] * MAr[((r * nR + j) * nA + l) * K + k]; }
                    for (std::size_t b = 0; b < nw; ++b) {
                        B.Da[M(k, v_wa.off + b)] += QY_A * dqa * Jwa[l * nw + b];
                        B.Dr[M(k, v_wa.off + b)] += QY_A * dqr * Jwa[l * nw + b];
                    }
                }
            }
            // w_rho_a
            if (v_wra.present) {
                const std::size_t nw = nra - 1;
                for (std::size_t k = 0; k < K; ++k) for (std::size_t r = 0; r < nra; ++r) {
                    double dqr = direct * dadirr_dwra[k * nra + r];
                    if (sens) for (std::size_t j = 0; j < nR; ++j) { if (u[j] == 0.0) continue;
                        for (std::size_t l = 0; l < nA; ++l) dqr += u[j] * w_a[l] * MAr[((r * nR + j) * nA + l) * K + k]; }
                    for (std::size_t b = 0; b < nw; ++b) B.Dr[M(k, v_wra.off + b)] += QY_A * dqr * Jwra[r * nw + b];
                }
            }
            const BayesianDecayVariableInfo v_ex = VI("EX_AG");
            if (v_ex.present && has_ex) for (std::size_t k = 0; k < K; ++k) {
                B.Da[M(k, v_ex.off)] += QY_A * a_dir[k] * d_direct_dex * bayesian_decay_dxdz(v_ex, EX_AG);
                B.Dr[M(k, v_ex.off)] += QY_A * a_dir_r[k] * d_direct_dex * bayesian_decay_dxdz(v_ex, EX_AG);
            }
            const BayesianDecayVariableInfo v_qa = VI("QY_A");
            if (v_qa.present) for (std::size_t k = 0; k < K; ++k) {
                B.Da[M(k, v_qa.off)] += B.acc[k] / QY_A * bayesian_decay_dxdz(v_qa, QY_A);
                B.Dr[M(k, v_qa.off)] += B.accr[k] / QY_A * bayesian_decay_dxdz(v_qa, QY_A);
            }
        }
        blk[si_] = std::move(B);
    });

    // mixing and the instrument, per physics channel
    std::vector<double> J(nkeys * K * dim, 0.0);
    std::map<std::string, const BayesianDecayPart*> part_of;
    for (auto& pt : f.parts) part_of[pt.channel] = &pt;
    internal::bayesian_parallel_for(nkeys, [&](std::size_t ic) {
        const BayesianDecayPart& pt = *part_of.at(f.keys[ic]);
        const std::size_t si = pt.scope; const BayesianDecayScope& sc = f.scopes[si]; const Blocks& B = blk[si];
        const bool vh = pt.pol == 1, red = pt.colour == 1;
        const bool has_don = !sc.spectrum.empty(), has_acc = sc.has_acc;
        //: the polarised channel weight through PhotophysicsPolarisation.h: 2 - 3 l1 (VV), -1 + 3 l2 (VH)
        const double sign = anisotropy_weight(vh ? POL_VH : POL_VV, bayesian_decay_scalar(v, "l1"), bayesian_decay_scalar(v, "l2"));
        const std::string gname = (vh && red && v.count("g_r")) ? "g_r" : "g";
        const double gf = vh ? bayesian_decay_scalar(v, gname) : 1.0;
        const std::string r0n = sc.r0, rhon = sc.rho;
        const double r0 = bayesian_decay_scalar(v, r0n), r0_a = bayesian_decay_scalar(v, "r0_a");
        const std::vector<double>& w_r = v.at(rhon);
        const std::size_t nrho = w_r.size();
        std::vector<double> Srho(K * Kint, 0.0);
        for (std::size_t r = 0; r < nrho; ++r) for (std::size_t j = 0; j < K * Kint; ++j) Srho[j] += w_r[r] * e.S_rho->d[r * K * Kint + j];
        const std::string cdn = red ? "C_RD" : "C_GD", can = red ? "C_RA" : "C_GA", gn = red ? "G_RED" : "G_GREEN";
        const double Gc = bayesian_decay_scalar(v, gn), CD = bayesian_decay_scalar(v, cdn), CA = bayesian_decay_scalar(v, can);
        std::vector<double> rot_dn(K, 0.0), pol_d(K, 0.0), pol_a(K, 0.0), mix(K, 0.0);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t i = 0; i < Kint; ++i) rot_dn[k] += Srho[k * Kint + i] * B.don[1 + i];
        mix_polarised(B.don.data(), rot_dn.data(), K, r0, sign, 1.0, pol_d.data());
        mix_polarised(B.acc.data(), B.accr.data(), K, r0_a, sign, 1.0, pol_a.data());
        for (std::size_t k = 0; k < K; ++k) mix[k] = Gc * (CD * pol_d[k] + CA * pol_a[k]) / gf;
        std::vector<double> Jk(K * dim, 0.0);
        if (has_don) {
            for (std::size_t k = 0; k < K; ++k) for (std::size_t c = 0; c < dim; ++c) {
                double t = B.Dd[M(k, c)];
                for (std::size_t i = 0; i < Kint; ++i) t += sign * r0 * Srho[k * Kint + i] * B.Dd[M(1 + i, c)];
                Jk[M(k, c)] += Gc * CD / gf * t;
            }
            const BayesianDecayVariableInfo v_wr = VI(rhon);
            if (v_wr.present) {
                const std::vector<double> Jwr = alr_jac(rhon); const std::size_t nw = nrho - 1;
                for (std::size_t k = 0; k < K; ++k) for (std::size_t r = 0; r < nrho; ++r) {
                    double dr = 0.0; for (std::size_t i = 0; i < Kint; ++i) dr += e.S_rho->d[(r * K + k) * Kint + i] * B.don[1 + i];
                    for (std::size_t b = 0; b < nw; ++b) Jk[M(k, v_wr.off + b)] += Gc * CD / gf * sign * r0 * dr * Jwr[r * nw + b];
                }
            }
            const BayesianDecayVariableInfo v_r0 = VI(r0n);
            if (v_r0.present) for (std::size_t k = 0; k < K; ++k) Jk[M(k, v_r0.off)] += Gc * CD / gf * sign * rot_dn[k] * bayesian_decay_dxdz(v_r0, r0);
        }
        if (has_acc) {
            for (std::size_t k = 0; k < K; ++k) for (std::size_t c = 0; c < dim; ++c)
                Jk[M(k, c)] += Gc * CA / gf * (B.Da[M(k, c)] + sign * r0_a * B.Dr[M(k, c)]);
            const BayesianDecayVariableInfo v_ra = VI("r0_a");
            if (v_ra.present) for (std::size_t k = 0; k < K; ++k) Jk[M(k, v_ra.off)] += Gc * CA / gf * sign * B.accr[k] * bayesian_decay_dxdz(v_ra, r0_a);
        }
        const double dmix_dsign_c = 1.0 / gf;
        const BayesianDecayVariableInfo v_l1 = VI("l1"), v_l2 = VI("l2");
        for (std::size_t k = 0; k < K; ++k) {
            const double dms = Gc * dmix_dsign_c * (CD * r0 * rot_dn[k] * (has_don ? 1.0 : 0.0) + CA * r0_a * B.accr[k] * (has_acc ? 1.0 : 0.0));
            if (!vh && v_l1.present) Jk[M(k, v_l1.off)] += dms * (-3.0) * bayesian_decay_dxdz(v_l1, bayesian_decay_scalar(v, "l1"));
            if (vh && v_l2.present) Jk[M(k, v_l2.off)] += dms * 3.0 * bayesian_decay_dxdz(v_l2, bayesian_decay_scalar(v, "l2"));
        }
        const BayesianDecayVariableInfo v_g = VI(gname), v_cd = VI(cdn), v_ca = VI(can), v_gc = VI(gn);
        for (std::size_t k = 0; k < K; ++k) {
            if (vh && v_g.present) Jk[M(k, v_g.off)] += -mix[k] / gf * bayesian_decay_dxdz(v_g, gf);
            if (v_cd.present) Jk[M(k, v_cd.off)] += Gc / gf * pol_d[k] * bayesian_decay_dxdz(v_cd, CD);
            if (v_ca.present) Jk[M(k, v_ca.off)] += Gc / gf * pol_a[k] * bayesian_decay_dxdz(v_ca, CA);
            if (v_gc.present) Jk[M(k, v_gc.off)] += mix[k] / Gc * bayesian_decay_dxdz(v_gc, Gc);
        }
        // the instrument stage's derivatives (internal/TCSPCInstrument.h), in amplitude space
        const std::string samp = pt.sample, chan = pt.channel;
        const internal::TCSPCInstrumentSettings ins = bayesian_decay_instrument_settings(K);
        internal::TCSPCInstrumentParameters<double> ip;
        ip.scale = std::exp(bayesian_decay_scalar(v, "log_scale_" + samp));
        ip.scatter = bayesian_decay_scalar(v, "scat_" + chan);
        ip.background = bayesian_decay_scalar(v, "bkg_" + chan);
        std::vector<double> dpar(K * 4);
        internal::tcspc_instrument_components_jacobian(mix.data(), K, ins, ip, Jk.data(), dim, J.data() + ic * K * dim, dpar.data());
        const BayesianDecayVariableInfo v_ls = VI("log_scale_" + samp), v_sc = VI("scat_" + chan), v_bk = VI("bkg_" + chan);
        for (std::size_t k = 0; k < K; ++k) {
            if (v_ls.present) J[(ic * K + k) * dim + v_ls.off] += ip.scale * dpar[k * 4 + 0];   // d scale / d log_scale = scale
            if (v_sc.present) J[(ic * K + k) * dim + v_sc.off] += dpar[k * 4 + 1] * bayesian_decay_dxdz(v_sc, ip.scatter);
            if (v_bk.present) J[(ic * K + k) * dim + v_bk.off] += dpar[k * 4 + 3] * bayesian_decay_dxdz(v_bk, ip.background);
        }
    });
    return J;
}

// ---------------------------------------------------------------------------
// the response with its tangents, the expected counts and the count Jacobian
//
// Forward-mode by hand through the four steps of InstrumentModel.basis --
// drop_background, the clamped phase-ramp shift, the unit-sum normalisation,
// the exact periodic kernel with its clamp and column normalisation -- for
// the response's own coordinates: its background, the detector's shift and
// the sample's time offset. Every step is linear in its tangent apart from
// the softplus (torch's threshold 20 in its derivative too) and the clamps
// (derivative 1 where the argument is >= 0, torch's clamp_min rule).
// ---------------------------------------------------------------------------

struct BayesianDecayResponseTangents {
    std::vector<double> B;                    // (n, K)
    std::vector<std::vector<double>> dB;      // per tangent, (n, K), d/dz
    std::vector<std::string> names;           // the free coordinates the tangents belong to
    std::vector<std::size_t> offsets;              // their offsets in theta
    std::vector<double> chain;                // d constrained / d z
};


inline double bayesian_decay_constrained_scalar(const BayesianDecayExperiment& f, const BayesianDecayValues& v, const std::string& name) {
    if (v.count(name)) return v.at(name)[0];
    auto it = f.fixed_values.find(name);
    if (it != f.fixed_values.end()) return it->second[0];
    return 0.0;
}

inline BayesianDecayResponseTangents bayesian_decay_response_basis(const BayesianDecayExperiment& f, const std::vector<double>& tau_c, const BayesianDecayValues& v,
                                    const std::string& det, const std::string& samp, bool derivs = true,
                                    bool clamp_in_tangent = true) {
    (void)tau_c;                      // the kernel on this grid is the experiment's own (f.kernel())
    const BayesianDecayAxis ax = f.axis();
    
    const double soft = f.soft, dt = ax.dt;
    std::string phys; double off_ns = 0.0;
    for (auto& pt : f.parts) if (pt.det == det) { phys = pt.phys; off_ns = pt.offset_ns; break; }
    std::size_t ri = 0;
    for (; ri < f.responses.size(); ++ri)
        if (f.responses[ri].sample == samp && f.responses[ri].det == phys) break;
    const BayesianDecayResponse& r = f.responses[ri];
    const std::string bg_n = r.bg_var, sh_n = r.shift_var, of_n = r.offset_var;
    const std::vector<double>& raw = f["response_" + std::to_string(ri)].d;

    BayesianDecayResponseTangents out;
    //: each free coordinate's tangent is d/d b (the background fraction) or d/d shift,
    //: times its transform's chain factor; the detector's shift and the sample's
    //: offset share the shift tangent
    std::vector<int> kind_of;    // per name: 0 background, 1 shift
    auto add = [&](const std::string& nm, int k) {
        if (nm.empty() || !derivs) return;
        const BayesianDecayVariableInfo vi = bayesian_decay_variable_info(f, nm);
        if (!vi.present) return;
        out.names.push_back(nm); out.offsets.push_back(vi.off); kind_of.push_back(k);
        out.chain.push_back(bayesian_decay_dxdz(vi, bayesian_decay_constrained_scalar(f, v, nm)));
    };
    add(bg_n, 0); add(sh_n, 1); add(of_n, 1);
    const double shift_ns = bayesian_decay_constrained_scalar(f, v, sh_n) + off_ns + (of_n.empty() ? 0.0 : bayesian_decay_constrained_scalar(f, v, of_n));
    BayesianResponseOptions opt;
    opt.soft = soft;
    opt.clamp_in_tangent = clamp_in_tangent;
    BayesianResponseBasis rb = bayesian_response_basis(f.kernel(), raw, bayesian_decay_constrained_scalar(f, v, bg_n),
                                                                shift_ns / dt, derivs && !out.names.empty(), opt);
    out.dB.resize(out.names.size());
    for (std::size_t t = 0; t < out.names.size(); ++t) {
        out.dB[t] = kind_of[t] == 0 ? rb.dB_background : rb.dB_shift;
        const double c = out.chain[t] * (kind_of[t] == 0 ? 1.0 : 1.0 / dt);
        for (double& q : out.dB[t]) q *= c;
    }
    out.B = std::move(rb.B);
    return out;
}

//! the expected counts (n_data, n) and, when asked, d counts / d theta (n_data * n, dim)
//! `cols`, when given, receives per data histogram the columns its Jacobian rows
//! can be non-zero in: the union over its parts of the amplitude Jacobian's
//! non-zero columns and the response's own coordinates (a donor-only or a
//! reference-dye histogram touches half of theta or less)
inline std::vector<double> bayesian_decay_expected_counts(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const BayesianDecayValues& v,
                                           std::vector<double>* J = nullptr, const std::vector<double>* Ja = nullptr,
                                           const std::vector<double>* a2_in = nullptr,
                                           std::vector<std::vector<std::size_t>>* cols = nullptr, bool clamp_in_tangent = true) {
    const BayesianDecayAxis ax = f.axis();
    const std::size_t n = ax.n, K = e.K, nd = f.data_keys.size(), dim = f.dim;
    const double soft = f.soft;
    const std::vector<double> a2 = a2_in ? *a2_in : bayesian_decay_amplitudes(f, e, v);
    const std::vector<double>& tau_c = e.tau_c->d;
    std::vector<double> lam(nd * n, 0.0), raw(nd * n, 0.0);
    if (J) J->assign(nd * n * dim, 0.0);
    std::vector<std::vector<std::size_t>> out_cols(nd);
    //: one basis per (detector, sample), built in parallel; the kernel's
    //: FFT cache is filled first, on this thread, so the workers only read it
    
    const std::vector<BayesianDecayPart>& parts = f.parts;
    std::vector<std::pair<std::string, std::string>> combos;
    std::vector<std::size_t> combo_of(parts.size());
    for (std::size_t q = 0; q < parts.size(); ++q) {
        auto key = std::make_pair(parts[q].det, parts[q].sample);
        auto it = std::find(combos.begin(), combos.end(), key);
        combo_of[q] = std::size_t(it - combos.begin());
        if (it == combos.end()) combos.push_back(key);
    }
    std::vector<BayesianDecayResponseTangents> bases(combos.size());
    internal::bayesian_parallel_for(combos.size(), [&](std::size_t c) { bases[c] = bayesian_decay_response_basis(f, tau_c, v, combos[c].first, combos[c].second, J != nullptr, clamp_in_tangent); });
    std::vector<std::vector<std::size_t>> parts_of_out(nd);
    for (std::size_t q = 0; q < parts.size(); ++q) parts_of_out[parts[q].out].push_back(q);
    internal::bayesian_parallel_for(nd, [&](std::size_t o_) {
      for (std::size_t q : parts_of_out[o_]) [&, q]() {
        const BayesianDecayPart& pt = parts[q];
        const std::size_t o = pt.out, ai = pt.amp_index;
        const BayesianDecayResponseTangents& rb = bases[combo_of[q]];
        const double* a = a2.data() + ai * K;
        for (std::size_t i = 0; i < n; ++i) {
            double t = 0.0; for (std::size_t k = 0; k < K; ++k) t += rb.B[i * K + k] * a[k];
            raw[o * n + i] += t;
        }
        if (!J) return;
        // the part's live columns, and its amplitude Jacobian gathered onto them
        std::vector<std::size_t> pc;
        for (std::size_t c = 0; c < dim; ++c) {
            bool live = false;
            for (std::size_t k = 0; k < K && !live; ++k) live = (*Ja)[(ai * K + k) * dim + c] != 0.0;
            if (live) pc.push_back(c);
        }
        const std::size_t mp = pc.size();
        std::vector<double> jg(K * mp);
        for (std::size_t k = 0; k < K; ++k) for (std::size_t c = 0; c < mp; ++c) jg[k * mp + c] = (*Ja)[(ai * K + k) * dim + pc[c]];
        for (std::size_t c : pc) out_cols[o].push_back(c);
        for (std::size_t t = 0; t < rb.names.size(); ++t) out_cols[o].push_back(rb.offsets[t]);
        std::vector<double> acc(mp);
        for (std::size_t i = 0; i < n; ++i) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (std::size_t k = 0; k < K; ++k) {
                const double bk = rb.B[i * K + k];
                if (bk == 0.0) continue;
                const double* jr = jg.data() + k * mp;
                for (std::size_t c = 0; c < mp; ++c) acc[c] += bk * jr[c];
            }
            double* row = J->data() + (o * n + i) * dim;
            for (std::size_t c = 0; c < mp; ++c) row[pc[c]] += acc[c];
            for (std::size_t t = 0; t < rb.names.size(); ++t) {
                double q = 0.0; for (std::size_t k = 0; k < K; ++k) q += rb.dB[t][i * K + k] * a[k];
                row[rb.offsets[t]] += q;
            }
        }
      }();
    });
    for (auto& oc : out_cols) { std::sort(oc.begin(), oc.end()); oc.erase(std::unique(oc.begin(), oc.end()), oc.end()); }
    //: detection per histogram (internal/TCSPCInstrument.h): pile-up on the fluorescence
    //: and scatter, the uncorrelated background added after it, the DNL multiply. The
    //: background left the amplitude-space stage in the flat column (K-1): per part its
    //: amplitude is bkg / (1 + scat + bkg) of the part's amplitude total (every basis
    //: column sums to one, no pattern), so it is taken back out here, and with
    //: g = f lin the counts are g (raw - bg) + lin bg -- J = g J_raw + (lin - g) J_bg
    const bool dnl = f.arrays.count("linearization") > 0;
    if (f.pile_up || dnl) {
        std::vector<double> bg(nd * n, 0.0), Jbg;
        if (J) Jbg.assign(nd * n * dim, 0.0);
        for (std::size_t q = 0; q < parts.size(); ++q) {
            const BayesianDecayPart& pt = parts[q];
            const std::size_t o = pt.out, ai = pt.amp_index;
            const BayesianDecayResponseTangents& rb = bases[combo_of[q]];
            const double scat = bayesian_decay_scalar(v, "scat_" + pt.channel), bkg = bayesian_decay_scalar(v, "bkg_" + pt.channel);
            const double den = 1.0 + scat + bkg, ratio = bkg / den;
            double total = 0.0; for (std::size_t k = 0; k < K; ++k) total += a2[ai * K + k];
            const double amp = ratio * total;
            for (std::size_t i = 0; i < n; ++i) bg[o * n + i] += rb.B[i * K + K - 1] * amp;
            if (!J) continue;
            std::vector<double> jrow(dim, 0.0);
            for (std::size_t k = 0; k < K; ++k) for (std::size_t c = 0; c < dim; ++c) jrow[c] += ratio * (*Ja)[(ai * K + k) * dim + c];
            const BayesianDecayVariableInfo v_sc = bayesian_decay_variable_info(f, "scat_" + pt.channel), v_bk = bayesian_decay_variable_info(f, "bkg_" + pt.channel);
            if (v_sc.present) jrow[v_sc.off] += total * (-bkg / (den * den)) * bayesian_decay_dxdz(v_sc, scat);
            if (v_bk.present) jrow[v_bk.off] += total * ((1.0 + scat) / (den * den)) * bayesian_decay_dxdz(v_bk, bkg);
            for (std::size_t i = 0; i < n; ++i) {
                const double b = rb.B[i * K + K - 1];
                if (b == 0.0) continue;
                for (std::size_t c : out_cols[o]) Jbg[(o * n + i) * dim + c] += b * jrow[c];
            }
        }
        const BayesianDecayArray* lin = dnl ? &f["linearization"] : nullptr;
        const BayesianDecayArray& y = f["y"];
        internal::bayesian_parallel_for(nd, [&](std::size_t o) {
            internal::TCSPCInstrumentSettings ds;
            if (f.pile_up) {
                ds.pile_up_data = y.d.data() + o * n; ds.pile_up_n_data = int(n);
                ds.repetition_rate_mhz = f.repetition_rate_mhz; ds.dead_time_ns = f.dead_time_ns; ds.measurement_time_s = f.measurement_time_s;
            }
            if (lin) ds.linearization = lin->d.data() + o * n;
            std::vector<double> g(n, 1.0), fl(n);
            internal::tcspc_instrument_detection(g.data(), n, ds, (const double*)nullptr);     // g = f lin
            for (std::size_t i = 0; i < n; ++i) fl[i] = raw[o * n + i] - bg[o * n + i];
            internal::tcspc_instrument_detection(fl.data(), n, ds, bg.data() + o * n);
            for (std::size_t i = 0; i < n; ++i) raw[o * n + i] = fl[i];
            if (!J) return;
            for (std::size_t i = 0; i < n; ++i) {
                const double li = lin ? lin->d[o * n + i] : 1.0;
                double* row = J->data() + (o * n + i) * dim;
                const double* rbg = Jbg.data() + (o * n + i) * dim;
                for (std::size_t c : out_cols[o]) row[c] = g[i] * row[c] + (li - g[i]) * rbg[c];
            }
        });
    }
    for (std::size_t o = 0; o < nd; ++o) for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = o * n + i;
        lam[j] = bayesian_soft_positive(raw[j], soft, BAYESIAN_TORCH_SOFTPLUS_THRESHOLD);
        if (J) { const double sg = bayesian_soft_positive_derivative(raw[j], soft); for (std::size_t c : out_cols[o]) (*J)[j * dim + c] *= sg; }
    }
    if (cols) *cols = out_cols;
    return lam;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANDECAYMODEL_H */

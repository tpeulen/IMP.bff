/**
 *  \file ProbeModelComparison.cpp
 *  \brief An accessible volume and a rotamer ensemble of the same site.
 *
 *  Port of `compare_av_and_rotamer_positions`, `compare_av_and_rotamer_pairs`,
 *  `markdown_table`, `summary_numbers` and the two reference cases from the
 *  Python program `bin/imp_bff`. Where the program used numpy the sums are
 *  numpy's (internal/NumpyCompat.h), so the pins agree to the bit where numpy
 *  is deterministic.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ProbeModelComparison.h>

#include <IMP/bff/FRET.h>
#include <IMP/bff/HierarchyFrame.h>
#include <IMP/bff/ProbeAccessibleVolume.h>
#include <IMP/bff/ProbeAccessibleVolumeDecorator.h>
#include <IMP/bff/ProbeDataPaths.h>
#include <IMP/bff/ProbeRotamer.h>
#include <IMP/bff/States.h>
#include <IMP/bff/internal/NumpyCompat.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace probe_model_comparison {

namespace nc = internal::numpy_compat;

std::string read_file(const std::string& path) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in) {
    IMP_THROW("cannot read " << path, IOException);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

//! The keys of the object at \p key of a JSON document's top-level object,
//! in the order the text writes them (nlohmann sorts; Python keeps order).
class KeyOrder {
 public:
  explicit KeyOrder(const std::string& text) : s_(text), i_(0) {}

  std::vector<std::string> of(const std::string& key) {
    i_ = 0;
    ws();
    expect('{');
    std::vector<std::string> out;
    ws();
    if (peek() == '}') return out;
    while (true) {
      ws();
      const std::string k = string_token();
      ws();
      expect(':');
      ws();
      if (k == key && peek() == '{') {
        ++i_;
        ws();
        if (peek() == '}') return out;
        while (true) {
          ws();
          out.push_back(string_token());
          ws();
          expect(':');
          skip_value();
          ws();
          if (peek() == ',') {
            ++i_;
            continue;
          }
          return out;
        }
      }
      skip_value();
      ws();
      if (peek() == ',') {
        ++i_;
        continue;
      }
      return out;
    }
  }

 private:
  char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  void expect(char c) {
    if (peek() != c) {
      IMP_THROW("malformed JSON near offset " << i_, ValueException);
    }
    ++i_;
  }
  std::string string_token() {
    const std::size_t start = i_;
    skip_string();
    return nlohmann::json::parse(s_.substr(start, i_ - start)).get<std::string>();
  }
  void skip_string() {
    expect('"');
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\') ++i_;
      ++i_;
    }
    expect('"');
  }
  void skip_value() {
    ws();
    const char c = peek();
    if (c == '"') {
      skip_string();
    } else if (c == '{' || c == '[') {
      int depth = 0;
      do {
        const char d = peek();
        if (d == '"') {
          skip_string();
          continue;
        }
        if (d == '{' || d == '[') ++depth;
        if (d == '}' || d == ']') --depth;
        ++i_;
      } while (depth > 0 && i_ < s_.size());
    } else {
      while (i_ < s_.size() && s_[i_] != ',' && s_[i_] != '}' && s_[i_] != ']') ++i_;
    }
  }
  const std::string& s_;
  std::size_t i_;
};

//! A view getter's result as a vector.
template <class T, class G>
std::vector<double> take(const T& obj, G getter) {
  double* view = nullptr;
  int n = 0;
  (obj.*getter)(&view, &n);
  std::vector<double> out(view, view + n);
  std::free(view);
  return out;
}

std::string upper(std::string s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
  }
  return s;
}

bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

//! (R_mp, <R_DA>, <R_DA>_E, sigma_R, <kappa^2>) from the full pair matrix.
void ensemble_pair_statistics(const ProbeRotamerEnsemble& e1, const ProbeRotamerEnsemble& e2,
                              double forster_radius, double out[5]) {
  const std::vector<double>& p1 = e1.get_points_vector();
  const std::vector<double>& p2 = e2.get_points_vector();
  const std::size_t n1 = p1.size() / 4, n2 = p2.size() / 4;
  std::vector<double> xyz1(3 * n1), w1(n1), xyz2(3 * n2), w2(n2);
  for (std::size_t i = 0; i < n1; ++i) {
    for (int k = 0; k < 3; ++k) xyz1[3 * i + k] = p1[4 * i + k];
    w1[i] = p1[4 * i + 3];
  }
  for (std::size_t i = 0; i < n2; ++i) {
    for (int k = 0; k < 3; ++k) xyz2[3 * i + k] = p2[4 * i + k];
    w2[i] = p2[4 * i + 3];
  }
  const FRETPairGeometry g = fret_pair_geometry(xyz1, w1, xyz2, w2);
  const std::vector<double>& w = g.weight;
  const std::vector<double>& r = g.R;
  std::vector<double> rw(r.size()), rrw(r.size());
  for (std::size_t i = 0; i < r.size(); ++i) {
    rw[i] = r[i] * w[i];
    rrw[i] = (r[i] * r[i]) * w[i];
  }
  const std::vector<double> m1 = take(e1, &States::get_mean_position);
  const std::vector<double> m2 = take(e2, &States::get_mean_position);
  double d[3] = {m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2]};
  const double rmp = nc::norm3(d);
  const double rda = nc::pairwise_sum(rw);
  const double sigma = std::sqrt((std::max)(nc::pairwise_sum(rrw) - rda * rda, 0.0));
  const double e_static = fret_pair_efficiencies(g, forster_radius).static_efficiency;
  double rda_e;
  if (e_static <= 0) {
    rda_e = rda;
  } else if (e_static >= 1) {
    rda_e = 0.0;
  } else {
    rda_e = forster_radius * std::pow(1.0 / e_static - 1.0, 1.0 / 6.0);
  }
  double kappa2 = 2.0 / 3.0;
  const std::vector<double> mu1 = take(e1, &States::get_orientations);
  const std::vector<double> mu2 = take(e2, &States::get_orientations);
  if (mu1.size() / 3 == n1 && mu2.size() / 3 == n2 && !mu1.empty() && !mu2.empty()) {
    kappa2 = fret_pair_geometry(xyz1, w1, xyz2, w2, mu1, mu2).kappa2_avg;
  }
  out[0] = rmp;
  out[1] = rda;
  out[2] = rda_e;
  out[3] = sigma;
  out[4] = kappa2;
}

//! RMS distance of a weighted cloud from its mean.
double extent(const States& cloud) {
  const std::vector<double>& p = cloud.get_points_vector();
  const std::size_t n = p.size() / 4;
  if (n == 0) return std::numeric_limits<double>::quiet_NaN();
  const double total = nc::pairwise_sum(&p[3], n, 4);
  const std::vector<double> mean = take(cloud, &States::get_mean_position);
  std::vector<double> wd2(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double w = p[4 * i + 3] / total;
    const double dx = p[4 * i] - mean[0], dy = p[4 * i + 1] - mean[1], dz = p[4 * i + 2] - mean[2];
    const double d2 = ((0.0 + dx * dx) + dy * dy) + dz * dz;
    wd2[i] = w * d2;
  }
  return std::sqrt(nc::pairwise_sum(wd2));
}

struct Built {
  std::vector<ProbeAccessibleVolume> avs;
  std::vector<ProbeRotamerEnsemble> ensembles;
};

void build(const AVRotamerCase& c, double temperature, std::vector<AVRotamerPosition>& positions,
           Built& built) {
  const std::vector<ProteinFrame> frames = load_protein_frames(c.pdb, 1);
  if (frames.empty()) {
    IMP_THROW(c.pdb << " holds no model", ValueException);
  }
  const ProteinFrame& frame = frames[0];
  const std::size_t n_protein = frame.coords.size() / 3;
  std::vector<std::string> p_names(n_protein), p_chain(n_protein, std::string());
  std::vector<bool> p_heavy(n_protein);
  for (std::size_t i = 0; i < n_protein; ++i) {
    p_names[i] = upper(frame.atom_names[i]);
    p_heavy[i] = !starts_with(p_names[i], "H");
    if (!frame.chain_ids.empty()) p_chain[i] = upper(frame.chain_ids[i]);
  }
  for (std::size_t k = 0; k < c.names.size(); ++k) {
    const std::string& name = c.names[k];
    nlohmann::json pos = nlohmann::json::parse(c.positions_json[k]);
    pos.erase("strip_mask");
    // `float(disc_step or p.get("simulation_grid_resolution", 1.5))`
    const double ds = pos.contains("simulation_grid_resolution")
                              ? pos["simulation_grid_resolution"].get<double>()
                              : 1.5;
    ProbeAccessibleVolume av = get_av_from_structure(c.pdb, pos.dump(), ds);
    std::string chain;
    if (pos.contains("chain_identifier") && pos["chain_identifier"].is_string()) {
      chain = pos["chain_identifier"].get<std::string>();
    }
    const int residue = pos["residue_seq_number"].get<int>();
    ProbeRotamerSiteOptions options(temperature);
    ProbeRotamerEnsemble ens =
            ProbeRotamerEnsemble::from_site(c.pdb, chain, residue, c.libraries[k], options, name);

    // the interpenetration of the ensemble with the protein outside its residue
    const std::string ens_chain = ens.get_chain();
    const std::string ens_chain_upper = upper(ens_chain);
    std::vector<double> prot;
    for (std::size_t i = 0; i < n_protein; ++i) {
      const bool same = frame.residue_indices[i] == ens.get_residue() &&
                        (p_chain[i].empty() || p_chain[i] == ens_chain_upper || ens_chain.empty());
      if (p_heavy[i] && !same) {
        prot.push_back(frame.coords[3 * i]);
        prot.push_back(frame.coords[3 * i + 1]);
        prot.push_back(frame.coords[3 * i + 2]);
      }
    }
    const std::vector<std::string> atom_names = ens.get_atom_names();
    std::vector<std::size_t> keep;
    for (std::size_t a = 0; a < atom_names.size(); ++a) {
      const std::string nm = upper(atom_names[a]);
      if (starts_with(nm, "H") || nm == "N" || nm == "CA" || nm == "C" || nm == "O" || nm == "OXT") {
        continue;
      }
      keep.push_back(a);
    }
    const std::vector<double> atoms = take(ens, &ProbeRotamerEnsemble::get_atoms);
    const std::vector<double> weights = take(ens, &ProbeRotamerEnsemble::get_weights);
    const std::size_t n_atoms = atom_names.size();
    const std::size_t n_prot = prot.size() / 3;
    double w_clash = 0.0, d_min = std::numeric_limits<double>::infinity();
    for (int rot = 0; rot < ens.get_n_rotamers(); ++rot) {
      double dmin = std::numeric_limits<double>::infinity();
      for (std::size_t kk = 0; kk < keep.size(); ++kk) {
        const double* x = &atoms[3 * (rot * n_atoms + keep[kk])];
        for (std::size_t j = 0; j < n_prot; ++j) {
          const double dx = x[0] - prot[3 * j], dy = x[1] - prot[3 * j + 1],
                       dz = x[2] - prot[3 * j + 2];
          const double dist = std::sqrt(((0.0 + dx * dx) + dy * dy) + dz * dz);
          if (dist < dmin) dmin = dist;
        }
      }
      if (weights[rot] > 1e-3) d_min = (std::min)(d_min, dmin);
      if (dmin < 2.5) w_clash += weights[rot];
    }

    const std::vector<double> av_mean = take(av, &States::get_mean_position);
    const std::vector<double> ens_mean = take(ens, &States::get_mean_position);
    double dm[3] = {av_mean[0] - ens_mean[0], av_mean[1] - ens_mean[1], av_mean[2] - ens_mean[2]};

    AVRotamerPosition r;
    r.name = name;
    r.d_mean_position = nc::norm3(dm);
    r.av_n_points = av.get_n_points();
    r.n_rotamers = ens.get_n_rotamers();
    r.av_extent = extent(av);
    r.rot_extent = extent(ens);
    r.partition = ens.get_partition();
    r.interpenetration_weight = w_clash;
    r.min_heavy_atom_distance = d_min;
    positions.push_back(r);
    built.avs.push_back(av);
    built.ensembles.push_back(ens);
  }
}

std::string format(const char* fmt, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), fmt, v);
  return buf;
}

}  // namespace probe_model_comparison

AVRotamerCase av_rotamer_case(const std::string& system, int cutoff) {
  namespace pmc = probe_model_comparison;
  const std::string donor = "AlexaFluor 488 C1R cutoff" + std::to_string(cutoff);
  const std::string acceptor = "AlexaFluor 594 C1R cutoff" + std::to_string(cutoff);
  AVRotamerCase c;
  c.system = system;
  if (system == "hgbp1") {
    c.pdb = get_structure_dir("1DG3.pdb");
    const std::string path = get_example_path("structure/GBP/hGBP1.fps.json");
    const std::string text = pmc::read_file(path);
    const nlohmann::json fps = nlohmann::json::parse(text);
    const std::vector<std::string> order = pmc::KeyOrder(text).of("Positions");
    for (std::size_t i = 0; i < order.size(); ++i) {
      const nlohmann::json& p = fps["Positions"][order[i]];
      // residue 254 is not resolved in 1DG3; the other chain-A sites are
      if (!p.contains("chain_identifier") || p["chain_identifier"] != "A") continue;
      if (order[i].empty() || order[i][order[i].size() - 1] != 'F') continue;
      if (p["residue_seq_number"] == 254) continue;
      c.names.push_back(order[i]);
      c.positions_json.push_back(p.dump());
      c.libraries.push_back(p["residue_seq_number"] == 481 ? donor : acceptor);
    }
    std::vector<std::string> sorted = c.names;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      if (sorted[i] != "A481F") c.pairs.push_back(std::make_pair(std::string("A481F"), sorted[i]));
    }
    c.title = "hGBP1 1DG3 chain A \xe2\x80\x94 donor Alexa488 C1R at 481, acceptor Alexa594 C1R "
              "elsewhere (cutoff" + std::to_string(cutoff) + " libraries)";
  } else if (system == "t4l") {
    c.pdb = get_example_path("structure/T4L/3GUN.pdb");
    const std::string path = get_example_path("structure/T4L/fret.fps.json");
    const std::string text = pmc::read_file(path);
    const nlohmann::json fps = nlohmann::json::parse(text);
    const std::vector<std::string> order = pmc::KeyOrder(text).of("Positions");
    for (std::size_t i = 0; i < order.size(); ++i) {
      c.names.push_back(order[i]);
      c.positions_json.push_back(fps["Positions"][order[i]].dump());
      c.libraries.push_back(order[i][order[i].size() - 1] == 'D' ? donor : acceptor);
    }
    const std::vector<std::string> distances = pmc::KeyOrder(text).of("Distances");
    for (std::size_t i = 0; i < distances.size(); ++i) {
      const nlohmann::json& d = fps["Distances"][distances[i]];
      AVRotamerExperiment e;
      e.position1 = d["position1_name"].get<std::string>();
      e.position2 = d["position2_name"].get<std::string>();
      e.name = distances[i];
      e.distance = d["distance"].get<double>();
      e.error_neg = d["error_neg"].get<double>();
      e.error_pos = d["error_pos"].get<double>();
      e.distance_type = d.contains("distance_type") ? d["distance_type"].get<std::string>()
                                                    : std::string("RDAMean");
      c.pairs.push_back(std::make_pair(e.position1, e.position2));
      c.experiments.push_back(e);
    }
    c.title = "T4L 3GUN \xe2\x80\x94 the 99 fps.json distances (D = Alexa488 C1R, A = Alexa594 C1R, "
              "cutoff" + std::to_string(cutoff) + " libraries)";
  } else {
    IMP_THROW("unknown comparison system '" << system << "'; use hgbp1 or t4l", ValueException);
  }
  return c;
}

void compare_av_and_rotamer(const AVRotamerCase& c, int n_samples, double temperature,
                            double forster_radius, std::vector<AVRotamerPosition>& positions,
                            std::vector<AVRotamerPair>& pairs) {
  namespace pmc = probe_model_comparison;
  positions.clear();
  pairs.clear();
  pmc::Built built;
  pmc::build(c, temperature, positions, built);
  std::map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < c.names.size(); ++i) index[c.names[i]] = i;
  // experiments by pair; the last one recorded for a pair is the one used
  std::map<std::pair<std::string, std::string>, AVRotamerExperiment> experimental;
  for (std::size_t i = 0; i < c.experiments.size(); ++i) {
    experimental[std::make_pair(c.experiments[i].position1, c.experiments[i].position2)] =
            c.experiments[i];
  }
  for (std::size_t k = 0; k < c.pairs.size(); ++k) {
    const std::string& a = c.pairs[k].first;
    const std::string& b = c.pairs[k].second;
    if (!index.count(a) || !index.count(b)) {
      IMP_THROW("pair " << a << "-" << b << " names a position the case does not have",
                ValueException);
    }
    const std::size_t ia = index[a], ib = index[b];
    const std::vector<double> av_stats =
            av_pair_statistics(built.avs[ia], built.avs[ib], forster_radius, n_samples);
    double rot[5];
    pmc::ensemble_pair_statistics(built.ensembles[ia], built.ensembles[ib], forster_radius, rot);
    AVRotamerPair row;
    row.pair = a + "-" + b;
    row.p1 = a;
    row.p2 = b;
    row.Rmp_av = av_stats[0];
    row.Rmp_rot = rot[0];
    row.RDAMean_av = av_stats[1];
    row.RDAMean_rot = rot[1];
    row.RDAMeanE_av = av_stats[2];
    row.RDAMeanE_rot = rot[2];
    row.sigma_av = av_stats[3];
    row.sigma_rot = rot[3];
    row.kappa2_rot = rot[4];
    row.rotamer_valid = positions[ia].partition >= av_rotamer_z_cutoff() &&
                        positions[ib].partition >= av_rotamer_z_cutoff();
    std::map<std::pair<std::string, std::string>, AVRotamerExperiment>::const_iterator hit =
            experimental.find(c.pairs[k]);
    if (hit != experimental.end()) {
      const AVRotamerExperiment& e = hit->second;
      double model_av, model_rot;
      if (e.distance_type == "RDAMean") {
        model_av = row.RDAMean_av;
        model_rot = row.RDAMean_rot;
      } else if (e.distance_type == "RDAMeanE") {
        model_av = row.RDAMeanE_av;
        model_rot = row.RDAMeanE_rot;
      } else if (e.distance_type == "Rmp") {
        model_av = row.Rmp_av;
        model_rot = row.Rmp_rot;
      } else {
        IMP_THROW("distance type '" << e.distance_type << "' of " << row.pair
                                    << " is not RDAMean, RDAMeanE or Rmp",
                  ValueException);
      }
      row.has_experiment = true;
      row.exp_distance = e.distance;
      row.exp_type = e.distance_type;
      row.chi2_av = chi2_score(model_av, e.distance, e.error_neg, e.error_pos);
      row.chi2_rot = chi2_score(model_rot, e.distance, e.error_neg, e.error_pos);
    }
    pairs.push_back(row);
  }
}

std::vector<AVRotamerPosition> compare_av_and_rotamer_positions(const AVRotamerCase& c,
                                                                int n_samples,
                                                                double temperature) {
  (void)n_samples;
  std::vector<AVRotamerPosition> positions;
  probe_model_comparison::Built built;
  probe_model_comparison::build(c, temperature, positions, built);
  return positions;
}

std::vector<AVRotamerPair> compare_av_and_rotamer_pairs(const AVRotamerCase& c, int n_samples,
                                                        double temperature,
                                                        double forster_radius) {
  std::vector<AVRotamerPosition> positions;
  std::vector<AVRotamerPair> pairs;
  compare_av_and_rotamer(c, n_samples, temperature, forster_radius, positions, pairs);
  return pairs;
}

std::string av_rotamer_markdown_table(const std::vector<AVRotamerPosition>& positions,
                                      const std::vector<AVRotamerPair>& rows,
                                      const std::string& title) {
  namespace pmc = probe_model_comparison;
  namespace nc = internal::numpy_compat;
  using pmc::format;
  const double z = av_rotamer_z_cutoff();
  std::vector<std::string> lines;
  lines.push_back("### " + title);
  lines.push_back("");
  lines.push_back("| position | AV points | rotamers | Z | AV rms extent (\xc3\x85) | rot rms extent "
                  "(\xc3\x85) | \xce\x94 mean position (\xc3\x85) | weight within 2.5 \xc3\x85 of "
                  "protein | min heavy-atom distance, w > 1e-3 (\xc3\x85) |");
  lines.push_back("|---|---|---|---|---|---|---|---|---|");
  bool any_buried = false;
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const AVRotamerPosition& r = positions[i];
    const bool buried = r.partition < z;
    any_buried = any_buried || buried;
    lines.push_back("| " + r.name + (buried ? " \xe2\x80\xa1" : "") + " | " +
                    std::to_string(r.av_n_points) + " | " + std::to_string(r.n_rotamers) + " | " +
                    format("%.3f", r.partition) + " | " + format("%.1f", r.av_extent) + " | " +
                    format("%.1f", r.rot_extent) + " | " + format("%.1f", r.d_mean_position) +
                    " | " + format("%.2f", r.interpenetration_weight) + " | " +
                    format("%.2f", r.min_heavy_atom_distance) + " |");
  }
  if (any_buried) {
    lines.push_back("\n\xe2\x80\xa1 Z < " + nc::python_repr(z) +
                    ": every rotamer of the library clashes at this site (buried for the rotamer "
                    "model); the ensemble is FRETpredict's uniform fallback and its pairs are "
                    "excluded from \xce\xa3\xcf\x87\xc2\xb2.");
  }
  bool has_exp = false;
  for (std::size_t i = 0; i < rows.size(); ++i) has_exp = has_exp || rows[i].has_experiment;
  lines.push_back("");
  lines.push_back(std::string("| pair | R_mp AV / rot | \xe2\x9f\xa8R_DA\xe2\x9f\xa9 AV / rot | "
                              "\xe2\x9f\xa8R_DA\xe2\x9f\xa9_E AV / rot (\xce\xba\xc2\xb2=2/3) | "
                              "\xcf\x83_R AV / rot | \xe2\x9f\xa8\xce\xba\xc2\xb2\xe2\x9f\xa9 rot") +
                  (has_exp ? " | exp (type) | \xcf\x87\xc2\xb2 AV / rot |" : " |"));
  lines.push_back(std::string("|---|---|---|---|---|---") + (has_exp ? "|---|---|" : "|"));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const AVRotamerPair& r = rows[i];
    std::string line = "| " + r.pair + (r.rotamer_valid ? "" : " \xe2\x80\xa1") + " | " +
                       format("%.1f", r.Rmp_av) + " / " + format("%.1f", r.Rmp_rot) + " | " +
                       format("%.1f", r.RDAMean_av) + " / " + format("%.1f", r.RDAMean_rot) +
                       " | " + format("%.1f", r.RDAMeanE_av) + " / " +
                       format("%.1f", r.RDAMeanE_rot) + " | " + format("%.1f", r.sigma_av) + " / " +
                       format("%.1f", r.sigma_rot) + " | " + format("%.2f", r.kappa2_rot);
    if (has_exp) {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      line += " | " + format("%.1f", r.has_experiment ? r.exp_distance : nan) + " (" +
              (r.has_experiment ? r.exp_type : std::string()) + ") | " +
              format("%.2f", r.has_experiment ? r.chi2_av : nan) + " / " +
              format("%.2f", r.has_experiment ? r.chi2_rot : nan) + " |";
    } else {
      line += " |";
    }
    lines.push_back(line);
  }
  if (has_exp) {
    std::vector<double> chi_av, chi_rot, chi_av_all;
    std::size_t n_valid = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const double av = rows[i].has_experiment ? rows[i].chi2_av : 0.0;
      const double rot = rows[i].has_experiment ? rows[i].chi2_rot : 0.0;
      if (rows[i].rotamer_valid) {
        ++n_valid;
        chi_av.push_back(av);
        chi_rot.push_back(rot);
      }
      chi_av_all.push_back(av);
    }
    lines.push_back("");
    lines.push_back("\xce\xa3\xcf\x87\xc2\xb2 over the " + std::to_string(n_valid) +
                    " pairs with valid ensembles: AV " + format("%.2f", nc::python_sum(chi_av)) +
                    ", rotamer " + format("%.2f", nc::python_sum(chi_rot)) + " (AV over all " +
                    std::to_string(rows.size()) + " pairs: " +
                    format("%.2f", nc::python_sum(chi_av_all)) + ").");
  }
  std::string out;
  for (std::size_t i = 0; i < lines.size(); ++i) out += (i ? "\n" : "") + lines[i];
  return out + "\n";
}

std::string av_rotamer_summary_json(const std::vector<AVRotamerPosition>& positions,
                                    const std::vector<AVRotamerPair>& rows, int indent,
                                    int depth) {
  namespace nc = internal::numpy_compat;
  typedef nc::PyJson J;
  J pos = J::object();
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const AVRotamerPosition& r = positions[i];
    J entry = J::object();
    entry.set("d_mean_position", J::number(r.d_mean_position));
    entry.set("av_n_points", J::integer(r.av_n_points));
    entry.set("n_rotamers", J::integer(r.n_rotamers));
    entry.set("av_extent", J::number(r.av_extent));
    entry.set("rot_extent", J::number(r.rot_extent));
    entry.set("partition", J::number(r.partition));
    entry.set("interpenetration_weight_2.5A", J::number(r.interpenetration_weight));
    entry.set("min_heavy_atom_distance", J::number(r.min_heavy_atom_distance));
    pos.set(r.name, entry);
  }
  J list = J::array();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const AVRotamerPair& r = rows[i];
    J row = J::object();
    row.set("pair", J::string(r.pair));
    row.set("p1", J::string(r.p1));
    row.set("p2", J::string(r.p2));
    row.set("Rmp_av", J::number(r.Rmp_av));
    row.set("Rmp_rot", J::number(r.Rmp_rot));
    row.set("RDAMean_av", J::number(r.RDAMean_av));
    row.set("RDAMean_rot", J::number(r.RDAMean_rot));
    row.set("RDAMeanE_av", J::number(r.RDAMeanE_av));
    row.set("RDAMeanE_rot", J::number(r.RDAMeanE_rot));
    row.set("sigma_av", J::number(r.sigma_av));
    row.set("sigma_rot", J::number(r.sigma_rot));
    row.set("kappa2_rot", J::number(r.kappa2_rot));
    row.set("rotamer_valid", J::boolean(r.rotamer_valid));
    if (r.has_experiment) {
      row.set("exp_distance", J::number(r.exp_distance));
      row.set("exp_type", J::string(r.exp_type));
      row.set("chi2_av", J::number(r.chi2_av));
      row.set("chi2_rot", J::number(r.chi2_rot));
    }
    list.push(row);
  }
  J doc = J::object();
  doc.set("positions", pos);
  doc.set("pairs", list);
  return doc.dumps(indent, depth);
}

IMPBFF_END_NAMESPACE

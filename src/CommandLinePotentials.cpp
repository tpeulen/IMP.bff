/** \file CommandLinePotentials.cpp
 *  \brief `imp_bff potentials2pto`: build `data/potentials.pto` from the loose
 *         `.npy` tables ChiSurf ships.
 *
 *  Port of `bin/imp_bff_potentials2pto`. The conversions are the library's
 *  (`convert_*_potential` in ProbePotentialTables.h, the `.npy` reader in
 *  internal/Npy.h); this file is the grammar and the manifest -- the part
 *  four loose `.npy` files cannot carry: what the conversion did to the
 *  numbers, and from which bytes.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include <IMP/bff/ProbePotentialTables.h>
#include <IMP/bff/internal/Npy.h>
#include <IMP/bff/internal/Sha256.h>
#include <IMP/bff/internal/Text.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace potentials {

//! A JSON value kept in insertion order, printed as `json.dumps(indent=1)`.
struct Json {
  enum Kind { SCALAR, OBJECT, ARRAY } kind;
  std::string text;                                   //!< a spelled scalar
  std::vector<std::string> keys;                       //!< OBJECT
  std::vector<Json> items;                             //!< ARRAY values, or OBJECT values

  Json() : kind(OBJECT) {}
  static Json raw(const std::string& spelled) {
    Json j;
    j.kind = SCALAR;
    j.text = spelled;
    return j;
  }
  static Json str(const std::string& s) { return raw(json_string(s)); }
  static Json num(double v) { return raw(json_float(v)); }
  static Json integer(long v) { return raw(format("%ld", v)); }
  static Json array() {
    Json j;
    j.kind = ARRAY;
    return j;
  }
  Json& set(const std::string& key, const Json& value) {
    keys.push_back(key);
    items.push_back(value);
    return *this;
  }
  Json& push(const Json& value) {
    items.push_back(value);
    return *this;
  }

  std::string dump(int depth = 0) const {
    const std::string pad(static_cast<std::size_t>(depth + 1), ' ');
    const std::string close(static_cast<std::size_t>(depth), ' ');
    if (kind == SCALAR) return text;
    if (kind == ARRAY) {
      if (items.empty()) return "[]";
      std::string out = "[\n" + pad;
      for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += ",\n" + pad;
        out += items[i].dump(depth + 1);
      }
      return out + "\n" + close + "]";
    }
    if (keys.empty()) return "{}";
    std::string out = "{\n" + pad;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (i) out += ",\n" + pad;
      out += json_string(keys[i]) + ": " + items[i].dump(depth + 1);
    }
    return out + "\n" + close + "}";
  }
};

Json strings(const std::vector<std::string>& names) {
  Json a = Json::array();
  for (std::size_t i = 0; i < names.size(); ++i) a.push(Json::str(names[i]));
  return a;
}

Json integers(const std::vector<int>& values) {
  Json a = Json::array();
  for (std::size_t i = 0; i < values.size(); ++i) a.push(Json::integer(values[i]));
  return a;
}

//! The source array, with the dimensionality the conversion needs.
NpyArray load(const std::string& path, std::size_t ndim) {
  NpyArray a;
  try {
    a = read_npy(path);
  } catch (const std::exception& e) {
    throw SubError(e.what());
  }
  if (a.shape.size() != ndim) {
    throw SubError(format("%s: %lu dimensions, expected %lu", path.c_str(),
                          (unsigned long)a.shape.size(), (unsigned long)ndim));
  }
  return a;
}

std::string sha256_of(const std::string& path) { return sha256_hex(read_text_file(path)); }

struct Args {
  std::string database, output;
};

void run(const Args& a) {
  const std::string output = a.output.empty() ? get_potential_container_path() : a.output;
  const char* const names[] = {"mj", "hb", "unres", "rama_ala_pro_gly"};
  std::vector<std::string> sources;
  for (int i = 0; i < 4; ++i) {
    sources.push_back(path_join(a.database, std::string(names[i]) + ".npy"));
    if (!file_exists(sources.back())) throw SubExit(1, "missing: " + sources.back());
  }
  const std::string& mj_path = sources[0];
  const std::string& hb_path = sources[1];
  const std::string& unres_path = sources[2];
  const std::string& rama_path = sources[3];

  const double mj_cutoff = 6.5;
  const NpyArray mj = load(mj_path, 2);
  const PotentialConversion mj_table = convert_mj_potential(mj.values, mj_cutoff);

  const double unres_bin = 0.05, unres_min = 3.5, unres_repulsion = 100.0;
  const NpyArray unres = load(unres_path, 3);
  const int unres_bins = static_cast<int>(unres.shape[2]);
  const PotentialConversion unres_table =
      convert_unres_potential(unres.values, unres_bins, unres_bin, unres_min, unres_repulsion);

  const double hb_bin = 0.01, hb_physical = 1.3;
  const NpyArray hb = load(hb_path, 2);
  const PotentialConversion hb_table = convert_hbond_potential(
      hb.values, static_cast<int>(hb.shape[0]), static_cast<int>(hb.shape[1]), hb_bin, hb_physical);

  const int rama_bins = 360;
  const NpyArray rama = load(rama_path, 2);
  const PotentialConversion rama_table =
      convert_ramachandran_potential(rama.values, static_cast<int>(rama.shape[0]), rama_bins);

  const std::vector<std::string> residues = potential_residue_order();

  Json mj_meta;
  mj_meta.set("kind", Json::str("pot.pmf"))
      .set("bin_width", Json::num(mj_cutoff / 2.0))
      .set("n_bins", Json::integer(2))
      .set("cutoff_A", Json::num(mj_cutoff))
      .set("units", Json::str("kT"))
      .set("residues", strings(residues))
      .set("note", Json::str("a contact potential: flat to the cutoff, zero past it"))
      .set("source", Json::str("mj.npy"))
      .set("source_sha256", Json::str(sha256_of(mj_path)));

  Json unres_meta;
  unres_meta.set("kind", Json::str("pot.pmf"))
      .set("bin_width", Json::num(unres_bin))
      .set("n_bins", Json::integer(unres_bins))
      .set("residues", strings(residues))
      .set("max_A", Json::num(unres_bin * unres_bins))
      .set("min_dist_A", Json::num(unres_min))
      .set("repulsion", Json::num(unres_repulsion))
      .set("units", Json::str("kcal/mol"))
      .set("triangle", Json::str("upper"))
      .set("note",
           Json::str(format("%d NaNs in bin 0 and every bin below %s A hold the repulsion; the "
                            "source is filled in its upper triangle only and that half is what is "
                            "written, where the Python indexed it by the residues' order in the "
                            "chain and so read the empty half for about one contact in two; "
                            "multiply the restraint's weight by 0.593 for kT at 298 K",
                            unres_table.n_changed, json_float(unres_min).c_str())))
      .set("source", Json::str("unres.npy"))
      .set("source_sha256", Json::str(sha256_of(unres_path)));

  std::vector<std::string> hb_channels;
  hb_channels.push_back("CH");
  hb_channels.push_back("ON");
  hb_channels.push_back("OH");
  hb_channels.push_back("CN");
  Json hb_meta;
  hb_meta.set("kind", Json::str("pot.grid"))
      .set("shape", integers(hb_table.table.shape))
      .set("bin_width", Json::num(hb_bin))
      .set("channels", strings(hb_channels))
      .set("units", Json::str("kT"))
      .set("note", Json::str(format("%d values above 1e5 below %s A were held at the value there; "
                                    "no two atoms are that close",
                                    hb_table.n_changed, json_float(hb_physical).c_str())))
      .set("source", Json::str("hb.npy"))
      .set("source_sha256", Json::str(sha256_of(hb_path)));

  std::vector<std::string> rama_channels;
  rama_channels.push_back("general");
  rama_channels.push_back("PRO");
  rama_channels.push_back("GLY");
  Json rama_meta;
  rama_meta.set("kind", Json::str("pot.grid"))
      .set("shape", integers(rama_table.table.shape))
      .set("channels", strings(rama_channels))
      .set("resolution_deg", Json::num(360.0 / rama_bins))
      .set("units", Json::str("log(P/Pmax)"))
      .set("axes", Json::str("phi along axis 0, psi along axis 1, both from -180 deg"))
      .set("note",
           Json::str(format("channels 0 and 1 of the source are the phi and psi coordinate grids, "
                            "not maps, and are dropped; %d cells held a sentinel 1.0 (proline, "
                            "where the map has no data) and were set to the least probable "
                            "measured value of their channel, so every cell is a log-ratio and "
                            "the reader negates the map as a whole",
                            rama_table.n_changed)))
      .set("source", Json::str("rama_ala_pro_gly.npy"))
      .set("source_sha256", Json::str(sha256_of(rama_path)));

  Json tables;
  tables.set("mj", mj_meta).set("unres", unres_meta).set("hbond", hb_meta).set("ramachandran",
                                                                              rama_meta);
  Json manifest;
  manifest.set("doctype", Json::str("imp.bff potential tables"))
      .set("version", Json::integer(1))
      .set("made_by", Json::str("imp_bff potentials2pto"))
      .set("tables", tables);

  std::vector<PotentialTable> out;
  out.push_back(mj_table.table);
  out.push_back(unres_table.table);
  out.push_back(hb_table.table);
  out.push_back(rama_table.table);
  write_potential_tables(output, out, manifest.dump());

  struct stat st;
  const double size = stat(output.c_str(), &st) == 0 ? static_cast<double>(st.st_size) : 0.0;
  std::cout << "wrote " << output << format(" (%.2f MB)", size / 1e6) << "\n";
  const std::vector<std::string> written = potential_table_names(output);
  for (std::size_t i = 0; i < written.size(); ++i) std::cout << "  " << written[i] << "\n";
}

}  // namespace potentials

void add_potentials_subs(CLI::App& app) {
  std::shared_ptr<potentials::Args> a = std::make_shared<potentials::Args>();
  CLI::App* sub = app.add_subcommand("potentials2pto", R"doc(Write one PTO container from the four `.npy` tables.)doc");
  sub->footer(R"doc(One container carries every parameter table the coarse-grained potentials read.
Two of them -- Miyazawa-Jernigan and UNRES -- go in as **text in IMP's PMF
format**, because `IMP.core.StatisticalPairScore` reads that directly and the
potential is then the file rather than a loop; the other two go in as float64
grids, because their restraints are not pair scores over a distance.

The conversion is not a copy. What it does to the numbers, and why, is written
into the container's manifest, which is the part four loose `.npy` files cannot
carry:

* `unres.npy` has six NaNs, all in bin 0 (0 to 0.05 A). They become the
  short-range repulsion, which is what that bin means.
* `unres.npy` starts at zero distance, but the potential it belongs to applies
  a flat penalty below `min_dist`. That penalty is written into the low bins,
  so the branch is in the data rather than at evaluation time.
* `hb.npy` reaches 7e33 in its first 127 bins -- a divergence below 1.27 A,
  where no two atoms are. Those bins are held at the value the table has at
  1.3 A.
* `rama_ala_pro_gly.npy` has five channels, of which the first two are the phi
  and psi *coordinate* grids and only the last three are maps. The container
  keeps the three.

Run it against a ChiSurf checkout:

    imp_bff potentials2pto --database ../chisurf/chisurf/core/structure/potential/database)doc");
  sub->add_option("--database", a->database, "ChiSurf's structure/potential/database directory")
      ->required()
      ->check(CLI::ExistingDirectory);
  sub->add_option("--output", a->output, "the container to write; the shipped path by default");
  sub->callback([a] {
    set_current_sub("potentials2pto");
    potentials::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

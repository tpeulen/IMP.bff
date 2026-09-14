/** \file CommandLineLabelizer.cpp
 *  \brief `imp_bff labelizer`: score the labelling sites of a structure, and
 *         write one `.mmfdb.pto`.
 *
 *  Port of `bin/imp_bff_labelizer`. A native port of the Labelizer label-site
 *  score (Gebhardt *et al.*, *Nat. Commun.* **16**, 3305, 2025): every residue
 *  gets a per-parameter score and a combined label score; optionally every
 *  pair of labelable sites gets a FRET pair score. The kernels are
 *  LabelizerScore.h, LabelizerFRET.h and LabelizerIO.h; this file is the
 *  grammar, the R0 choice and the `--show` summary.
 *
 *  **The published arithmetic is the default, defects included**, because the
 *  paper's numbers were computed with them; `--corrected` selects the
 *  arithmetic the reference documents. Conservation is **imported, never
 *  computed** (a ConSurf `.grades` table or a B-factor PDB).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <IMP/bff/LabelizerFRET.h>
#include <IMP/bff/LabelizerIO.h>
#include <IMP/bff/LabelizerScore.h>
#include <IMP/bff/ProbeLibrary.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace labelizer {

struct Args {
  std::string src, out, conservation, donor, acceptor, dyes, second_structure;
  std::string donor_chain, acceptor_chain, extract_structure;
  bool no_conservation = false, corrected = false, pairs = false, show = false;
  double r0 = 52.0, kappa2 = 2.0 / 3.0, refractive_index = 1.4, threshold = 0.5;
  int refine = 0, max_pairs = 0;
};

long file_size(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 ? static_cast<long>(st.st_size) : 0L;
}

bool by_value_desc(const LabelizerScore& a, const LabelizerScore& b) {
  return a.value > b.value;
}

//! Print what a container holds, without unpacking it.
void show(const std::string& path) {
  if (path_suffix(path) != ".pto") {
    const std::string stem = path_with_suffix(basename_of(path), "");
    throw SubExit(2, "--show reads a container; " + path +
                         " is not one. Score the structure first, which writes " + stem +
                         ".mmfdb.pto beside it.");
  }
  const std::vector<LabelizerScore> scores = labelizer_read_pto_scores(path);
  const std::vector<LabelizerFRETPairScore> pairs = labelizer_read_pto_pairs(path);
  std::string settings_text = labelizer_read_pto_settings(path);
  if (settings_text.empty()) settings_text = "{}";
  const nlohmann::json settings = nlohmann::json::parse(settings_text);

  std::map<std::string, std::vector<LabelizerScore> > by_type;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    by_type[scores[i].score_type].push_back(scores[i]);
  }

  std::cout << path << "\n";
  std::cout << "  arithmetic     "
            << (settings.contains("arithmetic") ? settings["arithmetic"].get<std::string>()
                                                : std::string("unknown"))
            << "\n";
  std::string terms;
  if (settings.contains("model")) {
    const nlohmann::json& model = settings["model"];
    for (std::size_t i = 0; i < model.size(); ++i) {
      if (i) terms += ", ";
      terms += format("%s:%d", model[i]["tag"].get<std::string>().c_str(),
                      model[i]["weight"].get<int>());
    }
  }
  std::cout << "  model          " << terms << "\n";
  std::cout << "  score rows     " << scores.size() << "\n";
  for (std::map<std::string, std::vector<LabelizerScore> >::const_iterator it = by_type.begin();
       it != by_type.end(); ++it) {
    std::size_t scored = 0;
    for (std::size_t i = 0; i < it->second.size(); ++i) {
      if (it->second[i].status == "scored") ++scored;
    }
    std::cout << format("    %-22s %5lu scored, %4lu not", it->first.c_str(),
                        (unsigned long)scored, (unsigned long)(it->second.size() - scored))
              << "\n";
  }
  std::cout << "  pair rows      " << pairs.size() << "\n";

  std::vector<LabelizerScore> combined;
  if (by_type.count("combined")) {
    const std::vector<LabelizerScore>& rows = by_type["combined"];
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (rows[i].status == "scored") combined.push_back(rows[i]);
    }
  }
  if (!combined.empty()) {
    std::stable_sort(combined.begin(), combined.end(), by_value_desc);
    std::cout << "  best sites\n";
    for (std::size_t i = 0; i < combined.size() && i < 10; ++i) {
      const LabelizerScore& r = combined[i];
      std::cout << format("    %s%-5d %s  %.4f", r.asym_id.c_str(), r.seq_id,
                          r.comp_id.c_str(), r.value)
                << "\n";
    }
  }
  if (!pairs.empty()) {
    std::cout << "  best pairs\n";
    for (std::size_t i = 0; i < pairs.size() && i < 10; ++i) {
      const LabelizerFRETPairScore& p = pairs[i];
      std::cout << format("    %s%d-%s%-6d %.4f  d=%.1f A", p.asym_id_1.c_str(), p.seq_id_1,
                          p.asym_id_2.c_str(), p.seq_id_2, p.value, p.distance)
                << "\n";
    }
  }
}

void run(const Args& a) {
  if (!file_exists(a.src)) throw SubExit(1, a.src + " does not exist");

  if (!a.extract_structure.empty()) {
    const std::string digest = labelizer_extract_pto_structure(a.src, a.extract_structure);
    std::cout << a.extract_structure << "  sha256 " << digest << "  verified\n";
    return;
  }
  if (a.show) {
    show(a.src);
    return;
  }

  // the published model, minus the conservation term when it is dropped
  std::vector<LabelizerParameter> model;
  const std::vector<LabelizerParameter> paper = labelizer_model_paper();
  for (std::size_t i = 0; i < paper.size(); ++i) {
    if (a.no_conservation && paper[i].tag == "cs") continue;
    model.push_back(paper[i]);
  }

  LabelizerOptions options;
  options.model = a.corrected ? LABELIZER_MODEL_CORRECTED : LABELIZER_MODEL_PUBLISHED;

  const std::string conservation = a.conservation;
  if (conservation.empty() && !a.no_conservation) {
    std::cerr << sub_prefix()
              << "no --conservation given, so the conservation term is unavailable and "
                 "the combined score will be too. Pass --no-conservation to score without it.\n";
  }

  const std::vector<LabelizerScore> scores =
      labelizer_score_structure(a.src, model, options, conservation);

  LabelizerFRETOptions fret;
  fret.model = options.model;
  fret.forster_radius = a.r0;

  // A Forster radius is derived from two spectra and a quantum yield, not
  // supplied. When the dyes are named, say so in the output: an R0 from a
  // number on a command line and one from measured spectra are different
  // claims.
  if (a.donor.empty() != a.acceptor.empty()) {
    throw SubExit(1, "--donor and --acceptor go together");
  }
  if (!a.donor.empty()) {
    std::string whence;
    if (!a.dyes.empty()) {
      fret.forster_radius = probe_pto_forster_radius(a.dyes, a.donor, a.acceptor, a.kappa2,
                                                     a.refractive_index);
      whence = a.dyes;
    } else {
      fret.forster_radius =
          forster_radius(get_probe(a.donor), get_probe(a.acceptor), a.kappa2, a.refractive_index);
      whence = "the bundled dye library";
    }
    std::cout << format("R0(%s -> %s) = %.1f A  [kappa2=%.3f, n=%.2f, from %s]",
                        a.donor.c_str(), a.acceptor.c_str(), fret.forster_radius, a.kappa2,
                        a.refractive_index, whence.c_str())
              << "\n";
  }
  fret.label_score_threshold = a.threshold;
  fret.n_refine = a.refine;
  fret.donor_chain = a.donor_chain;
  fret.acceptor_chain = a.acceptor_chain;
  if (a.donor_chain.empty() != a.acceptor_chain.empty()) {
    throw SubExit(1, "--donor-chain and --acceptor-chain go together");
  }

  std::vector<LabelizerFRETPairScore> pairs;
  if (a.pairs || !a.second_structure.empty()) {
    const std::map<std::string, double> label_scores = labelizer_combined_by_key(scores);
    if (label_scores.empty()) {
      std::cerr << sub_prefix()
                << "no position has a combined score, so there is nothing to pair\n";
    } else if (!a.second_structure.empty()) {
      if (!file_exists(a.second_structure)) {
        throw SubExit(1, a.second_structure + " does not exist");
      }
      // scored with the same model, so the four label scores of a
      // two-state pair are comparable
      const std::vector<LabelizerScore> scores_2 =
          labelizer_score_structure(a.second_structure, model, options, conservation);
      const std::map<std::string, double> label_scores_2 = labelizer_combined_by_key(scores_2);
      if (label_scores_2.empty()) {
        throw SubExit(1, "no position of " + a.second_structure + " has a combined score");
      }
      pairs = labelizer_pair_scores_two_states(a.src, a.second_structure, label_scores,
                                               label_scores_2, fret);
    } else {
      pairs = labelizer_fret_pair_scores(a.src, label_scores, fret);
    }
    if (a.max_pairs > 0 && pairs.size() > static_cast<std::size_t>(a.max_pairs)) {
      pairs.resize(static_cast<std::size_t>(a.max_pairs));
    }
  }

  const std::string out =
      a.out.empty() ? path_with_suffix(path_with_suffix(a.src, ""), ".mmfdb.pto") : a.out;
  const std::string settings = labelizer_settings_json(model, options, fret, conservation);
  labelizer_write_pto(out, a.src, scores, pairs, settings);

  std::size_t scored = 0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (scores[i].score_type == "combined" && scores[i].status == "scored") ++scored;
  }
  std::cout << out
            << format("  %lu sites scored, %lu pairs, %ld bytes", (unsigned long)scored,
                      (unsigned long)pairs.size(), file_size(out))
            << "\n";
}

}  // namespace labelizer

void add_labelizer_subs(CLI::App& app) {
  std::shared_ptr<labelizer::Args> a = std::make_shared<labelizer::Args>();
  CLI::App* sub = app.add_subcommand(
      "labelizer", "Score the labelling sites of a structure, and write one .mmfdb.pto.");
  sub->footer(
      "Examples:\n"
      "  imp_bff labelizer structure.pdb                      # -> structure.mmfdb.pto\n"
      "  imp_bff labelizer structure.pdb --conservation grades.txt\n"
      "  imp_bff labelizer structure.pdb --pairs --refine 300 --r0 52\n"
      "  imp_bff labelizer closed.pdb --second-structure open.pdb \\\n"
      "      --donor-chain A --acceptor-chain B --donor Alexa555 --acceptor Alexa647\n"
      "  imp_bff labelizer scored.mmfdb.pto --show\n"
      "  imp_bff labelizer scored.mmfdb.pto --extract-structure recovered.pdb\n\n"
      "Conservation is imported, never computed. Without it the conservation term is\n"
      "unavailable for every position and, under the published model, the combined\n"
      "score is unavailable with it; --no-conservation drops the term instead.");
  sub->add_option("src", a->src, "a structure to score, or a container to inspect")->required();
  sub->add_option("-o,--out", a->out,
                  "the container to write; defaults to <stem>.mmfdb.pto beside the structure");
  sub->add_option("--conservation", a->conservation,
                  "a ConSurf .grades table, or a PDB carrying the normalised grade in its "
                  "B-factor column");
  sub->add_flag("--no-conservation", a->no_conservation,
                "drop the conservation term from the model, rather than leaving it "
                "unavailable and taking the whole combined score down with it");
  sub->add_flag("--corrected", a->corrected,
                "use the arithmetic the reference documents instead of the arithmetic it runs");
  sub->add_flag("--pairs", a->pairs, "also score every pair of labelable sites");
  sub->add_option("--r0,--forster-radius", a->r0,
                  "Forster radius for the pair score, A. Prefer --donor/--acceptor, which "
                  "derive it.")
      ->capture_default_str();
  sub->add_option("--donor", a->donor,
                  "derive the Forster radius from this donor and --acceptor rather than taking "
                  "--r0 on trust. Names resolve loosely: Alexa488, Cy3, Atto647N all work");
  sub->add_option("--acceptor", a->acceptor, "the acceptor dye; see --donor");
  sub->add_option("--dyes", a->dyes,
                  "a dye container (.mmfdb.pto) to take the spectra from; defaults to the "
                  "library bundled with IMP.bff");
  sub->add_option("--kappa2", a->kappa2,
                  "orientation factor for the derived R0. The default is the isotropic average "
                  "and is an assumption, not a measurement")
      ->capture_default_str();
  sub->add_option("--refractive-index", a->refractive_index,
                  "of the medium between the dyes")
      ->capture_default_str();
  sub->add_option("--refine", a->refine,
                  "recompute the top N pairs with real accessible volumes instead of the "
                  "analytic cone")
      ->capture_default_str();
  sub->add_option("--threshold", a->threshold,
                  "sites below this combined score are not paired (the reference's "
                  "LS_THRESHOLD)")
      ->capture_default_str();
  sub->add_option("--max-pairs", a->max_pairs,
                  "keep only the best N pairs in the container; 0 keeps all of them")
      ->capture_default_str();
  sub->add_flag("--show", a->show, "print what a container holds and exit");
  sub->add_option("--second-structure", a->second_structure,
                  "a second conformation. The pair score then becomes the CHANGE in transfer "
                  "efficiency between the two. Implies --pairs");
  sub->add_option("--donor-chain", a->donor_chain,
                  "restrict pairs to a donor in this chain. With --acceptor-chain this is what "
                  "makes a homodimer screen");
  sub->add_option("--acceptor-chain", a->acceptor_chain,
                  "restrict pairs to an acceptor in this chain");
  sub->add_option("--extract-structure", a->extract_structure,
                  "recover the embedded structure from a container, verifying it against its "
                  "recorded SHA-256, and exit");
  sub->callback([a] {
    set_current_sub("labelizer");
    labelizer::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

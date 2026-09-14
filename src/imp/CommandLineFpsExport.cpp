/** \file CommandLineFpsExport.cpp
 *  \brief `imp_bff fps-export ...`: FPS's exports, and the error estimation
 *         that fills them.
 *
 *  Port of `bin/imp_bff_fps_export`. The engines are `FPSExport.h` and the
 *  bootstrap in `Docking.h`; this file is the grammar, `results.json` (the
 *  manifest FPS's `SimulationResults.bin` needed and never had) and its
 *  reader. Help texts are the Python program's docstrings, verbatim.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/log.h>

#include <IMP/bff/Docking.h>
#include <IMP/bff/FPSExport.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace fps_export {

//! The labelling file to read, from either spelling.
std::string labelling_file(const std::string& fps_json, const std::string& positions) {
  if (!fps_json.empty() && !positions.empty()) {
    throw CLI::ValidationError("give either --fps-json or --positions, not both");
  }
  if (fps_json.empty() && positions.empty()) {
    throw CLI::ValidationError(
        "no labelling file: pass --fps-json, or --positions for the legacy C# FPS format");
  }
  return fps_json.empty() ? positions : fps_json;
}

FPSExportOptions options(const std::string& prefix, bool best_fit, bool save_pdb, bool fps_camera,
                         const FPSLabelPositions* labels) {
  FPSExportOptions o;
  o.prefix = prefix;
  o.best_fit = best_fit;
  o.save_pdb = save_pdb;
  o.camera_zero = !fps_camera;
  if (labels) o.labels = *labels;
  return o;
}

void write_all(const std::string& directory, const FPSResultTable& table,
               const FPSMolecules& molecules, const FPSExportOptions& opts,
               const std::string& fps_json) {
  std::vector<std::string> written = write_fps_exports(directory, table, molecules, opts);
  const std::string results = path_join(directory, "results.json");
  // Python re-reads get_json() (sorted keys, as nlohmann writes them) and
  // appends the two manifest keys; this spells the same file.
  nlohmann::json payload = nlohmann::json::parse(table.get_json());
  std::string text = json_dump_python(payload, 1);
  std::vector<std::pair<std::string, std::string> > appended;
  if (!fps_json.empty()) appended.push_back(std::make_pair("fps_json", json_string(path_abs(fps_json))));
  // The manifest FPS's `SimulationResults.bin` needed and never had: it
  // re-attached the open molecules **by position** with no check that they
  // were the same ones.
  std::string list = "[";
  for (std::size_t i = 0; i < molecules.size(); ++i) {
    const FPSMolecule& m = molecules[i];
    list += std::string(i ? "," : "") + "\n  {\n   \"path\": " + json_string(m.path) +
            ",\n   \"name\": " + json_string(m.name) + ",\n   \"n_atoms\": " +
            format("%d", m.n_atoms) + ",\n   \"center\": [\n    " + json_float(m.center[0]) +
            ",\n    " + json_float(m.center[1]) + ",\n    " + json_float(m.center[2]) +
            "\n   ]\n  }";
  }
  list += molecules.empty() ? "]" : "\n ]";
  appended.push_back(std::make_pair("molecules", list));
  // splice before the closing brace of the top-level object
  if (text == "{}") text = "{\n}";
  text.erase(text.size() - 2);  // "\n}"
  for (std::size_t i = 0; i < appended.size(); ++i) {
    text += (text == "{" ? "\n " : ",\n ") + json_string(appended[i].first) + ": " +
            appended[i].second;
  }
  text += "\n}";
  std::ofstream out(results.c_str());
  if (!out) throw SubError(results + ": cannot write");
  out << text;
  out.close();
  written.push_back(results);
  for (std::size_t i = 0; i < written.size(); ++i) std::cout << "wrote " << written[i] << "\n";
}

// ---- errors ------------------------------------------------------------------

struct ErrorsArgs {
  std::vector<std::string> pdb;
  std::string fps_json, positions, score_set, output_dir, perturbation = "split-normal",
                                                          parent_poses, prefix = "structure";
  int replicas = 10, seed = 1, steps = 200;
  double ev_weight = 0, clash_tolerance = 0;
  bool fps_d2 = false, keep_replicas = false, fps_rmsd_sign = false, best_fit = false,
       save_pdb = false, labels = false, fps_camera = false;
  CLI::Option* ev_opt = nullptr;
  CLI::Option* clash_opt = nullptr;
};

void run_errors(const ErrorsArgs& a) {
  const std::string labelling = labelling_file(a.fps_json, a.positions);
  DockingParameters params = fps_error_estimation_parameters();
  params.score_set = a.score_set;
  params.n_frames = a.steps;
  if (a.ev_opt->count()) params.ev_weight = a.ev_weight;
  if (a.clash_opt->count()) params.clash_tolerance = a.clash_tolerance;
  BootstrapParameters boot;
  boot.n_replicas = a.replicas;
  boot.seed = static_cast<unsigned int>(a.seed);
  boot.perturbation = a.perturbation == "fps-sign-split" ? FPS_SIGN_SPLIT_NORMAL : BFF_SPLIT_NORMAL;
  boot.perturb_deselected = !a.fps_d2;
  boot.keep_replica_dirs = a.keep_replicas;

  std::string poses;
  if (!a.parent_poses.empty()) poses = trimmed(read_text_file(a.parent_poses));

  const BootstrapResult result = fps_bootstrap(a.pdb, labelling, a.output_dir, poses, params, boot);

  std::cout << format("parent score      %.4f  (excluded volume %.4f)", result.parent_score,
                      result.parent_e_clash)
            << "\n";
  std::cout << format("replicas          %lu", (unsigned long)result.replicas.size()) << "\n";
  std::cout << format("perturbed / pinned %d / %d", result.n_perturbed, result.n_pinned) << "\n";
  if (result.n_pinned) {
    std::cout << std::flush;
    std::cerr << format("  %d deselected distance(s) carry no noise and pin the replicas to the "
                        "parent: the spread below is a lower bound (FPS's D2).",
                        result.n_pinned)
              << "\n";
  }
  std::cout << format("RMSD vs parent    %.3f +- %.3f A (max %.3f)", result.rmsd_mean,
                      result.rmsd_sd, result.rmsd_max)
            << "\n";
  // A spread of zero says something about the *restraint set*, not the FRET
  // network, whenever the excluded volume carries most of the score.
  if (result.parent_score > 0 && result.parent_e_clash > 0.5 * result.parent_score &&
      result.rmsd_mean < 0.05) {
    std::cout << std::flush;
    std::cerr << format("  the excluded volume is %.0f %% of the parent score and the spread is "
                        "~0: the pose is held by the clash term, not by the data. Lower "
                        "--ev-weight or --clash-tolerance to see what the FRET network alone "
                        "constrains.",
                        100.0 * result.parent_e_clash / result.parent_score)
              << "\n";
  }

  const DockingAssembly assembly = create_docking_assembly(a.pdb, labelling);
  FPSResultTable table = fps_bootstrap_table(result, static_cast<int>(a.pdb.size()));
  table = add_rmsd_columns(table, assembly, a.fps_rmsd_sign);
  if (a.best_fit) table = add_best_fit(table, assembly);
  FPSLabelPositions label_positions;
  if (a.labels) label_positions = fps_label_positions(assembly);
  const FPSExportOptions opts =
      options(a.prefix, a.best_fit, a.save_pdb, a.fps_camera, a.labels ? &label_positions : nullptr);
  write_all(a.output_dir, table, fps_molecules(a.pdb), opts, labelling);
}

// ---- table -------------------------------------------------------------------

struct TableArgs {
  std::vector<std::string> pdb;
  std::string results, output_dir, prefix = "structure", fps_json;
  bool best_fit = false, save_pdb = false, fps_camera = false;
};

double number_or(const nlohmann::json& entry, const char* key, double fallback) {
  if (!entry.contains(key) || entry[key].is_null()) return fallback;
  return entry[key].get<double>();
}

std::string string_or(const nlohmann::json& entry, const char* key, const std::string& fallback) {
  return entry.contains(key) ? entry[key].get<std::string>() : fallback;
}

void run_table(const TableArgs& a) {
  const nlohmann::json payload = nlohmann::json::parse(read_text_file(a.results));

  FPSResultTable parsed;
  parsed.n_molecules = static_cast<int>(
      number_or(payload, "n_molecules", static_cast<double>(a.pdb.size())));
  parsed.reference_number = static_cast<int>(number_or(payload, "reference_number", 0));
  parsed.distance_type = string_or(payload, "distance_type", "Rmp");
  if (payload.contains("rows")) {
    const nlohmann::json& rows = payload["rows"];
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const nlohmann::json& entry = rows[i];
      FPSResultRow row;
      row.number = entry["number"].get<int>();
      row.chi2 = entry["chi2"].is_null() ? std::nan("") : entry["chi2"].get<double>();
      row.chi2_bond = number_or(entry, "chi2_bond", 0.0);
      row.chi2_clash = number_or(entry, "chi2_clash", 0.0);
      row.converged = entry.contains("converged") ? entry["converged"].get<bool>() : false;
      row.parent = static_cast<int>(number_or(entry, "parent", 0));
      row.method = string_or(entry, "method", "Unknown");
      row.rmsd_previous = number_or(entry, "rmsd_previous", -1.0);
      row.rmsd_reference = number_or(entry, "rmsd_reference", 0.0);
      row.poses = json_dump_python(entry.contains("poses") ? entry["poses"] : nlohmann::json::array());
      if (entry.contains("pairs")) {
        const nlohmann::json& items = entry["pairs"];
        for (std::size_t k = 0; k < items.size(); ++k) {
          const nlohmann::json& item = items[k];
          PairDistance pair;
          pair.name = item["name"].get<std::string>();
          pair.position1 = item["position1"].get<std::string>();
          pair.position2 = item["position2"].get<std::string>();
          pair.distance_exp = number_or(item, "distance_exp", std::nan(""));
          pair.distance_model = number_or(item, "distance_model", std::nan(""));
          pair.distance_type = string_or(item, "distance_type", "Rmp");
          pair.error_neg = number_or(item, "error_neg", 0.0);
          pair.error_pos = number_or(item, "error_pos", 0.0);
          row.pairs.push_back(pair);
        }
      }
      parsed.rows.push_back(row);
    }
  }

  const std::string labelling =
      !a.fps_json.empty() ? a.fps_json : string_or(payload, "fps_json", "");
  if (a.best_fit) {
    if (labelling.empty()) {
      throw CLI::ValidationError("--best-fit needs an assembly to superpose with, and this results "
                                 "file records no labelling file: pass --fps-json");
    }
    parsed = add_best_fit(parsed, create_docking_assembly(a.pdb, labelling));
  }
  const FPSExportOptions opts = options(a.prefix, a.best_fit, a.save_pdb, a.fps_camera, nullptr);
  write_all(a.output_dir, parsed, fps_molecules(a.pdb), opts, labelling);
}

// ---- screen ------------------------------------------------------------------

struct ScreenArgs {
  std::vector<std::string> structures;
  std::string fps_json, positions, score_set, output_dir, prefix = "screening_";
  bool fps_filter_number = false;
};

void run_screen(const ScreenArgs& a) {
  const std::string labelling = labelling_file(a.fps_json, a.positions);
  const std::vector<ScreenedStructure> ranked =
      screen_structures(a.structures, labelling, a.score_set);
  FPSExportOptions opts;
  opts.prefix = a.prefix;
  opts.fps_filter_number = a.fps_filter_number;
  const std::string chi2 = write_fps_screening_chi2_table(a.output_dir, ranked, opts);
  const std::string r = write_fps_screening_r_table(a.output_dir, ranked, opts);
  std::cout << format("structures: %lu", (unsigned long)ranked.size()) << "\n";
  std::cout << "wrote " << chi2 << "\n";
  std::cout << "wrote " << r << "\n";
}

}  // namespace fps_export

void add_fps_export_subs(CLI::App& app) {
  CLI::App* group = app.add_subcommand("fps-export", R"doc(FPS's exports, and the error estimation that fills them.)doc");
  group->footer(R"doc(The door onto `SaveForm`, the dialog FPS writes every result file from, plus
`ErrorEstimation`, the mode that produces a table with more than one row in it.


    errors   perturb the docked model's own distances, re-fit, and report
             the spread -- FPS's parametric bootstrap
    table    write the five export files from a results table
    screen   rank a library and write FPS's two Filter-mode tables


Five files come out of `errors` and `table`:

    <prefix><N>.pml         one PyMOL script per row
    <prefix>Overlay.pml     every row as its own object
    <prefix>OverlayStates.pml   one object, one state per row
    <prefix>Rtable_<T>.txt  model distances, one row per result
    <prefix>chi2table.txt   the results grid

The sixth thing FPS writes, `SimulationResults.bin`, is a .NET
`BinaryFormatter` dump that cannot be read outside a legacy .NET target and is
meaningless without the project it sat beside. `results.json` replaces it, and
carries the molecule manifest the binary's "re-attach by position" reload
needed and never had.


Three FPS defects are not reproduced. Each is a flag, so an FPS number can
still be reproduced on demand:

  * the best fit is computed against the stated reference, not read from a
    stale GUI field (`--best-fit`);
  * `_tmp.pdb` is written into the export directory and removed;
  * `rotate`/`translate` carry `camera=0` (`--fps-camera` turns that off).


Author: Thomas-Otavio Peulen)doc");
  group->require_subcommand(1);

  {
    std::shared_ptr<fps_export::ErrorsArgs> a = std::make_shared<fps_export::ErrorsArgs>();
    CLI::App* sub = group->add_subcommand("errors", R"doc(FPS's error estimation: a parametric bootstrap around a docked pose.)doc");
    sub->footer(R"doc(The docked model's own distances become the truth -- residuals zeroed --
each is perturbed once by its own asymmetric error bars, and the structure
is re-minimised **from the parent pose**. The spread of the replicas'
RMSD against the parent is the reported uncertainty.


It is not `imp_bff dock-errors`. That command re-docks from independent
random starts and measures how reproducible the *optimiser* is; this one
holds the optimiser still and resamples the *data*.


The perturbation is a decision, not a detail. For error bars of 10 and
3 A:


    model            P(R' > R)   mean shift   density at the target
    split-normal        0.769      +5.585 A   continuous
    fps-sign-split      0.500      +2.792 A   jumps by a factor 0.3


Neither is unbiased -- an asymmetric density has its mean away from its
mode -- but only the first is a density with the quantiles its two error
bars claim. FPS's is the second.


Examples
--------

  imp_bff fps-export errors -p protein_1R0A.pdb -p dna.pdb \
      -j hiv_rt.fps.json -c resolved -o errors/ -n 10

  imp_bff fps-export errors -p a.pdb -p b.pdb -j labels.fps.json \
      -o fps_compat/ --perturbation fps-sign-split --fps-d2 \
      --fps-rmsd-sign)doc");
    sub->add_option("-p,--pdb", a->pdb, "One structure per rigid body; repeat, in body order.")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-j,--fps-json", a->fps_json, "Labelling and distance file.")
        ->check(CLI::ExistingFile);
    sub->add_option("--positions", a->positions, "Legacy C# FPS labelling-positions .txt instead.")
        ->check(CLI::ExistingFile);
    sub->add_option("-c,--score-set", a->score_set, "A named score set.");
    sub->add_option("-o,--output-dir", a->output_dir,
                    "Where the parent, the replicas and the exports go.")
        ->required();
    sub->add_option("-n,--replicas", a->replicas, "Synthetic datasets to draw and re-fit.")
        ->capture_default_str();
    sub->add_option("--seed", a->seed, "RNG seed.")->capture_default_str();
    sub->add_option("--perturbation", a->perturbation,
                    "split-normal: the two-piece normal, continuous, half masses s+/(s+ + s-). "
                    "fps-sign-split: FPS's, each half mass 1/2 and the density discontinuous at "
                    "the target.")
        ->check(CLI::IsMember({"fps-sign-split", "split-normal"}))
        ->capture_default_str();
    sub->add_flag("--fps-d2", a->fps_d2,
                  "Reproduce FPS's D2: leave deselected distances unperturbed so they pin the "
                  "replica to the parent pose. Biases the reported error downward.");
    sub->add_option("--steps", a->steps, "Minimisation steps per replica.")->capture_default_str();
    a->ev_opt = sub->add_option(
        "--ev-weight", a->ev_weight,
        "Excluded-volume weight. FPS's parameters give it a large share of the score at a docked "
        "pose, which pins the replicas; lower it to see what the data alone constrain.");
    a->clash_opt = sub->add_option("--clash-tolerance", a->clash_tolerance,
                                   "Overlap in A that costs one chi-square unit. FPS ships 0.5 "
                                   "for this mode, four times harder than docking.");
    sub->add_option("--parent-poses", a->parent_poses,
                    "JSON poses to bootstrap around; without it a docking run produces the parent "
                    "first.")
        ->check(CLI::ExistingFile);
    sub->add_flag("--keep-replicas", a->keep_replicas, "Keep each replica's own output directory.");
    sub->add_flag("--fps-rmsd-sign", a->fps_rmsd_sign,
                  "Report RMSD with FPS's sign error (|Ur - t| rather than |Ur + t|), for "
                  "comparing against an FPS number.");
    sub->add_option("--prefix", a->prefix, "File-name prefix.")->capture_default_str();
    sub->add_flag("--best-fit", a->best_fit,
                  "Superpose every row onto the parent in the PyMOL scripts.");
    sub->add_flag("--save-pdb", a->save_pdb,
                  "Have each script save its assembled complex as a PDB.");
    sub->add_flag("--labels", a->labels, "Add the labelling positions as PyMOL pseudoatoms.");
    sub->add_flag("--fps-camera", a->fps_camera,
                  "Omit camera=0, as FPS does. The script then only works in a session whose "
                  "camera has not been rotated.");
    sub->callback([a] {
      set_current_sub("fps-export errors");
      IMP::set_log_level(IMP::SILENT);
      fps_export::run_errors(*a);
    });
  }
  {
    std::shared_ptr<fps_export::TableArgs> a = std::make_shared<fps_export::TableArgs>();
    CLI::App* sub = group->add_subcommand("table", R"doc(Re-write the five export files from a results table.)doc");
    sub->footer(R"doc(Everything except `Converged`, `Method` and `Parent` can be recomputed
from the poses; those three cannot, which is why they are in the file and
not derived. See `okf/references/fps-export-formats.md` §9.)doc");
    sub->add_option("-p,--pdb", a->pdb, "One structure per rigid body; repeat, in body order.")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-r,--results", a->results, "A results.json written by a previous run.")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-o,--output-dir", a->output_dir, "Where the files go.")->required();
    sub->add_option("--prefix", a->prefix, "File-name prefix.")->capture_default_str();
    sub->add_option("-j,--fps-json", a->fps_json,
                    "The labelling file, when --best-fit needs an assembly to superpose with. "
                    "Defaults to the one recorded in the results file.")
        ->check(CLI::ExistingFile);
    sub->add_flag("--best-fit", a->best_fit, "Superpose every row onto the reference.");
    sub->add_flag("--save-pdb", a->save_pdb, "Have each script save its assembled complex.");
    sub->add_flag("--fps-camera", a->fps_camera, "Omit camera=0, as FPS does.");
    sub->callback([a] {
      set_current_sub("fps-export table");
      IMP::set_log_level(IMP::SILENT);
      fps_export::run_table(*a);
    });
  }
  {
    std::shared_ptr<fps_export::ScreenArgs> a = std::make_shared<fps_export::ScreenArgs>();
    CLI::App* sub = group->add_subcommand("screen", R"doc(Rank a library and write FPS's two Filter-mode tables.)doc");
    sub->footer(R"doc(
Read `NaNs` beside `Chi2r`: a distance with no model value is left out of
the reduced chi-square and counted there instead, so a structure whose
volumes came out empty scores flatteringly well on very few restraints.)doc");
    sub->add_option("-s,--structures", a->structures, "A structure or a directory of them; repeat.")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-j,--fps-json", a->fps_json, "Labelling and distance file.")
        ->check(CLI::ExistingFile);
    sub->add_option("--positions", a->positions, "Legacy C# FPS labelling-positions .txt instead.")
        ->check(CLI::ExistingFile);
    sub->add_option("-c,--score-set", a->score_set, "A named score set.");
    sub->add_option("-o,--output-dir", a->output_dir, "Where the tables go.")->required();
    sub->add_option("--prefix", a->prefix, "File-name prefix.")->capture_default_str();
    sub->add_flag("--fps-filter-number", a->fps_filter_number,
                  "Number the R table's rows FPS's way -- the 0-based array index, one less than "
                  "every other Number column in FPS. Join on File either way.");
    sub->callback([a] {
      set_current_sub("fps-export screen");
      IMP::set_log_level(IMP::SILENT);
      fps_export::run_screen(*a);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

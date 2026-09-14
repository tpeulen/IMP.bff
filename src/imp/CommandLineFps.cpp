/** \file CommandLineFps.cpp
 *  \brief `imp_bff fps ...`: FRET-restrained rigid-body docking and screening
 *         -- the FPS run modes.
 *
 *  Port of `bin/imp_bff_fps`. The engines are `Docking.h`, `FPSProject.h`
 *  and `FPS.h`; this file is the grammar, the flag-over-project precedence
 *  and the reports. Help texts are the Python program's docstrings, carried
 *  over verbatim.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/log.h>
#include <IMP/random.h>

#include <IMP/bff/Docking.h>
#include <IMP/bff/FPS.h>
#include <IMP/bff/FPSProject.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace fps {

//! The labelling file to read, from either spelling.
/*! fps.json is one file; the legacy C# FPS format is two, and
    `read_fps_json` takes the positions file and finds the distances beside
    it. Passing the distances file explicitly is the honest spelling, so it is
    accepted and checked rather than guessed at. */
std::string labelling_file(const std::string& fps_json, const std::string& positions,
                           const std::string& distances) {
  if (!fps_json.empty() && !positions.empty()) {
    throw CLI::ValidationError("give either --fps-json or --positions/--distances, not both");
  }
  if (!fps_json.empty()) return fps_json;
  if (positions.empty()) {
    throw CLI::ValidationError("no labelling file: pass --fps-json, or --positions with "
                               "--distances for the legacy C# FPS format");
  }
  if (!distances.empty()) {
    const std::string beside = path_join(path_dirname(path_abs(positions)), "Distances.txt");
    if (path_abs(distances) != beside) {
      throw CLI::ValidationError(
          "the legacy reader looks for the distances file beside the positions file and "
          "named Distances.txt; " + distances + " is neither. Copy it there, or convert once "
          "with `imp_bff fps convert`.");
    }
  }
  return positions;
}

//! Read a project and print what is wrong with it, if anything.
/*! Printed rather than raised: a run may deliberately point a project at
    other structures with `-p`, so a stale path is not a reason to refuse. */
FPSProject load_project(const std::string& path) {
  FPSProject project = read_fps_project(path);
  const std::vector<std::string> problems = project.get_problems();
  for (std::size_t i = 0; i < problems.size(); ++i) {
    std::cerr << "project: " << problems[i] << "\n";
  }
  return project;
}

//! The flags every run mode shares.
struct Inputs {
  std::vector<std::string> pdb;
  std::string fps_json, positions, distances, project, score_set;
};

void add_inputs(CLI::App* sub, Inputs& in, bool with_project_help_block) {
  sub->add_option("-p,--pdb", in.pdb,
                  "One structure per rigid body; repeat, in body order. Required unless "
                  "--project supplies them.")
      ->check(CLI::ExistingFile);
  sub->add_option("-j,--fps-json", in.fps_json, "Labelling and distance file (fps.json).")
      ->check(CLI::ExistingFile);
  sub->add_option("--positions", in.positions,
                  "Legacy C# FPS labelling-positions .txt (instead of -j).")
      ->check(CLI::ExistingFile);
  sub->add_option("--distances", in.distances,
                  "Legacy C# FPS distances .txt, beside the positions file.")
      ->check(CLI::ExistingFile);
  sub->add_option("-P,--project", in.project,
                  with_project_help_block
                      ? "A project file (.fps.json, or FPS's legacy project .txt) supplying the "
                        "structures, the labelling source, the score set and this mode's parameter "
                        "block. Any flag given explicitly wins over it."
                      : "A project file (.fps.json, or FPS's legacy project .txt) supplying the "
                        "structures, the labelling source, the score set and the mode's "
                        "parameters. Any flag given explicitly wins over it.")
      ->check(CLI::ExistingFile);
}

//! The structures, the labelling file and the score set a run will use.
/*! **An explicit flag always beats the project.** A project is a saved set of
    defaults, not a lock: `-p` replaces the whole body list, `-j`/`--positions`
    replaces the labelling source, and a non-empty `-c` replaces the score set. */
void run_inputs(const FPSProject* project, const Inputs& in, std::vector<std::string>& pdbs,
                std::string& labelling, std::string& score_set) {
  score_set = in.score_set;
  if (project == nullptr) {
    pdbs = in.pdb;
    labelling = labelling_file(in.fps_json, in.positions, in.distances);
    return;
  }
  pdbs = in.pdb.empty() ? project->get_structure_paths() : in.pdb;
  if (pdbs.empty()) {
    throw CLI::ValidationError("the project lists no structures; pass -p once per rigid body");
  }
  if (!in.fps_json.empty() || !in.positions.empty()) {
    labelling = labelling_file(in.fps_json, in.positions, in.distances);
  } else {
    labelling = project->get_labelling_path();
  }
  if (score_set.empty()) score_set = project->score_set;
}

//! A number for a table cell; `nan` and `inf` are printed, not hidden.
std::string num(double value, int digits) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  return format("%.*f", digits, value);
}

std::string lower(std::string s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  }
  return s;
}

//! Print a score and, optionally, the experiment-against-model table.
void report(const DockingResult& result, const std::string& score_set, bool show_pairs) {
  if (std::isinf(result.score)) {
    std::cerr << "score: inf -- at least one distance has no model value. The table below marks "
                 "them; a position whose accessible volume came out empty is the usual cause, "
                 "and the volume says so when it is computed.\n";
  } else {
    std::cout << format("score (chi2): %.4f", result.score) << "\n";
  }
  std::cout << format("volumes: %d   distances: %d%s", result.n_avs, result.n_distances,
                      score_set.empty() ? "" : ("   score set: " + score_set).c_str())
            << "\n";
  if (result.n_bonds) {
    // FPS's Ebond. A *subset* of the score, printed as one.
    std::cout << format("bonds: %d   E_bond: %.4f (of the score above, not added to it)",
                        result.n_bonds, result.e_bond)
              << "\n";
  }
  if (!show_pairs) return;
  std::cout << "\n";
  std::cout << format("%-26s %8s %8s %8s %10s %6s", "pair", "exp", "model", "dev", "chi2", "kind")
            << "\n";
  for (std::size_t i = 0; i < result.pairs.size(); ++i) {
    const PairDistance& p = result.pairs[i];
    // `Rmp` here rather than the file's own type means an end of the pair is a
    // point and there was no cloud to average over; `bond` means both ends are.
    const std::string kind = p.is_bond ? std::string("bond") : p.distance_type;
    if (std::isfinite(p.distance_model)) {
      std::cout << format("%-26s %8.2f %8.2f %8.2f %10.3f %6s", p.name.c_str(), p.distance_exp,
                          p.distance_model, p.get_residual(), p.get_chi2(), kind.c_str())
                << "\n";
    } else {
      std::cout << format("%-26s %8.2f %8s %8s %10s %6s", p.name.c_str(), p.distance_exp, "-",
                          "-", "no volume", kind.c_str())
                << "\n";
    }
  }
}

//! A path as a project should store it: relative when it stays inside.
std::string portable(const std::string& path, const std::string& out_dir) {
  if (path.empty()) return "";
  const std::string relative = path_rel(path, out_dir.empty() ? std::string(".") : out_dir);
  if (relative.compare(0, 2, "..") != 0) return relative;
  return path_abs(path);
}

//! Write a project file, with its paths stored the way it will be read.
/*! `self_contained` copies the labelling into the project itself; otherwise
    the project points at the labelling source it was given. A legacy pair is
    stored as the pair. */
void write_project(FPSProject& project, const std::string& out_path,
                   const std::vector<std::string>& structures, const std::string& labelling,
                   bool self_contained = true) {
  const std::string out_dir = path_dirname(path_abs(out_path));
  project.structures.clear();
  for (std::size_t i = 0; i < structures.size(); ++i) {
    project.structures.push_back(portable(structures[i], out_dir));
  }
  std::string positions_json = "{}", distances_json = "{}", score_sets_json = "{}";
  if (self_contained) {
    const FPSDocument document = read_fps_json(labelling);
    positions_json = document.positions;
    distances_json = document.distances;
    score_sets_json = document.score_sets;
    project.labelling_json = "";
    project.positions_path = "";
    project.distances_path = "";
  } else if (ends_with(labelling, ".json")) {
    project.labelling_json = portable(labelling, out_dir);
    project.positions_path = "";
    project.distances_path = "";
  } else {
    project.labelling_json = "";
    project.positions_path = portable(labelling, out_dir);
    const std::string beside = path_join(path_dirname(path_abs(labelling)), "Distances.txt");
    project.distances_path = file_exists(beside) ? portable(beside, out_dir) : std::string();
  }
  write_fps_project(out_path, project, positions_json, distances_json, score_sets_json, "{}",
                    true);
}

// ---- score -----------------------------------------------------------------

struct ScoreArgs {
  Inputs in;
  std::string output_csv, clash_radii_source;
  bool full_av = true, pairs = true;
  double clash_radii_scale = 1.0;
  CLI::Option* source_opt = nullptr;
  CLI::Option* scale_opt = nullptr;
};

void run_score(const ScoreArgs& a) {
  std::unique_ptr<FPSProject> project;
  if (!a.in.project.empty()) project.reset(new FPSProject(load_project(a.in.project)));
  std::vector<std::string> pdbs;
  std::string labelling, score_set;
  run_inputs(project.get(), a.in, pdbs, labelling, score_set);
  const double sigma_da = project ? project->sigma_da : 6.0;
  const std::string source = lower(a.source_opt->count()
                                       ? a.clash_radii_source
                                       : (project ? project->clash_radii_source : "imp"));
  const double scale =
      a.scale_opt->count() ? a.clash_radii_scale : (project ? project->clash_radii_scale : 1.0);
  const DockingResult result = score_structures(pdbs, labelling, score_set, !a.full_av, sigma_da,
                                                a.output_csv, source, scale);
  report(result, score_set, a.pairs);
  if (source != "imp" || scale != 1.0) {
    // the clash term is part of `score`, so a reader comparing two runs has to
    // be told the spheres were not the same size
    std::cout << format("clash radii: %s x %g", source.c_str(), scale) << "\n";
  }
  if (!a.output_csv.empty()) std::cout << "wrote " << a.output_csv << "\n";
}

// ---- dock ------------------------------------------------------------------

struct DockArgs {
  Inputs in;
  std::string output_dir, save_project, optimize_selected, clash_radii_source;
  int iterations = 0, fixed_body = 0, refine_cycles = 0, seed = 0;
  double shuffle = 0, ev_weight = 0, sigma_da = 0, max_force = 0, clash_tolerance = 0,
         clash_radii_scale = 0;
  bool coarse_clash = true, resume = true;
  std::map<std::string, CLI::Option*> opt;
  bool given(const char* name) const {
    std::map<std::string, CLI::Option*>::const_iterator it = opt.find(name);
    return it != opt.end() && it->second->count() > 0;
  }
};

void run_dock(const DockArgs& a) {
  std::unique_ptr<FPSProject> project;
  if (!a.in.project.empty()) project.reset(new FPSProject(load_project(a.in.project)));
  std::vector<std::string> pdbs;
  std::string labelling, score_set;
  run_inputs(project.get(), a.in, pdbs, labelling, score_set);
  if (a.given("seed")) IMP::random_number_generator.seed(static_cast<unsigned>(a.seed));
  // The project's Dock block is the starting point; every flag that was
  // actually typed overrides it. Without a project this is exactly
  // `DockingParameters()`, the documented defaults.
  DockingParameters params = project ? project->get_docking_parameters("Dock") : DockingParameters();
  if (a.given("iterations")) params.n_frames = a.iterations;
  params.score_set = score_set;
  if (a.given("shuffle")) params.shuffle_max_translation = a.shuffle;
  if (a.given("fixed_body")) params.fixed_body = a.fixed_body;
  if (a.given("ev_weight")) params.ev_weight = a.ev_weight;
  if (a.given("sigma_da")) params.sigma_da = a.sigma_da;
  if (a.given("refine_cycles")) params.refine_av_cycles = a.refine_cycles;
  if (a.given("optimize_selected")) params.optimize_selected = a.optimize_selected;
  if (a.given("max_force")) params.max_force = a.max_force;
  if (a.given("clash_tolerance")) params.clash_tolerance = a.clash_tolerance;
  if (a.given("coarse_clash")) params.coarse_clash = a.coarse_clash;
  if (a.given("clash_radii_source")) params.clash_radii_source = a.clash_radii_source;
  params.clash_radii_source = lower(params.clash_radii_source);
  if (a.given("clash_radii_scale")) params.clash_radii_scale = a.clash_radii_scale;
  // `dock_minimize` refuses this too, but only after the output directory
  // exists; saying it here keeps a typo from leaving a half-made run behind
  if (params.coarse_clash &&
      (params.clash_radii_source != "imp" || params.clash_radii_scale != 1.0)) {
    throw CLI::ValidationError(
        "--clash-radii-source/--clash-radii-scale describe atoms and the coarse clash term is "
        "one bead per residue. Add --no-coarse-clash.");
  }
  std::string initial_poses;
  if (project && a.resume && !project->poses.empty() && project->poses != "[]") {
    initial_poses = project->poses;
    std::cout << "resuming from the pose stored in " << a.in.project << "\n";
  }
  make_directory(a.output_dir);
  const DockingResult result =
      dock_minimize(pdbs, labelling, a.output_dir, params, nullptr, initial_poses);
  report(result, score_set, true);
  std::cout << "\n";
  std::cout << "wrote " << path_join(a.output_dir, "docked.pdb") << "\n";
  std::cout << "      " << path_join(a.output_dir, "scores.csv") << "\n";
  std::cout << "      " << path_join(a.output_dir, "convergence.csv") << "\n";
  if (!a.save_project.empty()) {
    // What is saved is what the run *did*, not what the project said: a
    // continued run has to start from the same settings.
    FPSProject updated = project ? *project : FPSProject();
    updated.mode = "Dock";
    updated.score_set = score_set;
    updated.poses = result.poses;
    updated.fixed_body = params.fixed_body;
    updated.ev_weight = params.ev_weight;
    updated.sigma_da = params.sigma_da;
    updated.shuffle = params.shuffle_max_translation;
    updated.refine_av_cycles = params.refine_av_cycles;
    updated.coarse_clash = params.coarse_clash;
    updated.clash_radii_source = params.clash_radii_source;
    updated.clash_radii_scale = params.clash_radii_scale;
    FPSModeParameters block = updated.get_parameters("Dock");
    block.max_force = params.max_force;
    block.clash_tolerance = params.clash_tolerance;
    block.optimize_selected = params.optimize_selected;
    block.iterations = params.n_frames;
    updated.set_parameters("Dock", block);
    write_project(updated, a.save_project, pdbs, labelling);
    std::cout << "      " << a.save_project << " (the docked pose, ready to continue)\n";
  }
}

// ---- refine ----------------------------------------------------------------

struct RefineArgs {
  Inputs in;
  std::string output_dir;
  int steps = 500;
  double ev_weight = 1.0;
  CLI::Option* steps_opt = nullptr;
  CLI::Option* ev_opt = nullptr;
};

void run_refine(const RefineArgs& a) {
  std::unique_ptr<FPSProject> project;
  if (!a.in.project.empty()) project.reset(new FPSProject(load_project(a.in.project)));
  std::vector<std::string> pdbs;
  std::string labelling, score_set;
  run_inputs(project.get(), a.in, pdbs, labelling, score_set);
  // FPS's Refine block, not its Dock block; `refine_docking` takes only the
  // iteration budget and the clash weight
  const int steps = a.steps_opt->count() ? a.steps : (project ? project->refine.iterations : 500);
  const double ev_weight = a.ev_opt->count() ? a.ev_weight : (project ? project->ev_weight : 1.0);
  make_directory(a.output_dir);
  const DockingResult result =
      refine_docking(pdbs, labelling, a.output_dir, score_set, steps, ev_weight);
  report(result, score_set, true);
  std::cout << "\n";
  std::cout << "wrote " << path_join(a.output_dir, "refined.pdb") << "\n";
}

// ---- screen ----------------------------------------------------------------

struct ScreenArgs {
  std::vector<std::string> structures;
  std::string fps_json, positions, distances, project, score_set, output_csv, pairs_csv;
  int top = 10;
};

void run_screen(const ScreenArgs& a) {
  // A screen brings its own library with `-s`, so a project supplies only the
  // labelling source and the score set -- its `structures` are the rigid
  // bodies of a docking, which is a different list.
  std::string labelling, score_set = a.score_set;
  if (!a.project.empty()) {
    const FPSProject project = load_project(a.project);
    labelling = (a.fps_json.empty() && a.positions.empty())
                    ? project.get_labelling_path()
                    : labelling_file(a.fps_json, a.positions, a.distances);
    if (score_set.empty()) score_set = project.score_set;
  } else {
    labelling = labelling_file(a.fps_json, a.positions, a.distances);
  }
  const std::vector<ScreenedStructure> ranked =
      screen_structures(a.structures, labelling, score_set, a.output_csv, false);
  std::size_t n_nan = 0;
  for (std::size_t i = 0; i < ranked.size(); ++i) {
    if (!std::isfinite(ranked[i].score)) ++n_nan;
  }
  std::cout << format("%lu structures scored%s", (unsigned long)ranked.size(),
                      n_nan ? format(", %lu without a score", (unsigned long)n_nan).c_str() : "")
            << "\n";
  if (!ranked.empty() && n_nan == ranked.size()) {
    std::cerr << "every structure scored NaN -- that is a broken run, not a bad library. Check "
                 "that the labelling file resolves against these structures (`imp_bff fps score` "
                 "on one).\n";
  }
  std::cout << "\n";
  std::cout << format("%12s %10s %4s %3s %3s %5s %9s  %s", "score", "chi2_r", "1s", "2s", "3s",
                      "inv", "refRMSD", "structure")
            << "\n";
  const std::size_t shown =
      a.top == 0 ? ranked.size() : std::min(ranked.size(), static_cast<std::size_t>(a.top));
  for (std::size_t i = 0; i < shown; ++i) {
    const ScreenedStructure& e = ranked[i];
    std::cout << format("%12s %10s %4d %3d %3d %5d %9s  %s", num(e.score, 4).c_str(),
                        num(e.chi2_r, 4).c_str(), e.sigma1, e.sigma2, e.sigma3, e.invalid_r,
                        num(e.ref_rmsd, 3).c_str(), e.path.c_str())
              << "\n";
  }
  if (!a.pairs_csv.empty()) write_screening_pairs_csv(a.pairs_csv, ranked);
  if (!a.output_csv.empty()) {
    std::cout << "\n";
    std::cout << "wrote " << a.output_csv << "\n";
  }
  if (!a.pairs_csv.empty()) std::cout << "wrote " << a.pairs_csv << "\n";
}

// ---- convert ---------------------------------------------------------------

struct ConvertArgs {
  std::string positions, output;
  std::vector<std::string> pdb;
};

std::size_t json_size(const std::string& text) {
  return nlohmann::json::parse(text.empty() ? std::string("{}") : text).size();
}

void run_convert(const ConvertArgs& a) {
  const FPSDocument doc = read_old_lps_txt(a.positions, a.pdb);
  const std::string beside = path_join(path_dirname(path_abs(a.positions)), "Distances.txt");
  const bool have = file_exists(beside);
  const std::string distances = have ? read_old_distances_txt(beside) : std::string("{}");
  write_fps_json(a.output, doc.positions, distances, doc.score_sets, doc.extra, true);
  std::string molecules;
  for (std::size_t i = 0; i < doc.molecules.size(); ++i) {
    molecules += (i ? ", " : "") + doc.molecules[i];
  }
  std::cout << "molecules: " << molecules << "\n";
  std::cout << format("positions: %lu   distances: %lu", (unsigned long)json_size(doc.positions),
                      (unsigned long)json_size(distances))
            << "\n";
  std::cout << "wrote " << a.output << "\n";
  if (!have) {
    std::cerr << "no Distances.txt beside " << a.positions << "; wrote positions only\n";
  }
}

// ---- project ---------------------------------------------------------------

struct ProjectInitArgs {
  std::vector<std::string> pdb;
  std::string fps_json, positions, distances, output, score_set, mode = "Dock";
  bool self_contained = true;
  int fixed_body = 0;
  double ev_weight = 1.0, sigma_da = 6.0, shuffle = 10.0;
};

void run_project_init(const ProjectInitArgs& a) {
  const std::string labelling = labelling_file(a.fps_json, a.positions, a.distances);
  FPSProject p;
  p.mode = a.mode;
  p.score_set = a.score_set;
  p.fixed_body = a.fixed_body;
  p.ev_weight = a.ev_weight;
  p.sigma_da = a.sigma_da;
  p.shuffle = a.shuffle;
  write_project(p, a.output, a.pdb, labelling, a.self_contained);
  const FPSDocument document = read_fps_json(labelling);
  const nlohmann::json sets =
      nlohmann::json::parse(document.score_sets.empty() ? std::string("{}") : document.score_sets);
  std::cout << format("structures: %lu   positions: %lu   distances: %lu   score sets: %lu",
                      (unsigned long)a.pdb.size(), (unsigned long)json_size(document.positions),
                      (unsigned long)json_size(document.distances), (unsigned long)sets.size())
            << "\n";
  if (!a.score_set.empty()) {
    if (!sets.contains(a.score_set)) {
      std::string names;
      for (nlohmann::json::const_iterator it = sets.begin(); it != sets.end(); ++it) {
        names += (names.empty() ? "" : ", ") + it.key();
      }
      std::cerr << "score set '" << a.score_set << "' is not in the labelling file; it has "
                << (names.empty() ? std::string("none") : names) << "\n";
    } else {
      std::cout << format("score set: %s (%lu distances)", a.score_set.c_str(),
                          (unsigned long)sets[a.score_set]["distances"].size())
                << "\n";
    }
  }
  std::cout << "wrote " << a.output << "\n";
  std::cout << "run it with: imp_bff fps score --project " << a.output << "\n";
}

struct ProjectShowArgs {
  std::string path;
  bool parameters = true;
};

void run_project_show(const ProjectShowArgs& a) {
  const FPSProject p = read_fps_project(a.path);
  if (!p.source.empty()) std::cout << "source: " << p.source << "\n";
  std::cout << "mode: " << p.mode << "\n";
  const std::vector<std::string> structures = p.get_structure_paths();
  std::cout << format("structures (%lu, in body order):", (unsigned long)structures.size()) << "\n";
  for (std::size_t i = 0; i < structures.size(); ++i) {
    std::cout << format("  [%lu] %-58s %s", (unsigned long)i, structures[i].c_str(),
                        file_exists(structures[i]) ? "ok" : "MISSING")
              << "\n";
  }
  const std::string labelling = p.get_labelling_path();
  const bool same = path_abs(labelling) == path_abs(a.path);
  std::cout << "labelling: " << labelling << (same ? " (this file)" : "") << "\n";
  if (!p.distances_path.empty()) std::cout << "distances: " << p.get_distances_path() << "\n";
  std::cout << "score set: " << (p.score_set.empty() ? std::string("(every distance)") : p.score_set)
            << "\n";
  if (!p.selected_distances.empty()) {
    std::string names;
    for (std::size_t i = 0; i < p.selected_distances.size(); ++i) {
      names += (i ? ", " : "") + p.selected_distances[i];
    }
    std::cout << format("selected distances (%lu): %s", (unsigned long)p.selected_distances.size(),
                        names.c_str())
              << "\n";
  } else if (!p.selected_flags.empty()) {
    // FPS's own spelling: a Boolean[] parallel to the distances file
    int on = 0;
    for (std::size_t i = 0; i < p.selected_flags.size(); ++i) on += p.selected_flags[i];
    std::cout << format("selected distances: %d of %lu, by position (FPS's Boolean[]); run "
                        "`project convert` with the distances file to name them",
                        on, (unsigned long)p.selected_flags.size())
              << "\n";
  }
  const nlohmann::json poses = nlohmann::json::parse(p.poses.empty() ? std::string("[]") : p.poses);
  std::cout << "poses: "
            << (poses.size()
                    ? format("%lu bodies stored (a run can be continued from here)",
                             (unsigned long)poses.size())
                    : std::string("none stored (runs start from the input structures)"))
            << "\n";
  std::cout << format("fixed_body %d   ev_weight %g   sigma_da %g   shuffle %g", p.fixed_body,
                      p.ev_weight, p.sigma_da, p.shuffle)
            << "\n";
  std::cout << format("clash: %s radii x %g, %s", p.clash_radii_source.c_str(), p.clash_radii_scale,
                      p.coarse_clash ? "one bead per residue" : "all atoms")
            << "\n";
  std::cout << format("AV globals: grid %g (min %g)  linker sphere %g  nodes %d  E samples %d",
                      p.av.grid_size, p.av.min_grid_size, p.av.linker_initial_sphere,
                      p.av.link_search_nodes, p.av.e_samples)
            << "\n";
  std::cout << format("conversion: R0 %g A, polynomial order %d", p.conversion.forster_radius,
                      p.conversion.polynomial_order)
            << "\n";
  if (a.parameters) {
    std::cout << "\n";
    std::cout << format("%-17s %6s %6s %10s %11s %6s  %-15s %5s", "mode", "visc", "dt", "max_iter",
                        "max_force", "clash", "optimize", "n")
              << "\n";
    const std::vector<std::string> names = fps_mode_names();
    for (std::size_t i = 0; i < names.size(); ++i) {
      const FPSModeParameters b = p.get_parameters(names[i]);
      std::cout << format("%-17s %6.2f %6.2f %10d %11.1f %6.2f  %-15s %5d", names[i].c_str(),
                          b.viscosity_factor, b.time_step_factor, b.max_iterations, b.max_force,
                          b.clash_tolerance, b.optimize_selected.c_str(), b.iterations)
                << "\n";
    }
  }
  std::cout << "\n";
  const std::vector<std::string> problems = p.get_problems();
  if (problems.empty()) std::cout << "no problems\n";
  std::cout << std::flush;
  for (std::size_t i = 0; i < problems.size(); ++i) {
    std::cerr << "problem: " << problems[i] << "\n";
  }
}

struct ProjectConvertArgs {
  std::string path, output, fps_json, positions, distances;
  std::vector<std::string> pdb;
  bool self_contained = true;
};

void run_project_convert(const ProjectConvertArgs& a) {
  FPSProject p = read_fps_project_txt(a.path);
  std::cout << "read " << basename_of(a.path) << (p.source.empty() ? "" : " (" + p.source + ")")
            << "\n";
  const std::string labelling = (!a.fps_json.empty() || !a.positions.empty())
                                    ? labelling_file(a.fps_json, a.positions, a.distances)
                                    : p.get_labelling_path();
  if (!file_exists(labelling)) {
    throw CLI::ValidationError("the labelling file the project names does not exist here (" +
                               labelling + "); pass -j, or --positions with --distances");
  }
  const std::vector<std::string> structures = a.pdb.empty() ? p.get_structure_paths() : a.pdb;
  std::string missing;
  for (std::size_t i = 0; i < structures.size(); ++i) {
    if (!file_exists(structures[i])) missing += (missing.empty() ? "" : ", ") + structures[i];
  }
  if (!missing.empty()) {
    throw CLI::ValidationError("these structures do not exist here: " + missing +
                               ". Pass -p once per rigid body, in body order.");
  }
  // FPS's selection is positional -- a Boolean[] parallel to the LINE ORDER of
  // the legacy distances file. Name it now, while that file is at hand.
  const std::vector<int> flags = p.selected_flags;
  int on = 0;
  for (std::size_t i = 0; i < flags.size(); ++i) on += flags[i];
  std::string legacy_distances;
  if (!p.distances_path.empty() && file_exists(p.get_distances_path())) {
    legacy_distances = p.get_distances_path();
  } else if (!a.distances.empty()) {
    legacy_distances = a.distances;
  } else if (!ends_with(labelling, ".json")) {
    const std::string beside = path_join(path_dirname(path_abs(labelling)), "Distances.txt");
    if (file_exists(beside)) legacy_distances = beside;
  }
  if (!flags.empty() && !legacy_distances.empty()) {
    const std::vector<std::string> names = read_old_distances_order(legacy_distances);
    p.selected_distances = p.resolve_selected_distances(names);
    std::cout << format("mode %s, %lu structures, %d selected of %lu distances", p.mode.c_str(),
                        (unsigned long)structures.size(), on, (unsigned long)flags.size())
              << "\n";
    if (flags.size() != names.size()) {
      std::cerr << format("the selection has %lu flags but %s has %lu distances; only the "
                          "overlap could be named",
                          (unsigned long)flags.size(), basename_of(legacy_distances).c_str(),
                          (unsigned long)names.size())
                << "\n";
    }
  } else if (!flags.empty()) {
    std::cout << format("mode %s, %lu structures, %d selected of %lu distances (by position only)",
                        p.mode.c_str(), (unsigned long)structures.size(), on,
                        (unsigned long)flags.size())
              << "\n";
    std::cerr << "the selection is positional and its distances file is not here, so the flags "
                 "could not be turned into names; pass --distances to name them\n";
  } else {
    std::cout << format("mode %s, %lu structures, no explicit selection", p.mode.c_str(),
                        (unsigned long)structures.size())
              << "\n";
  }
  write_project(p, a.output, structures, labelling, a.self_contained);
  std::cout << "wrote " << a.output << "\n";
}

void silent() { IMP::set_log_level(IMP::SILENT); }

}  // namespace fps

void add_fps_subs(CLI::App& app) {
  CLI::App* group = app.add_subcommand("fps", R"doc(FRET-restrained rigid-body docking and screening -- the FPS run modes.)doc");
  group->footer(R"doc(A port of the run modes of FPS, the FRET Positioning and Screening toolkit
(Kalinin *et al.*, *Nat. Methods* **9**, 1218, 2012), onto IMP. FPS is a
Windows C# application; the physics it needs -- accessible volumes, the
asymmetric chi-square, rigid bodies and excluded volume -- is native here, and
this program is the door onto it.

Four modes, all reading the same two things: one PDB per rigid body, and one
labelling-and-distance file.

    score    what a structure scores against the data, pair by pair
    dock     move the bodies until they fit the data (minimisation)
    refine   polish a pose that is already roughly right
    screen   rank a library of structures, best first

A **project** file holds all of that in one place -- the structures in body
order, the labelling source, the selected distances, the five per-mode
parameter blocks and the pose a run reached -- so a run is one flag,
`--project`, rather than a twelve-flag command line. See `imp_bff fps project
--help`. Both project doors are read: the readable `.txt` twin FPS writes
beside every `.bin` it saves, and this module's extended `.fps.json`.

The labelling file is either an `fps.json` or the legacy **pair** of C# FPS
text files -- a labelling-positions file and a distances file -- which are read
directly, no conversion step. See `--positions/--distances`.

Worked examples on the shipped data: `imp_bff fps score --help`, and the
"Examples" section of each mode. The example data is HIV-1 reverse
transcriptase with its DNA primer/template (1R0A), which is FPS's own docking
test case: 11 labelling positions, 20 measured distances.


Author: Thomas-Otavio Peulen)doc");
  group->require_subcommand(1);
  group->set_version_flag("--version", "IMP.bff FPS modes");

  // ---- score
  {
    std::shared_ptr<fps::ScoreArgs> a = std::make_shared<fps::ScoreArgs>();
    CLI::App* sub = group->add_subcommand("score", R"doc(What a structure scores against the data, pair by pair.)doc");
    sub->footer(R"doc(The first thing to run: it says whether the labelling file resolves against
the structures at all, how many volumes were built, and which measured
distance the model misses and by how much.


Examples
--------
Score the shipped HIV-RT complex against its 18 usable distances:


  imp_bff fps score \
      -p $IMP/examples/bff/structure/HIV_RT/protein_1R0A.pdb \
      -p $IMP/examples/bff/structure/HIV_RT/dna.pdb \
      -j $IMP/examples/bff/structure/HIV_RT/hiv_rt.fps.json \
      -c resolved


The same, straight from the legacy C# FPS files, no conversion step:


  imp_bff fps score -p protein_1R0A.pdb -p dna.pdb \
      --positions LabelingPositions.txt --distances Distances.txt


Score set `all` includes the two distances of `p66_K287C`, whose volume is
empty in the assembled complex -- the site is buried at the protein-DNA
interface. Those pairs report "no volume" and the total is `inf`, which is
the honest answer rather than a total over whatever happened to work.


How big is an atom
------------------
The score has two halves: the FRET restraints and an excluded-volume term.
The second one measures overlap with the radii `read_pdb` gave the atoms --
IMP's CHARMM-derived **united-atom** set, which carries implicit hydrogens
(carbon 1.85-2.275 A). FPS uses element-keyed **Bondi** radii (carbon
1.70). `--clash-radii-source olga` selects the Bondi-scale table; the
restraint half does not move, only the clash half:


  imp_bff fps score -p $E/protein_1R0A.pdb -p $E/dna.pdb \
      -j $E/hiv_rt.fps.json -c resolved --clash-radii-source olga


On that command (measured):


  clash radii     score    of which clash   score - clash
  imp (default)  59.0404          24.8005         34.2399
  imp x0.90      38.8090           4.5691         34.2399
  olga           35.6291           1.3893         34.2399


Two thirds of the shipped example's score is the protein-DNA interface read
through radii the clash constant was not calibrated for. `score - clash` is
identical in every row, which is the check that the flag reaches the clash
term and nothing else -- the accessible volumes keep their own
`radii_source`.


Positions that are a point, not a volume
----------------------------------------
Two `simulation_type` values carry no accessible volume:


  XYZ    a mean dye position measured once, in some other structure's
         frame. Give it `reference_atoms` -- the atoms that define that
         frame -- and it is Kabsch-fitted onto whatever structure is being
         scored and the coordinate is carried through the fit. Without
         them the coordinate is used exactly as written, which is only
         right if the file was made against this structure.
  ATOM   an atom of the structure itself: a crosslink end, an anchor.


**A distance with either kind of end is scored as R_mp**, whatever its
`distance_type` says -- there is no cloud at that end for <R_DA> to average
over. The table's `kind` column shows `Rmp` for those rows, so what was
scored is visible rather than assumed.)doc");
    fps::add_inputs(sub, a->in, false);
    sub->add_option("-c,--score-set", a->in.score_set,
                    "Named score set from the file's chi2 section (default: every distance).");
    sub->add_option("-o,--output-csv", a->output_csv, "Write the per-pair table here as CSV.");
    sub->add_flag("--full-av,!--mean-position", a->full_av,
                  "Rebuild both volumes per evaluation (default), or score the separation of "
                  "their mean positions (faster, approximate).");
    sub->add_flag("--pairs,!--no-pairs", a->pairs,
                  "Print the experiment-against-model table (default).");
    a->source_opt = sub->add_option(
        "--clash-radii-source", a->clash_radii_source,
        "Which van der Waals radii the EXCLUDED-VOLUME part of the score measures overlap by "
        "(default: imp, or the project's). 'imp' is each particle's own united-atom radius; "
        "'olga' is the Bondi-scale table (C 1.70) FPS uses -- FPS's hard sphere.");
    a->source_opt->check(CLI::IsMember({"imp", "olga"}, CLI::ignore_case));
    a->scale_opt = sub->add_option("--clash-radii-scale", a->clash_radii_scale,
                                   "Multiply the clash radii by this (default: 1.0, or the "
                                   "project's); composes with --clash-radii-source.");
    sub->callback([a] {
      set_current_sub("fps score");
      fps::silent();
      fps::run_score(*a);
    });
  }

  // ---- dock
  {
    std::shared_ptr<fps::DockArgs> a = std::make_shared<fps::DockArgs>();
    CLI::App* sub = group->add_subcommand("dock", R"doc(Move the bodies until they fit the data (minimisation).)doc");
    sub->footer(R"doc(Each PDB becomes one rigid body. The bodies are shuffled, then driven by
conjugate gradients under the measured distances and an excluded-volume
term until the score stops falling. This is FPS's approach; what differs is
the optimiser underneath, so **scores are comparable with FPS and
coordinates are not**.

Writes `docked.pdb`, `scores.csv` (the per-pair table) and
`convergence.csv` (score against iteration, written as the run goes, so a
plot can watch it).


Examples
--------
Dock the DNA against the protein on the shipped example:


  imp_bff fps dock \
      -p $IMP/examples/bff/structure/HIV_RT/protein_1R0A.pdb \
      -p $IMP/examples/bff/structure/HIV_RT/dna.pdb \
      -j $IMP/examples/bff/structure/HIV_RT/hiv_rt.fps.json \
      -c resolved -o dock_out --seed 1


Start from the deposited pose instead of a random one, which is what to do
when the complex is roughly right already and only needs settling:


  imp_bff fps dock ... --shuffle 0 -o dock_out


A docking run is not deterministic unless `--seed` is given, and a single
run is one local minimum. Repeat it with different seeds and compare the
scores before believing any one pose.


Store the run and continue it
-----------------------------
`--project` supplies the structures, the labelling file, the score set and
the Dock parameter block; `--save-project` writes them back with the
docked pose, and a second `dock --project` on that file **resumes from
that pose** (there is no shuffle on a resume). What is saved is what the
run did, not what the project said, so a flag typed on the first run is
still in force on the second:


  imp_bff fps dock --project hiv_rt.project.fps.json -o out1           --shuffle 0 --seed 1 --save-project after1.fps.json
  imp_bff fps dock --project after1.fps.json -o out2           --save-project after2.fps.json


`--no-resume` ignores a stored pose and starts over from the input
structures.


**A resumed run does not start at the score the last one ended on, and
that is not a bug.** During a dock the volumes are frozen: each one is
computed once, at the input pose, and its mean position rides on the rigid
body as a proxy. Resuming re-samples every volume **in the pose it is
resuming from**, which is the first honest look at what the docked complex
does to the linkers. On the shipped HIV-RT example, `-c resolved
--shuffle 0 --seed 1 -n 30` (measured, 2026-09-01):


  first run, at the input pose (frozen volumes)   41.8586
  first run, at the end                           30.6265
  resumed at that same pose, volumes recomputed   39.0207


The gap is the size of the frozen-volume approximation at that pose, and
it is the number to look at before believing a docked score. `score` on
the docked structure gives a third number again -- it computes the full
<R_DA>_E over both clouds rather than a transfer function of the mean
positions.


The FPS protocol knobs
----------------------
Three settings that change the answer rather than decorating it. The
defaults are FPS's own docking defaults (`ProjectData.cs:67-81`).


  --optimize-selected  which DISTANCES contribute. In FPS "selected" is a
                       per-distance checkbox; here it is the score set, so
                       `all` scores every distance in the file and
                       `selected-then-all` runs the score set for half the
                       budget and then everything. **Clashes are always
                       global** -- FPS never gates them, and neither does
                       this.
  --max-force          the restraint is a parabola only out to
                       MaxForce*err^2/2 and is straight beyond it, so one
                       grossly violated distance cannot dominate. At 400
                       and an error of 3 A the knee is at 1800 A and the
                       cap is inert; at an error of 0.1 A it is at 2 A and
                       it bites. `--max-force 0` is a pure harmonic.
  --clash-tolerance    the overlap in angstrom that costs one chi2 unit.
                       1.0 is FPS's docking value, 0.5 its refinement one
                       -- four times harder per angstrom.


On the shipped HIV-RT example, `-c resolved --shuffle 0 -n 200`
(measured, 2026-08-31):


  --optimize-selected selected           score 30.5079   10 volumes, 18 distances
  --optimize-selected all                score 33.1715   11 volumes, 20 distances
  --optimize-selected selected-then-all  score 33.4733   11 volumes, 20 distances


The two extra distances are `p66_K287C_p_1bp` and `p66_K287C_p_19bp`, the
ones the `resolved` set leaves out -- so `all` and `selected` are not two
attempts at the same number, they are scores over different data. Compare
modes only through the same score set, or through `scores.csv` row by row.


The hard sphere: which radii the clash term measures by
-------------------------------------------------------
`--clash-tolerance` says what an angstrom of overlap costs. It does not say
**how big an atom is**, and FPS and IMP do not agree about that:


  FPS   element-keyed Bondi radii (vdW.txt): carbon 1.70 A.
  IMP   united-atom radii assigned by read_pdb from the CHARMM type,
        carrying implicit hydrogens: carbon 1.85-2.275 A.


`k_clash = 2/ClashTolerance^2` was calibrated against the first and is
applied here to the second, so the same interface reads as a far worse
clash. Measured statically at the input pose over HIV-RT's protein-DNA
interface at k = 8 (ClashTolerance 0.5, the error-estimation value):


  radii              overlapping pairs   total overlap   energy
  imp (default)                    268         90.49 A   198.40
  olga (Bondi scale)                30          7.59 A    11.11


A factor of **17.9**. `--clash-radii-source olga` is FPS's hard sphere;
`--clash-radii-scale` shrinks whatever the source gives. Both need
`--no-coarse-clash`, because with the coarse term the clash spheres are
one 2.5 A bead per residue and a per-atom radii table has nothing to say
about a residue -- the program refuses the combination rather than
approximating it.


**The scale is not a substitute for the source.** Bisected over that same
interface, the IMP-radii scale that reproduces Olga's pair count is
0.8206, its total overlap 0.8386, and its energy 0.8476 -- three answers,
and at the first the energy is 5.45 against Olga's 11.11. IMP's ratio to
Bondi is per-element (C 0.808, N 0.838, P 0.841, O 0.889, S 0.900 here)
and IMP's carbon alone spans 1.70-2.275 A where Bondi's is a single 1.70,
so one factor cannot match the contact distribution, only one number of
it. Use the scale to loosen a clash term, and the source to reproduce
FPS.


Nothing is written on the structure's own radii: a substituted set is
carried by shadow spheres. The accessible volumes still inflate their
obstacles by the radii the particles carry, which is what keeps this flag
a statement about the clash term alone.


A worked pair, on the shipped example (measured):


  imp_bff fps dock -p $E/protein_1R0A.pdb -p $E/dna.pdb \
      -j $E/hiv_rt.fps.json -c resolved -o out -n 20 --shuffle 0 --seed 1 \
      --no-coarse-clash --clash-tolerance 0.5 [--clash-radii-source olga]


  default (imp radii)   score 144.0617
  --clash-radii-source olga   score  34.7498


The per-pair table is nearly the same in both -- `p51_E194C_p_10bp` reads
60.14 against 59.91 A -- so the 109 units of difference are the excluded
volume, not the fit.


What it is worth, measured. On HIV-RT `resolved` with FPS's error-estimation
settings (ClashTolerance 0.5, --no-coarse-clash, 60 iterations) the parent
score and the bootstrap spread `imp_bff fps-export errors` reports:


  clash radii        parent   of which clash   RMSD spread (A)
  imp x1.00 (default) 150.69          110.98   0.0000 +- 0.0000
  olga x1.00           35.88            1.93   0.0689 +- 0.0155
  imp x0.85            43.29            2.71   0.0983 +- 0.0365
  imp x0.80            39.96            0.86   0.4040 +- 0.0953
  ev_weight 0          24.34            0.00   2.1297 +- 1.4584


The default puts 74 % of the score into the clash term and the bootstrap
then reports 0.000 A, correctly -- the pose is held by geometry, not by the
data. FPS's hard sphere takes that to 5 % and the spread stops being
identically zero. **It does not make the run data-driven**: 0.069 A is
still thirty times below what the same run gives with no clash term at all.
Read `parent_e_clash` before believing any spread.


Bond restraints
---------------
A distance whose two ends are both `ATOM` positions -- plain atoms of the
structure, no dye and no volume -- is a **bond**: a crosslink, or a
covalent tie holding two subunits together. Its two anchor atoms are
excluded from clash detection (their radii drop to 0.4 A) and its energy is
reported separately as `E_bond`, which is **part of the score, not an
addition to it**. The per-pair table marks such rows `bond`.)doc");
    fps::add_inputs(sub, a->in, true);
    sub->add_option("-o,--output-dir", a->output_dir,
                    "Where docked.pdb, scores.csv and convergence.csv go.")
        ->required();
    sub->add_option("--save-project", a->save_project,
                    "Write the project back here with the docked pose in it, so the run can be "
                    "continued (`dock --project` resumes from a project's stored pose).");
    sub->add_option("-c,--score-set", a->in.score_set,
                    "Named score set from the file's chi2 section (default: every distance).");
    a->opt["iterations"] = sub->add_option(
        "-n,--iterations", a->iterations,
        "Minimiser iteration budget (default: 500, or the project's Dock block).");
    a->opt["shuffle"] = sub->add_option(
        "--shuffle", a->shuffle,
        "Initial random displacement of the mobile bodies, A; 0 starts from the input pose "
        "(default: 10.0, or the project's).");
    a->opt["fixed_body"] = sub->add_option(
        "--fixed-body", a->fixed_body, "body_id held still; the others move (default: 0, or the "
                                       "project's).");
    a->opt["ev_weight"] = sub->add_option(
        "--ev-weight", a->ev_weight,
        "Weight of the excluded-volume (clash) term (default: 1.0, or the project's).");
    a->opt["sigma_da"] = sub->add_option(
        "--sigma-da", a->sigma_da,
        "Width of the mean-position transfer function, A (default: 6.0, or the project's).");
    a->opt["refine_cycles"] = sub->add_option(
        "--refine-cycles", a->refine_cycles,
        "FPS-style refinement cycles after docking: re-sample the volumes in the docked context "
        "and minimise again (default: 0, or the project's).");
    a->opt["seed"] =
        sub->add_option("--seed", a->seed, "Seed for the random shuffle; makes a run reproducible.");
    a->opt["optimize_selected"] =
        sub->add_option("--optimize-selected", a->optimize_selected,
                        "Which DISTANCES contribute: the score set, every distance in the file, "
                        "or the score set then everything (two phases of half the budget each). "
                        "Clashes are never gated (default: selected, or the project's Dock block).");
    a->opt["optimize_selected"]->check(
        CLI::IsMember({"selected", "all", "selected-then-all"}, CLI::ignore_case));
    a->opt["max_force"] = sub->add_option(
        "--max-force", a->max_force,
        "FPS MaxForce: past MaxForce*err^2/2 a restraint goes linear instead of parabolic. 0 is a "
        "pure harmonic (default: 400.0, or the project's Dock block).");
    a->opt["clash_tolerance"] = sub->add_option(
        "--clash-tolerance", a->clash_tolerance,
        "FPS ClashTolerance, A: the overlap costing one chi2 unit (k = 2/tolerance^2). 0 keeps "
        "IMP's own k = 1 (default: 1.0, or the project's Dock block).");
    a->opt["coarse_clash"] = sub->add_flag(
        "--coarse-clash,!--no-coarse-clash", a->coarse_clash,
        "Detect clashes on one 2.5 A bead per residue instead of every atom (about 3x cheaper per "
        "step). Must be OFF for --clash-radii-source / --clash-radii-scale (default: coarse, or "
        "the project's).");
    a->opt["clash_radii_source"] = sub->add_option(
        "--clash-radii-source", a->clash_radii_source,
        "Which van der Waals radii the EXCLUDED-VOLUME term measures overlap by. 'imp' is each "
        "particle's own united-atom radius; 'olga' is the Bondi-scale table (C 1.70) FPS's "
        "ClashTolerance was calibrated against. Needs --no-coarse-clash (default: imp, or the "
        "project's).");
    a->opt["clash_radii_source"]->check(CLI::IsMember({"imp", "olga"}, CLI::ignore_case));
    a->opt["clash_radii_scale"] = sub->add_option(
        "--clash-radii-scale", a->clash_radii_scale,
        "Multiply the clash radii by this. Composes with --clash-radii-source. Needs "
        "--no-coarse-clash (default: 1.0, or the project's).");
    sub->add_flag("--resume,!--no-resume", a->resume,
                  "Start from the pose stored in --project, when it has one (default).");
    sub->callback([a] {
      set_current_sub("fps dock");
      fps::silent();
      fps::run_dock(*a);
    });
  }

  // ---- refine
  {
    std::shared_ptr<fps::RefineArgs> a = std::make_shared<fps::RefineArgs>();
    CLI::App* sub = group->add_subcommand("refine", R"doc(Polish a pose that is already roughly right.)doc");
    sub->footer(R"doc(No shuffle, and the volumes' own mean positions are scored rather than
proxies riding on the rigid bodies -- the local settle after `dock`, or
after a pose has come from somewhere else entirely (a crystal structure, a
prediction, a previous run's `docked.pdb`).


Examples
--------
Settle the deposited HIV-RT complex against the data:


  imp_bff fps refine \
      -p $IMP/examples/bff/structure/HIV_RT/protein_1R0A.pdb \
      -p $IMP/examples/bff/structure/HIV_RT/dna.pdb \
      -j $IMP/examples/bff/structure/HIV_RT/hiv_rt.fps.json \
      -c resolved -o refine_out)doc");
    fps::add_inputs(sub, a->in, true);
    sub->add_option("-o,--output-dir", a->output_dir, "Where refined.pdb goes.")->required();
    sub->add_option("-c,--score-set", a->in.score_set,
                    "Named score set from the file's chi2 section (default: every distance).");
    a->steps_opt = sub->add_option("-n,--steps", a->steps,
                                   "Minimiser iteration budget (default: 500, or the project's "
                                   "Refine block).");
    a->ev_opt = sub->add_option("--ev-weight", a->ev_weight,
                                "Weight of the excluded-volume (clash) term (default: 1.0, or the "
                                "project's).");
    sub->callback([a] {
      set_current_sub("fps refine");
      fps::silent();
      fps::run_refine(*a);
    });
  }

  // ---- screen
  {
    std::shared_ptr<fps::ScreenArgs> a = std::make_shared<fps::ScreenArgs>();
    CLI::App* sub = group->add_subcommand("screen", R"doc(Rank a library of structures against the data, best first.)doc");
    sub->footer(R"doc(FRET screening: the complementary question to docking. Rather than moving
anything, every candidate structure is scored as it stands and the library
is ordered.

Each structure carries FPS's screening diagnostics beside its score:


  chi2_r     chi-square per scored distance (FPS's E) -- no clash term,
             so it is the column to compare between structures
  1s 2s 3s   distances off by more than 1, 2, 3 times the error on the
             side they are off to (too long: error_pos; too short:
             error_neg). The counts nest.
  inv        distances with no model value at all: an accessible volume
             that came out empty at one end
  refRMSD    Kabsch fit of the labelling positions' `reference_atoms`
             onto this structure, A -- how far a fixed (XYZ) position had
             to be carried to land here. 0 when the file declares none.


A fixed (`XYZ`) position is not just reported on, it is **scored**: the
frame is fitted onto each library structure, the coordinate is transported
through the fit, and every distance touching it enters chi2 as R_mp. Read
`refRMSD` beside the score for those runs -- a fixed position carried onto
a structure its frame does not fit is worth exactly as much as that fit.

A structure that cannot be scored is kept in the table with a NaN rather
than dropped -- a table with a row missing looks like a smaller experiment.
**Read the NaNs**: a table that is entirely NaN is not a library of bad
models, it is a broken run.


Examples
--------
Rank two models of T4 lysozyme against the 33-pair C2 state:


  imp_bff fps screen \
      -s $IMP/examples/bff/structure/T4L/3GUN.pdb \
      -s $IMP/examples/bff/structure/T4L/3GUN_faspr_port.pdb \
      -j $IMP/examples/bff/structure/T4L/fret.fps.json \
      -c chi2_C2_33p -o ranked.csv


which prints (measured, 2026-08-31):


         score     chi2_r   1s  2s  3s   inv   refRMSD  structure
       22.3165     1.3525   12   3   0     0     0.000  3GUN_faspr_port.pdb
       22.5277     1.3653   10   3   0     0     0.000  3GUN.pdb


-- the two are a hair apart in chi2_r and disagree on which pairs they
miss, which is exactly what the sigma columns are for.


Rank every PDB in a directory and keep the per-pair rows, which is what
says *which* restraint the runners-up miss:


  imp_bff fps screen -s ./library -j fret.fps.json -c chi2_C2_33p \
      -o ranked.csv --pairs-csv pairs.csv


Screening scores each structure **on its own**, so a labelling file whose
positions span two rigid bodies (the HIV-RT example) has nothing to screen
-- score or dock that one instead.)doc");
    sub->add_option("-s,--structure", a->structures, "A structure, or a directory of *.pdb; repeat.")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-j,--fps-json", a->fps_json, "Labelling and distance file (fps.json).")
        ->check(CLI::ExistingFile);
    sub->add_option("--positions", a->positions,
                    "Legacy C# FPS labelling-positions .txt (instead of -j).")
        ->check(CLI::ExistingFile);
    sub->add_option("--distances", a->distances,
                    "Legacy C# FPS distances .txt, beside the positions file.")
        ->check(CLI::ExistingFile);
    sub->add_option("-P,--project", a->project,
                    "A project file (.fps.json, or FPS's legacy project .txt) supplying the "
                    "structures, the labelling source, the score set and this mode's parameter "
                    "block. Any flag given explicitly wins over it.")
        ->check(CLI::ExistingFile);
    sub->add_option("-c,--score-set", a->score_set,
                    "Named score set from the file's chi2 section (default: every distance).");
    sub->add_option("-o,--output-csv", a->output_csv,
                    "Write the ranked table with its diagnostic columns here, best first.");
    sub->add_option("-n,--top", a->top, "How many of the best to print; 0 prints all.")
        ->capture_default_str();
    sub->add_option("--pairs-csv", a->pairs_csv,
                    "Write every structure's experiment-against-model rows here: one row per "
                    "structure and distance.");
    sub->callback([a] {
      set_current_sub("fps screen");
      fps::silent();
      fps::run_screen(*a);
    });
  }

  // ---- convert
  {
    std::shared_ptr<fps::ConvertArgs> a = std::make_shared<fps::ConvertArgs>();
    CLI::App* sub = group->add_subcommand("convert", R"doc(Read the legacy C# FPS files and write one fps.json.)doc");
    sub->footer(R"doc(Every mode reads the legacy pair directly, so this is not a required step
-- it is for when the file should be kept, edited or shared. The legacy
formats are **read only** here, deliberately: a decade of measurements live
in them and they stay readable, but nothing should produce another one.

A legacy position names its site by **atom serial number**, which means
nothing without the structure it was numbered in; the serial is resolved
to chain, residue and atom name against the PDBs given.


All five of FPS's position types that mean something are read: `AV1`,
`AV3`, `XYZ` and `ATOM` (`AVS` and `EDF` are spellings FPS parses and never
computes). `ATOM` lines -- `Name Molecule D/A ATOM <serial>`, the ends of a
crosslink -- were being dropped as an unrecognised dialect until
2026-08-31, so a converted file had fewer restraints than the original and
said nothing about it.


Examples
--------

  imp_bff fps convert --positions LabelingPositions.txt \
      -p protein_1R0A.pdb -p dna.pdb -o hiv_rt.fps.json)doc");
    sub->add_option("--positions", a->positions, "Legacy C# FPS labelling-positions .txt.")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-p,--pdb", a->pdb,
                    "Structures the atom serials are resolved against; repeat. Without any, the "
                    "positions file's own directory is scanned for *.pdb.")
        ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", a->output, "Where the fps.json goes.")->required();
    sub->callback([a] {
      set_current_sub("fps convert");
      fps::silent();
      fps::run_convert(*a);
    });
  }

  // ---- project
  CLI::App* project = group->add_subcommand("project", R"doc(The file that binds a run together: init, show, convert.)doc");
  project->footer(R"doc(A project holds the structures in body order, the labelling source, the
selected distances, the five per-mode parameter blocks (FPS's own shipped
values), the AV globals and the pose a run reached. Every run mode takes
`--project` in place of the pile of flags, and `dock --save-project` writes
the docked pose back, so a run can be stored and continued.


Two formats are read and one is written:


  *.fps.json   this module's own -- an fps.json with a `Project` section
               beside `Positions`, `Distances` and the chi2 sets. Read and
               written. Schema 1.4.
  *.txt        FPS's legacy project export: the readable twin `om.Export`
               writes beside every `.bin` a project is saved as. **Read
               only.** The `.bin` itself is a gzipped .NET BinaryFormatter
               dump and is not read at all -- the twin is always there.


Nothing writes a legacy format, deliberately: those files are a decade of
measurements and they stay readable, but nothing should produce another.


Quick start, on the shipped HIV-RT example:
  imp_bff fps project show $IMP/examples/bff/structure/HIV_RT/hiv_rt.project.fps.json
  imp_bff fps score --project $IMP/examples/bff/structure/HIV_RT/hiv_rt.project.fps.json)doc");
  project->require_subcommand(1);
  {
    std::shared_ptr<fps::ProjectInitArgs> a = std::make_shared<fps::ProjectInitArgs>();
    CLI::App* sub = project->add_subcommand("init", R"doc(Write a new project from the structures and the labelling file.)doc");
    sub->footer(R"doc(The five parameter blocks start at **FPS's own shipped values**
(`ProjectData.cs:61-149`) -- Dock and Refine really do differ, and by a
lot: MaxForce 400 against 10000, ClashTolerance 1.0 against 0.5. Edit the
file to change them; it is JSON.


Examples
--------
One self-contained file for the shipped HIV-RT run:


  imp_bff fps project init \
      -p $IMP/examples/bff/structure/HIV_RT/protein_1R0A.pdb \
      -p $IMP/examples/bff/structure/HIV_RT/dna.pdb \
      -j $IMP/examples/bff/structure/HIV_RT/hiv_rt.fps.json \
      -c resolved -o hiv_rt.project.fps.json


which prints (measured, 2026-09-01):


  structures: 2   positions: 11   distances: 20   score sets: 2
  score set: resolved (18 distances)
  wrote hiv_rt.project.fps.json
  run it with: imp_bff fps score --project hiv_rt.project.fps.json


Straight from the legacy C# FPS pair, which converts them on the way in:


  imp_bff fps project init -p protein_1R0A.pdb -p dna.pdb \
      --positions LabelingPositions.txt --distances Distances.txt \
      -o run.fps.json


`--reference` instead keeps the labelling where it is and stores a path to
it, which is what to use when several projects share one labelling file.)doc");
    sub->add_option("-p,--pdb", a->pdb, "One structure per rigid body; repeat, in body order.")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-j,--fps-json", a->fps_json, "Labelling and distance file (fps.json).")
        ->check(CLI::ExistingFile);
    sub->add_option("--positions", a->positions,
                    "Legacy C# FPS labelling-positions .txt (instead of -j).")
        ->check(CLI::ExistingFile);
    sub->add_option("--distances", a->distances,
                    "Legacy C# FPS distances .txt, beside the positions file.")
        ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", a->output, "Where the project goes (.fps.json).")->required();
    sub->add_option("-c,--score-set", a->score_set,
                    "Named score set from the file's chi2 section (default: every distance).");
    sub->add_option("-m,--mode", a->mode, "FPS's ProjectFPSMode: Dock moves bodies, Filter screens.")
        ->check(CLI::IsMember({"None", "Dock", "Filter"}))
        ->capture_default_str();
    sub->add_flag("--self-contained,!--reference", a->self_contained,
                  "Copy the labelling into the project (default; one file for a whole run), or "
                  "point at the file it came from.");
    sub->add_option("--fixed-body", a->fixed_body, "body_id held still; the others move.")
        ->capture_default_str();
    sub->add_option("--ev-weight", a->ev_weight, "Weight of the excluded-volume (clash) term.")
        ->capture_default_str();
    sub->add_option("--sigma-da", a->sigma_da, "Width of the mean-position transfer function, A.")
        ->capture_default_str();
    sub->add_option("--shuffle", a->shuffle,
                    "Initial random displacement of the mobile bodies, A.")
        ->capture_default_str();
    sub->callback([a] {
      set_current_sub("fps project init");
      fps::silent();
      fps::run_project_init(*a);
    });
  }
  {
    std::shared_ptr<fps::ProjectShowArgs> a = std::make_shared<fps::ProjectShowArgs>();
    CLI::App* sub = project->add_subcommand("show", R"doc(Print what a project says, and what is wrong with it.)doc");
    sub->footer(R"doc(Reads either door -- a `.fps.json` project or FPS's legacy project text
export -- so it is also how to look inside a file that came off a Windows
machine without opening FPS.


Examples
--------

  imp_bff fps project show \
      $IMP/examples/bff/structure/HIV_RT/hiv_rt.project.fps.json


which prints (measured, 2026-09-01; the paths are absolute in a real run):


  mode: Dock
  structures (2, in body order):
    [0] .../HIV_RT/protein_1R0A.pdb                                 ok
    [1] .../HIV_RT/dna.pdb                                          ok
  labelling: .../HIV_RT/hiv_rt.project.fps.json (this file)
  score set: resolved
  poses: none stored (runs start from the input structures)
  fixed_body 0   ev_weight 1   sigma_da 6   shuffle 10
  AV globals: grid 0.2 (min 0.4)  linker sphere 0.5  nodes 3  E samples 200000
  conversion: R0 52 A, polynomial order 3


  mode                visc     dt   max_iter   max_force  clash  optimize            n
  Dock                1.00   1.00     200000       400.0   1.00  Selected          500
  Refine              0.70   0.50     500000     10000.0   0.50  All               500
  Error estimation    0.70   1.00     100000     10000.0   0.50  All               500
  Sample              1.00   1.00       8000       400.0   1.00  Selected          500
  Screening           1.00   1.00     200000       400.0   1.00  Selected            0


  no problems


`max_iter` is FPS's own budget and is **not** what this port runs: FPS
counts damped-Verlet integrator steps, this counts conjugate-gradient
iterations, and `n` is that. FPS's value is carried so the settings behind
a published number survive the conversion.)doc");
    sub->add_option("path", a->path, "the project")->required()->check(CLI::ExistingFile);
    sub->add_flag("--parameters,!--no-parameters", a->parameters,
                  "Print the five per-mode parameter blocks (default).");
    sub->callback([a] {
      set_current_sub("fps project show");
      fps::silent();
      fps::run_project_show(*a);
    });
  }
  {
    std::shared_ptr<fps::ProjectConvertArgs> a = std::make_shared<fps::ProjectConvertArgs>();
    CLI::App* sub = project->add_subcommand("convert", R"doc(Read FPS's legacy project export and write one .fps.json.)doc");
    sub->footer(R"doc(FPS saves a project as a `.bin` and writes a readable `.txt` twin beside it
every single time (`MainForm.cs:559-560`). **That twin is what this reads**
-- the `.bin` is a gzipped .NET BinaryFormatter dump carrying
assembly-qualified type names, and it is not read at all.


One thing the twin loses, and FPS loses it too: the structure paths are
written space-separated and unquoted on one line, so a path containing a
space cannot be recovered. Pass `-p` once per body to re-point the project
when that happens -- or, more usually, because the file names `C:\Users\...`
and this is not Windows.


Examples
--------
Convert the shipped fixture, whose paths are relative and therefore
resolve:


  imp_bff fps project convert \
      $IMP/examples/bff/structure/HIV_RT/hiv_rt.project.txt \
      -o converted.fps.json


which prints (measured, 2026-09-01):


  read hiv_rt.project.txt (IMP.bff fixture in FPS project-export layout)
  mode Dock, 2 structures, 18 selected of 20 distances
  wrote converted.fps.json


-- 18 of 20, because the fixture's selection is FPS's `Boolean[]` with the
two `p66_K287C` distances off, which is exactly the `resolved` score set.
**The flags are positional**, parallel to the line order of the legacy
`Distances.txt`, so they are named against that file's order and not
against the fps.json (whose distances come back keyed and sorted). Without
the distances file they survive as `selected_flags` and stay unnamed.


A project off a Windows machine, re-pointed at local copies:


  imp_bff fps project convert MyProject.bin.txt -o run.fps.json \
      -p protein.pdb -p dna.pdb --positions LabelingPositions.txt)doc");
    sub->add_option("path", a->path, "FPS's legacy project .txt export")
        ->required()
        ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", a->output, "Where the converted project goes (.fps.json).")
        ->required();
    sub->add_option("-p,--pdb", a->pdb,
                    "Structures to use instead of the ones the project names; repeat, in body "
                    "order. A legacy project carries Windows paths, which do not exist here.")
        ->check(CLI::ExistingFile);
    sub->add_option("-j,--fps-json", a->fps_json, "Labelling file to use instead of the one it names.")
        ->check(CLI::ExistingFile);
    sub->add_option("--positions", a->positions,
                    "Legacy positions .txt to use instead of the one it names.")
        ->check(CLI::ExistingFile);
    sub->add_option("--distances", a->distances,
                    "Legacy distances .txt, beside the positions file.")
        ->check(CLI::ExistingFile);
    sub->add_flag("--self-contained,!--reference", a->self_contained,
                  "Copy the labelling into the project (default), or point at it.");
    sub->callback([a] {
      set_current_sub("fps project convert");
      fps::silent();
      fps::run_project_convert(*a);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

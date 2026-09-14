/** \file CommandLineAnalysis.cpp
 *  \brief `imp_bff analyze-trajectories` and `imp_bff av-vs-rotamer`: the
 *         analysis commands of the former Python program `bin/imp_bff`.
 *
 *  The engines are ProbeTrajectoryDensity.h and ProbeModelComparison.h; this
 *  file is the grammar and the two documents `av-vs-rotamer` writes. Help texts
 *  are the click docstrings. One thing the Python did that a compiled program
 *  cannot: it found its defaults beside its own file -- `traj_latest/` and
 *  `analysis/latest/` next to the program, the repository's `okf/` and
 *  `test/references/` above it. Here the first two are relative to the working
 *  directory, and the repository is `--repo`, or the nearest directory above
 *  the working directory that has `okf/prds`.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <IMP/bff/ProbeModelComparison.h>
#include <IMP/bff/ProbeTrajectoryDensity.h>
#include <IMP/bff/internal/NumpyCompat.h>
#include <IMP/bff/internal/Text.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace analysis {

namespace nc = numpy_compat;

// ---- analyze-trajectories ---------------------------------------------------

struct TrajectoryArgs {
  ProbeTrajectoryDensityOptions options;
};

// ---- av-vs-rotamer -------------------------------------------------------------

struct AVRotamerArgs {
  std::string system = "both", okf_dir, pins, repo, cutoffs = "30,10";
  int n_samples = 50000;
  double temperature = 298.15;
};

//! The nearest directory at or above the working directory with `okf/prds`.
std::string find_repo() {
  std::string dir = path_abs(".");
  while (true) {
    if (path_is_dir(path_join(dir, "okf/prds"))) return dir;
    if (dir == "/" || dir.empty()) return std::string();
    const std::string parent = path_dirname(dir);
    if (parent == dir) return std::string();
    dir = parent.empty() ? std::string("/") : parent;
  }
}

//! `str(Path(p))`
std::string pathlib_str(const std::string& p) {
  std::string out;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p[i] == '/' && !out.empty() && out[out.size() - 1] == '/') continue;
    out += p[i];
  }
  while (out.size() > 1 && out[out.size() - 1] == '/') out.erase(out.size() - 1);
  while (out.compare(0, 2, "./") == 0) out.erase(0, 2);
  return out.empty() ? std::string(".") : out;
}

void run_av_vs_rotamer(const AVRotamerArgs& a) {
  const std::string repo = a.repo.empty() ? find_repo() : a.repo;
  std::vector<std::string> systems;
  if (a.system == "both") {
    systems.push_back("hgbp1");
    systems.push_back("t4l");
  } else {
    systems.push_back(a.system);
  }
  const double r0 = av_rotamer_forster_radius();
  std::vector<std::string> md;
  md.push_back("# AV \xe2\x86\x94 rotamer-ensemble cross-validation (PRD-108)");
  md.push_back("");
  md.push_back("Recorded by `imp_bff av-vs-rotamer`. AVs from the fps.json positions (default FPS "
               "strip; the authored T4L `strip_mask` is outside the current dialect), rotamer "
               "ensembles screened at T = " + nc::python_repr(a.temperature) +
               " K without electrostatics; R0 = " + nc::python_repr(r0) +
               " \xc3\x85 (\xce\xba\xc2\xb2 = 2/3). This table is authoritative; the test pins the "
               "numbers and asserts loose sanity bounds only.");
  md.push_back("");
  nc::PyJson numbers = nc::PyJson::object();
  numbers.set("_note", nc::PyJson::string(
          "AV vs rotamer-ensemble numbers recorded by `imp_bff av-vs-rotamer` (PRD-108 stage 2); "
          "drift pins, not physics gates."));
  nc::PyJson settings = nc::PyJson::object();
  settings.set("n_samples", nc::PyJson::integer(a.n_samples));
  settings.set("temperature", nc::PyJson::number(a.temperature));
  settings.set("forster_radius", nc::PyJson::number(r0));
  settings.set("donor_library", nc::PyJson::string("AlexaFluor 488 C1R"));
  settings.set("acceptor_library", nc::PyJson::string("AlexaFluor 594 C1R"));
  settings.set("cutoffs", nc::PyJson::string(a.cutoffs));
  numbers.set("settings", settings);

  std::vector<int> cutoffs;
  {
    std::string token;
    std::istringstream in(a.cutoffs);
    while (std::getline(in, token, ',')) {
      const std::string t = trimmed(token);
      if (t.empty()) continue;
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 10);
      if (*end != '\0') throw SubError("invalid literal for int() with base 10: '" + token + "'");
      cutoffs.push_back(static_cast<int>(v));
    }
  }
  for (std::size_t c = 0; c < cutoffs.size(); ++c) {
    for (std::size_t s = 0; s < systems.size(); ++s) {
      const AVRotamerCase reference = av_rotamer_case(systems[s], cutoffs[c]);
      std::vector<AVRotamerPosition> positions;
      std::vector<AVRotamerPair> pairs;
      compare_av_and_rotamer(reference, a.n_samples, a.temperature, r0, positions, pairs);
      md.push_back(av_rotamer_markdown_table(positions, pairs, reference.title));
      numbers.set(systems[s] + format("_cutoff%d", cutoffs[c]),
                  nc::PyJson::raw(av_rotamer_summary_json(positions, pairs, 1, 1)));
    }
  }
  std::string text;
  for (std::size_t i = 0; i < md.size(); ++i) text += (i ? "\n" : "") + md[i];
  std::cout << text << "\n";

  std::string okf_path = !a.okf_dir.empty() ? pathlib_str(a.okf_dir)
                                            : (repo.empty() ? std::string()
                                                            : path_join(repo, "okf/validation"));
  if (!okf_path.empty()) {
    make_directory(okf_path);
    const std::string md_path = path_join(okf_path, "av_vs_rotamer.md");
    std::ofstream out(md_path.c_str());
    if (!out) throw SubError(md_path + ": cannot write");
    out << text;
    out.close();
    std::cout << "wrote " << md_path << "\n";
  }
  const std::string pin_path =
          !a.pins.empty() ? pathlib_str(a.pins)
                          : (repo.empty() ? std::string()
                                          : path_join(repo, "test/references/cgprobe_av_vs_rotamer_pins.json"));
  if (!pin_path.empty()) {
    std::ofstream out(pin_path.c_str());
    if (!out) throw SubError(pin_path + ": cannot write");
    out << numbers.dumps(1);
    out.close();
    std::cout << "wrote " << pin_path << "\n";
  }
}

}  // namespace analysis

void add_analysis_subs(CLI::App& app) {
#if !defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF)
  {
    std::shared_ptr<analysis::TrajectoryArgs> a = std::make_shared<analysis::TrajectoryArgs>();
    ProbeTrajectoryDensityOptions& o = a->options;
    CLI::App* sub = app.add_subcommand("analyze-trajectories", R"doc(Probe density and distance statistics over simulation trajectories.)doc");
    sub->add_option("--traj-root", o.traj_root,
                    "Directory holding the run subdirectory (default: traj_latest, in the working "
                    "directory).");
    sub->add_option("--runs-root", o.runs_root,
                    "Directory containing run*/ subdirs for combined analysis.");
    sub->add_flag("--combine-runs", o.combine_runs,
                  "Combine all run*/ trajectories under --runs-root into one analysis.");
    sub->add_option("--output-dir", o.output_dir,
                    "Where the per-mobile output goes (default: analysis/latest, in the working "
                    "directory).");
    sub->add_option("--mobile", o.mobiles,
                    "Mobile component name(s) to analyze (e.g. atto655). Repeatable.")
        ->required();
    sub->add_option("--mobile-template-cif", o.mobile_template_cifs,
                    "Path to template CIF for the mobile component, one per --mobile in order. A "
                    "mobile without one looks for cgprobe/templates/{mobile}.template.cif.");
    sub->add_option("--fixed-name", o.fixed_name,
                    "Fixed component name in the RMF hierarchy. Inferred if not provided.");
    sub->add_option("--system-name", o.system_name,
                    "Run output directory name (e.g. 'CX4_atto655_imp'). This is the subdirectory "
                    "under --traj-root that the runner wrote. Defaults to "
                    "'{fixed-name}_{mobile}_imp' when --fixed-name is set, otherwise auto-detected.");
    sub->add_option("--axis-element", o.axis_element,
                    "One-letter element symbol to orient the fixed-component axis (e.g. S).");
    sub->add_option("--resolution", o.resolution, "Density map resolution, A.")->capture_default_str();
    sub->add_option("--voxel-size", o.voxel_size, "Density map voxel size, A.")->capture_default_str();
    sub->add_option("--bin-width", o.bin_width, "Profile and histogram bin width, A.")
        ->capture_default_str();
    sub->add_option("--max-frames", o.max_frames, "0 means all frames.")->capture_default_str();
    sub->callback([a] {
      set_current_sub("analyze-trajectories");
      analyze_probe_trajectories(a->options, std::cout);
      std::cout << std::flush;
    });
  }
#endif
  {
    std::shared_ptr<analysis::AVRotamerArgs> a = std::make_shared<analysis::AVRotamerArgs>();
    CLI::App* sub = app.add_subcommand("av-vs-rotamer", R"doc(Compare AV clouds and rotamer ensembles position by position and pair by pair.)doc");
    sub->footer(
        "Writes the table to <okf-dir>/av_vs_rotamer.md and the numbers to --pins. Without\n"
        "either flag they go to okf/validation/ and test/references/ of --repo, or of the\n"
        "nearest directory above the working directory that has okf/prds; outside a checkout\n"
        "the table is only printed.");
    sub->add_option("--system", a->system, "hgbp1, t4l or both.")
        ->check(CLI::IsMember({"hgbp1", "t4l", "both"}))
        ->capture_default_str();
    sub->add_option("--okf-dir", a->okf_dir,
                    "Write av_vs_rotamer.md here (default: repo okf/validation if found).");
    sub->add_option("--pins", a->pins, "Write the pin JSON here (default: repo test/references).");
    sub->add_option("--repo", a->repo,
                    "The checkout whose okf/ and test/references/ the defaults point into.");
    sub->add_option("--n-samples", a->n_samples, "AV distance MC samples.")->capture_default_str();
    sub->add_option("--temperature", a->temperature)->capture_default_str();
    sub->add_option("--cutoffs", a->cutoffs,
                    "Rotamer-library cutoffs to compare (comma separated).")
        ->capture_default_str();
    sub->callback([a] {
      set_current_sub("av-vs-rotamer");
      analysis::run_av_vs_rotamer(*a);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

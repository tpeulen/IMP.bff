/** \file CommandLineDock.cpp
 *  \brief `imp_bff dock`, `imp_bff dock-errors`, `imp_bff flexfit`: sampling
 *         structures against FRET distances.
 *
 *  Ports of the three sampling commands of the Python program `bin/imp_bff`.
 *  The work is #IMP::bff::dock_replica_exchange,
 *  #IMP::bff::dock_from_independent_starts and #IMP::bff::flexible_fitting;
 *  this file is their grammar and their reports.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/random.h>

#include <IMP/bff/Docking.h>
#include <IMP/bff/FlexibleFitting.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace dock_commands {

struct DockArgs {
  std::vector<std::string> pdb;
  std::string fps_json, output_dir, score_set;
  int n_frames = 500, mc_steps = 10, n_best = 20, fixed_body = 0, seed = 0;
  double temperature = 1.0, max_translation = 4.0, max_rotation = 0.1, shuffle = 10.0,
         sigma_da = 6.0;
  bool full_av = false;
  CLI::Option* seed_opt = nullptr;
};

void run_dock(const DockArgs& a) {
  if (a.seed_opt->count()) IMP::random_number_generator.seed(static_cast<unsigned>(a.seed));
  DockingParameters params;
  params.score_set = a.score_set;
  params.n_frames = a.n_frames;
  params.mc_steps = a.mc_steps;
  params.mc_temperature = a.temperature;
  params.max_translation = a.max_translation;
  params.max_rotation = a.max_rotation;
  params.shuffle_max_translation = a.shuffle;
  params.n_best = a.n_best;
  params.fixed_body = a.fixed_body;
  params.sigma_da = a.sigma_da;
  params.mean_position_restraint = !a.full_av;
  const DockingResult result = dock_replica_exchange(a.pdb, a.fps_json, a.output_dir, params);
  std::cout << format("score %.4f over %d distances (%d accessible volumes)", result.score,
                      result.n_distances, result.n_avs)
            << "\n";
  std::cout << "wrote " << result.score_csv << "\n";
  if (!result.rmf_file.empty()) std::cout << "trajectory " << result.rmf_file << "\n";
}

struct DockErrorsArgs {
  std::vector<std::string> pdb;
  std::string fps_json, output_dir, score_set;
  int n_trials = 10, n_workers = 0, iterations = 500;
};

void run_dock_errors(const DockErrorsArgs& a) {
  DockingParameters params;
  params.score_set = a.score_set;
  params.n_frames = a.iterations;
  const DockingSpread spread =
      dock_from_independent_starts(a.pdb, a.fps_json, a.output_dir, params, a.n_trials, a.n_workers);
  std::cout << format("%d trials, score %.4f +/- %.4f", spread.n_trials, spread.score_mean,
                      spread.score_std)
            << "\n";
  std::cout << "best trial "
            << (spread.best_trial < 0 ? std::string("None") : format("%d", spread.best_trial))
            << " in "
            << (spread.trials.empty() ? std::string() : spread.trials.front().output_dir) << "\n";
}

struct FlexfitArgs {
  std::string input_pdb, labeling;
  FlexibleFittingParameters params;
  int seed = 0;
  CLI::Option* seed_opt = nullptr;
};

}  // namespace dock_commands

void add_dock_subs(CLI::App& app) {
  // ---- dock
  {
    std::shared_ptr<dock_commands::DockArgs> a = std::make_shared<dock_commands::DockArgs>();
    CLI::App* sub = app.add_subcommand("dock", "FRET-restrained rigid-body docking (Monte Carlo).");
    sub->footer(
        "Dock rigid bodies against FRET distances and report the fit.\n\n"
        "Writes PMI's run layout into --output-dir: initial.0.rmf3, rmfs/0.rmf3, the --n-best\n"
        "best frames as pdbs/model.<i>.pdb with pdbs/model.psf, best.scores.rex.py,\n"
        "stat.0.out and stat_replica.0.out, and scores.csv for the final pose.");
    sub->add_option("-p,--pdb", a->pdb, "PDB file; one per rigid body (repeatable).")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-j,--fps-json", a->fps_json, "fps.json labelling and distance file.")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-o,--output-dir", a->output_dir, "Where the RMF, PDBs and CSV go.")
        ->required();
    sub->add_option("-c,--score-set", a->score_set,
                    "Named score set (\"\" uses every distance).")
        ->capture_default_str();
    sub->add_option("-n,--n-frames", a->n_frames, "Outer Monte-Carlo iterations.")
        ->capture_default_str();
    sub->add_option("--mc-steps", a->mc_steps, "Monte-Carlo steps per frame.")
        ->capture_default_str();
    sub->add_option("--temperature", a->temperature,
                    "Monte-Carlo temperature (kT). As in the PMI sampler this command always "
                    "ran, one replica runs at kT = 1 whatever is given here.")
        ->capture_default_str();
    sub->add_option("--max-translation", a->max_translation, "Mover amplitude, Angstrom.")
        ->capture_default_str();
    sub->add_option("--max-rotation", a->max_rotation, "Mover amplitude, radian.")
        ->capture_default_str();
    sub->add_option("--shuffle", a->shuffle,
                    "Initial random shuffle, Angstrom; 0 does not shuffle.")
        ->capture_default_str();
    sub->add_option("--n-best", a->n_best, "Best-scoring models to write.")->capture_default_str();
    sub->add_option("--fixed-body", a->fixed_body, "body_id held fixed as the reference.")
        ->capture_default_str();
    sub->add_option("--sigma-da", a->sigma_da, "Mean-position transfer width, Angstrom.")
        ->capture_default_str();
    sub->add_flag("--full-av,!--mean-position", a->full_av,
                  "Rebuild both volumes every evaluation, or score their mean positions (the "
                  "default, and what sampling wants).");
    a->seed_opt = sub->add_option("--seed", a->seed,
                                  "Seed IMP's random number generator; makes a run reproducible.");
    sub->callback([a] {
      set_current_sub("dock");
      dock_commands::run_dock(*a);
    });
  }

  // ---- dock-errors
  {
    std::shared_ptr<dock_commands::DockErrorsArgs> a =
        std::make_shared<dock_commands::DockErrorsArgs>();
    CLI::App* sub = app.add_subcommand(
        "dock-errors", "Repeat docking from random starts and report the spread.");
    sub->footer(
        "Estimate the spread of a docking result over independent starts.\n\n"
        "Trial i minimises from a random start seeded i + 1 into <output-dir>/trial_<iii>;\n"
        "the per-run best models are superposed on the fixed body and their per-atom RMSF\n"
        "written to uncertainty.pdb and uncertainty.csv.");
    sub->add_option("-p,--pdb", a->pdb, "PDB file; one per rigid body (repeatable).")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-j,--fps-json", a->fps_json, "fps.json labelling and distance file.")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("-o,--output-dir", a->output_dir, "Where the trial directories go.")
        ->required();
    sub->add_option("-n,--n-trials", a->n_trials, "Independent docking trials.")
        ->capture_default_str();
    sub->add_option("-c,--score-set", a->score_set,
                    "Named score set (\"\" uses every distance).")
        ->capture_default_str();
    sub->add_option("--n-workers", a->n_workers,
                    "Parallel workers; 1 forces serial (default: one per trial, up to the CPU "
                    "count).");
    sub->add_option("--iterations", a->iterations, "Minimiser iterations per trial.")
        ->capture_default_str();
    sub->callback([a] {
      set_current_sub("dock-errors");
      dock_commands::run_dock_errors(*a);
    });
  }

  // ---- flexfit
  {
    std::shared_ptr<dock_commands::FlexfitArgs> a =
        std::make_shared<dock_commands::FlexfitArgs>();
    CLI::App* sub =
        app.add_subcommand("flexfit", "flexible fitting to experimental inter-label distances");
    sub->footer(
        "Samples the backbone dihedrals of the flexible residues named in the labelling "
        "file's FlexFit block against its FRET distances and a soft-sphere clash term:\n"
        "an anneal of 25 temperatures, then --num-frames Monte-Carlo steps.\n\n"
        "Example (examples/structure/TG2):\n"
        "  imp_bff flexfit -i topology.pdb -l flex.fps.json -c S1_chi2 -f FexResSet1");
    sub->add_option("-i,--input-pdb", a->input_pdb, "Input PDB file.")->required();
    sub->add_option("-l,--labeling", a->labeling,
                    "Input JSON (restraints & flexible amino acids)")
        ->required();
    sub->add_option("-c,--score-set", a->params.score_set, "Distance set used for scoring")
        ->required();
    sub->add_option("-f,--flex-set", a->params.flex_set, "Name of flexible amino acid set");
    sub->add_option("-o,--output", a->params.output, "Out file either RMF3 or PDB")
        ->capture_default_str();
    sub->add_option("-n,--num-frames", a->params.num_frames, "Total number of frames.")
        ->capture_default_str();
    sub->add_option("-p,--period", a->params.period, "Steps between write to output file")
        ->capture_default_str();
    sub->add_option("-t,--temperature", a->params.temperature, "Temperature in sampling step.")
        ->capture_default_str();
    sub->add_option("-r,--radii-scaling", a->params.radii_scaling,
                    "Radii scaling parameter (0.3 < r < 1.0).")
        ->capture_default_str();
    sub->add_option("-m,--mode", a->params.mode,
                    "Mode (F)ast uses computed the positional distributions of labels once, and "
                    "updates the mean label positions with reference toe the rigid body the "
                    "label is attached to. The option (S)low updated / recomputes the AV at each "
                    "iteration")
        ->capture_default_str();
    a->seed_opt = sub->add_option("--seed", a->seed,
                                  "Seed IMP's random number generator; makes a run reproducible.");
    sub->callback([a] {
      set_current_sub("flexfit");
      if (a->seed_opt->count()) IMP::random_number_generator.seed(static_cast<unsigned>(a->seed));
      flexible_fitting(a->input_pdb, a->labeling, a->params);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

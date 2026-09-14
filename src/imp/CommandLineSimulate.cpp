/** \file CommandLineSimulate.cpp
 *  \brief `imp_bff simulate`: propagate an explicit dye on a force-field
 *         system (MD, MC, or both).
 *
 *  Port of the `simulate` command of the Python `bin/imp_bff`. The runner is
 *  #IMP::bff::run_probe_system_simulation (`ProbeSystemSimulation.h`); this
 *  file is the grammar.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <IMP/log.h>
#include <IMP/random.h>

#include <IMP/bff/ProbeSystemSimulation.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace simulate {

struct Args {
  std::string system_cif, systems_dir;
  ProbeSystemSimulationOptions options;
};

void run(const Args& a) {
  // what the Python's IMP.setup_from_argv + set_check_level did, with the
  // generator seeded so a run repeats
  IMP::set_check_level(IMP::NONE);
  IMP::random_number_generator.seed(static_cast<boost::int64_t>(a.options.init_seed));
  const std::vector<std::string> paths = probe_system_paths(a.system_cif, a.systems_dir);
  for (std::size_t i = 0; i < paths.size(); ++i) {
    run_probe_system_simulation(paths[i], a.options, std::cout);
  }
}

}  // namespace simulate

void add_simulate_subs(CLI::App& app) {
  std::shared_ptr<simulate::Args> a = std::make_shared<simulate::Args>();
  ProbeSystemSimulationOptions& o = a->options;
  CLI::App* sub = app.add_subcommand("simulate", R"doc(Propagate an explicit dye on a force-field system (MD, MC, or both).)doc");
  sub->add_option("--system-cif", a->system_cif, "A force-field system mmCIF.");
  sub->add_option("--systems-dir", a->systems_dir, "A directory of system mmCIFs; each *.cif is run.");
  sub->add_option("--output-root", o.output_root,
                  "Root directory for trajectory output. Defaults to 'output/trajs' relative to "
                  "the current working directory.");
  sub->add_option("--md-steps", o.md_steps, "Total MD steps after MC pre-stage.")
      ->capture_default_str();
  sub->add_option("--write-every", o.write_every, "Steps per written frame.")->capture_default_str();
  sub->add_option("--sampling-mode", o.sampling_mode, "simple_md, hybrid_md_mc or multi_restart.")
      ->check(CLI::IsMember({"simple_md", "hybrid_md_mc", "multi_restart"}))
      ->capture_default_str();
  sub->add_option("--mc-steps", o.mc_steps, "Rigid-body MC steps per block.")->capture_default_str();
  sub->add_option("--mc-temperature", o.mc_temperature, "kT of the rigid-body Metropolis test.")
      ->capture_default_str();
  sub->add_option("--rb-max-translation", o.rb_max_translation,
                  "Rigid translation scale, A (the step's sigma is a third of it).")
      ->capture_default_str();
  sub->add_option("--rb-max-rotation-deg", o.rb_max_rotation_deg,
                  "Rigid rotation scale, degrees (the step's sigma is a third of it).")
      ->capture_default_str();
  sub->add_option("--friction-ps", o.friction_ps,
                  "Langevin friction coefficient in ps^-1. Overrides system CIF value.");
  sub->add_option("--temperature-k", o.temperature_k,
                  "MD temperature in K. Overrides system CIF value.");
  sub->add_option("--mobile-group", o.mobile_group,
                  "The group that moves (default: <mobile component>_all).");
  sub->add_option("--fixed-flex-mode", o.fixed_flex_mode,
                  "'static': fixed component fully rigid. 'flex': atoms in the {name}_flex group "
                  "are released.")
      ->check(CLI::IsMember({"static", "flex"}))
      ->capture_default_str();
  sub->add_flag("--freeze-mobile-rings,!--no-freeze-mobile-rings", o.freeze_mobile_rings,
                "Fallback only: use ring detection as MD-fixed set when mmCIF has no "
                "_ff_dof_md_fixed_member (default: on).");
  sub->add_flag("--init-placement,!--no-init-placement", o.init_placement,
                "Place the guest around the host by score first (default: on).");
  sub->add_option("--init-distance-a", o.init_distance_a, "Placement centre distance, A.")
      ->capture_default_str();
  sub->add_option("--init-trials", o.init_trials, "Placement trials.")->capture_default_str();
  sub->add_option("--init-seed", o.init_seed,
                  "Seed of the placement search and of the run's random number generator.")
      ->capture_default_str();
  sub->add_option("--com-pull-k", o.com_pull_k,
                  "Weak harmonic force pulling guest COM to host COM.");
  sub->add_option("--go-mobile-k", o.go_mobile_k, "Native-contact k of the mobile component.")
      ->capture_default_str();
  sub->add_option("--go-fixed-k", o.go_fixed_k, "Native-contact k of released fixed atoms.")
      ->capture_default_str();
  sub->add_option("--go-cutoff", o.go_cutoff, "Native-contact cutoff, A.")->capture_default_str();
  sub->add_option("--log-every-frames", o.log_every_frames,
                  "Progress print frequency in written frames.")
      ->capture_default_str();
  sub->add_option("--n-restarts", o.n_restarts,
                  "Number of independent restarts (for multi_restart mode).")
      ->capture_default_str();
  sub->callback([a] {
    set_current_sub("simulate");
    simulate::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

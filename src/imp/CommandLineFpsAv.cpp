/** \file CommandLineFpsAv.cpp
 *  \brief `imp_bff fps-av`: compute one accessible volume -- FPS's AV
 *         interface.
 *
 *  Port of `bin/imp_bff_fps_av`, itself a port of the standalone AV dialog
 *  of FPS (Kalinin *et al.*, *Nat. Methods* **9**, 1218, 2012), which FPS
 *  reaches as `FpsGui -av`. One site, one dye, one volume, written where a
 *  viewer can open it. The dye presets are FPS's own
 *  (#IMP::bff::fps_linker_presets); `--dye` selects one and anything it sets
 *  can still be overridden.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/Model.h>
#include <IMP/Particle.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/Selection.h>
#include <IMP/atom/pdb.h>
#include <IMP/log.h>

#include <IMP/bff/FPS.h>
#include <IMP/bff/PathMap.h>
#include <IMP/bff/ProbeAccessibleVolumeDecorator.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace fps_av {

struct Args {
  std::string pdb, chain, atom = "CB", dye, radii, output;
  int residue = 0;
  bool av1 = false, as_json = false;
  double linker_length = 0, linker_width = 0, grid = 0, clearance = 0;
  bool have_length = false, have_width = false, have_grid = false, have_clearance = false;
};

std::vector<std::string> preset_names() {
  std::vector<std::string> out;
  const std::vector<FPSLinkerPreset> presets = fps_linker_presets();
  for (std::size_t i = 0; i < presets.size(); ++i) out.push_back(presets[i].name);
  return out;
}

void run(const Args& a) {
  IMP::set_log_level(IMP::SILENT);

  double length = 20.0, width = 4.5;
  std::vector<double> r(3, 0.0);
  r[0] = 3.5;
  if (!a.dye.empty()) {
    const std::vector<FPSLinkerPreset> presets = fps_linker_presets();
    for (std::size_t i = 0; i < presets.size(); ++i) {
      if (presets[i].name != a.dye) continue;
      length = presets[i].linker_length;
      width = presets[i].linker_width;
      if (a.av1) {
        r[0] = presets[i].radius_av1;
        r[1] = r[2] = 0.0;
      } else {
        r.assign(presets[i].radii_av3, presets[i].radii_av3 + 3);
      }
    }
  } else if (a.av1) {
    throw CLI::ValidationError("--av1 selects a column of a --dye preset");
  }
  if (a.have_length) length = a.linker_length;
  if (a.have_width) width = a.linker_width;
  if (!a.radii.empty()) {
    std::string text = a.radii;
    for (std::size_t i = 0; i < text.size(); ++i) {
      if (text[i] == ',') text[i] = ' ';
    }
    std::istringstream in(text);
    std::vector<double> values;
    std::string token;
    while (in >> token) {
      char* end = nullptr;
      const double v = std::strtod(token.c_str(), &end);
      if (*end != '\0') throw CLI::ValidationError("--radii: '" + token + "' is not a number");
      values.push_back(v);
    }
    if (values.size() != 1 && values.size() != 3) {
      throw CLI::ValidationError("--radii takes one or three values");
    }
    r.assign(3, 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) r[i] = values[i];
  }
  // FPS's own rule, so a preset reproduces its grid
  const double grid = a.have_grid ? a.grid : fps_default_grid(length, width, r);

  IMP_NEW(IMP::Model, model, ());
  const IMP::atom::Hierarchy hier = IMP::atom::read_pdb(
      a.pdb, model, new IMP::atom::NonWaterNonHydrogenPDBSelector());
  IMP::atom::Selection selection(hier);
  selection.set_residue_index(a.residue);
  selection.set_atom_type(IMP::atom::AtomType(a.atom));
  if (!a.chain.empty()) selection.set_chain_id(a.chain);
  const IMP::ParticlesTemp chosen = selection.get_selected_particles();
  if (chosen.empty()) {
    throw SubError(a.pdb + " has no atom " + a.atom + " in residue " +
                   (a.chain.empty() ? std::string("*") : a.chain) + format("%d", a.residue));
  }

  IMP::Particle* particle = new IMP::Particle(model);
  ProbeAccessibleVolumeDecorator::do_setup_particle(
      model, particle->get_index(), chosen[0]->get_index(), length,
      IMP::algebra::Vector3D(r[0], r[1], r[2]), width, a.have_clearance ? a.clearance : -1.0,
      0.0, -1, grid);
  ProbeAccessibleVolumeDecorator av(model, particle->get_index());
  av.resample();

  const std::size_t n_voxels = av.get_map()->get_xyz_density().size();
  const double used_clearance = av.get_effective_allowed_sphere_radius();
  if (n_voxels == 0) {
    throw SubExit(1, format("no accessible voxel: the site is buried, or the clearance (%.2f A) "
                            "does not clear the obstacle inflation of half the linker width "
                            "(%.2f A). Nothing written.",
                            used_clearance, 0.5 * width));
  }

  write_av(av, a.output);
  const IMP::algebra::Vector3D mean = av.get_mean_position(false);
  if (a.as_json) {
    OrderedJson out;
    out.str("pdb", a.pdb).str("chain", a.chain).integer("residue", a.residue).str("atom", a.atom);
    out.num("linker_length", length).num("linker_width", width);
    out.list("radii", {json_float(r[0]), json_float(r[1]), json_float(r[2])});
    out.num("grid", grid).num("clearance", used_clearance);
    out.integer("n_voxels", static_cast<long>(n_voxels));
    out.list("mean_position", {json_float(mean[0]), json_float(mean[1]), json_float(mean[2])});
    out.str("output", a.output);
    std::cout << out.dump() << "\n";
    return;
  }
  std::string radii_text;
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (r[i] <= 0) continue;
    if (!radii_text.empty()) radii_text += ", ";
    radii_text += format("%g", r[i]);
  }
  std::cout << format("site      %s%d %s", a.chain.empty() ? "*" : a.chain.c_str(), a.residue,
                      a.atom.c_str())
            << "\n";
  std::cout << format("dye       L = %g A, W = %g A, radii = %s", length, width, radii_text.c_str())
            << "\n";
  std::cout << format("grid      %g A   clearance %.2f A", grid, used_clearance) << "\n";
  std::cout << format("volume    %lu voxels", (unsigned long)n_voxels) << "\n";
  std::cout << format("mean      (%.3f, %.3f, %.3f)", mean[0], mean[1], mean[2]) << "\n";
  std::cout << "wrote     " << a.output << "\n";
}

}  // namespace fps_av

void add_fps_av_subs(CLI::App& app) {
  std::shared_ptr<fps_av::Args> a = std::make_shared<fps_av::Args>();
  CLI::App* sub = app.add_subcommand(
      "fps-av", "Compute one accessible volume and write it where a viewer can open it.");
  sub->footer(
      ".xyz and .pqr write the point cloud with its weights, .dx and .mrc the density grid.\n"
      "PyMOL and VMD open all four.\n\n"
      "Examples:\n"
      "  imp_bff fps-av -p 3GUN.pdb -c A -r 132 -d alexa488-long -o d132.xyz\n"
      "  imp_bff fps-av -p 3GUN.pdb -c A -r 132 -d alexa488-long --av1 -o d132.xyz\n"
      "  imp_bff fps-av -p 3GUN.pdb -r 132 -l 20 -w 4.5 --radii 3.5 -o d132.xyz\n\n"
      "On the clearance: the path search inflates every obstacle by half the linker width,\n"
      "so the free sphere around the attachment atom has to clear that inflation or the\n"
      "source is walled in and the volume comes back empty. Leaving --clearance unset\n"
      "derives a value that does. FPS's own LinkerInitialSphere x width is a different\n"
      "quantity, so an FPS number does not transfer here. An empty volume is a legitimate\n"
      "answer -- a buried site has no room for a dye -- and nothing is written for it.");
  sub->add_option("-p,--pdb", a->pdb, "The structure the dye is attached to.")
      ->required()
      ->check(CLI::ExistingFile);
  sub->add_option("-c,--chain", a->chain, "Chain of the attachment site (default: any).");
  sub->add_option("-r,--residue", a->residue, "Residue number of the attachment site.")->required();
  sub->add_option("-a,--atom", a->atom, "Attachment atom.")->capture_default_str();
  sub->add_option("-d,--dye", a->dye, "An FPS linker preset; sets length, width and radii.")
      ->check(CLI::IsMember(fps_av::preset_names()));
  sub->add_flag("--av1", a->av1, "Use the preset's single AV1 radius instead of its three.");
  sub->add_option("-l,--linker-length", a->linker_length, "Override the linker length, A.")
      ->each([a](const std::string&) { a->have_length = true; });
  sub->add_option("-w,--linker-width", a->linker_width, "Override the linker width (diameter), A.")
      ->each([a](const std::string&) { a->have_width = true; });
  sub->add_option("--radii", a->radii, "Override the dye radii, e.g. '3.5' or '5,4.5,1.5'.");
  sub->add_option("-g,--grid", a->grid,
                  "Grid spacing, A. Default follows FPS: max(min(0.2L, 0.2W, 0.4R_i), 0.4).")
      ->each([a](const std::string&) { a->have_grid = true; });
  sub->add_option("--clearance", a->clearance,
                  "Source clearance, A. Derived when not given, which is almost always right.")
      ->each([a](const std::string&) { a->have_clearance = true; });
  sub->add_option("-o,--output", a->output, "Where to write it: .xyz, .pqr, .dx or .mrc.")
      ->required();
  sub->add_flag("--json", a->as_json, "Report as JSON.");
  sub->callback([a] {
    set_current_sub("fps-av");
    fps_av::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

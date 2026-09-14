/** \file CommandLineModelling.cpp
 *  \brief `imp_bff av-export`, `openmm`, `build-system`, `select-pairs`,
 *         `rotamer r0|predict`: the thin modelling commands.
 *
 *  Ports of the commands of the former `bin/imp_bff` whose engines were
 *  already C++: the accessible-volume decorator, the flat-bottom MD restraint
 *  system and its OpenMM writers, the force-field system builder, Olga's
 *  greedy pair selection and the rotamer FRET prediction. What lives here is
 *  the grammar, the ensemble loading `select-pairs` did in numpy, and the
 *  reports. Help texts are the click docstrings, carried over verbatim.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <IMP/Model.h>
#include <IMP/Particle.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/Selection.h>
#include <IMP/atom/pdb.h>
#include <IMP/core/XYZ.h>
#include <IMP/log.h>

#include <IMP/bff/Clustering.h>
#include <IMP/bff/FRETRotamer.h>
#include <IMP/bff/ProbeAccessibleVolumeDecorator.h>
#include <IMP/bff/ProbeAccessibleVolumeMeanDistanceRestraint.h>
#include <IMP/bff/ProbeForceFieldCIF.h>
#include <IMP/bff/ProbeLibrary.h>
#include <IMP/bff/ProbeNetworkRestraint.h>
#include <IMP/bff/ProbePairSelection.h>
#include <IMP/bff/ProbeTopology.h>
#include <IMP/bff/internal/Text.h>

//! `select-pairs --rmf` reads the ensemble with IMP.rmf, as the Python did:
//! the restraint's volumes see the radii IMP.rmf restores. The module build
//! always has IMP.rmf; a core+imp build that links it says so with
//! IMPBFF_WITH_IMP_RMF.
#if !defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF)
#  define IMPBFF_CLI_HAS_IMP_RMF 1
#  include <IMP/rmf/atom_io.h>
#  include <IMP/rmf/frames.h>
#  include <RMF/FileConstHandle.h>
#else
#  define IMPBFF_CLI_HAS_IMP_RMF 0
#endif

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace modelling {

//! Python's `repr` of a str, for messages that quoted one with `%r`.
std::string py_repr(const std::string& s) {
  const char quote = (s.find('\'') != std::string::npos && s.find('"') == std::string::npos) ? '"' : '\'';
  std::string out(1, quote);
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\') out += "\\\\";
    else if (c == quote) out += std::string("\\") + quote;
    else if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out + quote;
}

IMP::atom::Hierarchy read_structure(const std::string& path, IMP::Model* model) {
  return IMP::atom::read_pdb(path, model, new IMP::atom::NonWaterNonHydrogenPDBSelector());
}

// ---- av-export ---------------------------------------------------------------

struct AvExportArgs {
  std::string pdb, chain, atom = "CB", output;
  int residue = 0;
  double linker_length = 20.5, linker_width = 1.5, radius1 = 3.5, grid = 1.5;
  bool chain_weighting = false;
};

void run_av_export(const AvExportArgs& a) {
  IMP::set_log_level(IMP::SILENT);
  IMP_NEW(IMP::Model, model, ());
  const IMP::atom::Hierarchy hier = read_structure(a.pdb, model);
  IMP::atom::Selection site(hier);
  site.set_chain_id(a.chain);
  site.set_residue_index(a.residue);
  site.set_atom_type(IMP::atom::AtomType(a.atom));
  const IMP::ParticlesTemp chosen = site.get_selected_particles();
  if (chosen.empty()) {
    throw SubError(a.pdb + " has no atom " + a.atom + " in residue " + a.chain +
                   format("%d", a.residue));
  }
  IMP::Particle* particle = new IMP::Particle(model);
  ProbeAccessibleVolumeDecorator av = ProbeAccessibleVolumeDecorator::setup_particle(
      model, particle->get_index(), chosen[0]->get_index());
  // the JSON the script built, in its key order
  OrderedJson parameters;
  parameters.num("linker_length", a.linker_length).num("linker_width", a.linker_width);
  parameters.num("radius1", a.radius1).num("simulation_grid_resolution", a.grid);
  parameters.boolean("chain_weighting", a.chain_weighting);
  av.set_av_parameter(parameters.dump());
  av.resample();

  write_av(av, a.output);
  const IMP::algebra::Vector3D mean = IMP::core::XYZ(av).get_coordinates();
  std::cout << format("%s%d %s: mean position (%.2f, %.2f, %.2f) A", a.chain.c_str(), a.residue,
                      a.atom.c_str(), mean[0], mean[1], mean[2])
            << "\n";
  std::cout << "wrote " << a.output << "\n";
}

// ---- openmm ------------------------------------------------------------------

struct OpenmmArgs {
  std::string fps_json, pdb, score_set, output = "fret_restraints.py", json_path, tether_atom,
                                        force_field = "amber14-all.xml amber14/tip3pfb.xml";
  double f_max = 15.0, tether_k = 30.0;
  int n_steps = 100000;
};

void run_openmm(const OpenmmArgs& a) {
  IMP::set_log_level(IMP::SILENT);
  IMP_NEW(IMP::Model, model, ());
  const IMP::atom::Hierarchy hier = read_structure(a.pdb, model);
  const MDRestraintSystem system =
      md_flat_bottom_restraints(hier, a.fps_json, a.score_set, a.f_max, a.tether_k, 100.0,
                                a.tether_atom);
  write_openmm_script(system, a.output, hier, a.pdb, a.force_field, a.n_steps);
  std::cout << format("%lu probes, %lu FRET wells", (unsigned long)system.get_probes().size(),
                      (unsigned long)system.get_wells().size())
            << (a.score_set.empty() ? std::string() : " from score set " + a.score_set) << "\n";
  std::cout << "wrote " << a.output << "\n";
  if (!a.json_path.empty()) {
    write_openmm_restraints(system, a.json_path, hier);
    std::cout << "wrote " << a.json_path << "\n";
  }
  std::cout << "run it with: python " << basename_of(a.output) << "\n";
}

// ---- build-system ------------------------------------------------------------

struct BuildSystemArgs {
  std::vector<std::string> components;
  std::string output_cif;
  double bond_k = 2000.0, angle_k = 400.0, pi_dihedral_k = 12.0, linker_dihedral_k = 1.5,
         ring_improper_k = 40.0, pi_improper_k = 180.0, flat_improper_k = 120.0,
         orient_improper_k = 220.0, default_radius = 1.7, default_mass = 12.0, nonbonded_k = 5.0,
         nonbonded_cutoff = 6.0;
  int n_steps = 500000, write_every = 1000, minimize_steps = 200;
};

std::string strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

void run_build_system(const BuildSystemArgs& a) {
  // `name=X,mol2=Y,template=Z,role=fixed|mobile` is this command's own
  // syntax, so it is parsed here; the builder takes typed specs.
  std::vector<FFComponentSpec> specs;
  for (std::size_t i = 0; i < a.components.size(); ++i) {
    const std::string& spec = a.components[i];
    std::map<std::string, std::string> parts;
    std::size_t start = 0;
    while (start <= spec.size()) {
      std::size_t comma = spec.find(',', start);
      if (comma == std::string::npos) comma = spec.size();
      const std::string part = spec.substr(start, comma - start);
      const std::size_t eq = part.find('=');
      if (eq != std::string::npos) parts[strip(part.substr(0, eq))] = strip(part.substr(eq + 1));
      start = comma + 1;
    }
    std::string missing;
    const char* required[] = {"name", "mol2", "role"};
    for (int k = 0; k < 3; ++k) {
      if (!parts.count(required[k])) missing += (missing.empty() ? "" : ", ") + std::string(required[k]);
    }
    if (!missing.empty()) {
      throw SubError("--component " + py_repr(spec) + " is missing " + missing);
    }
    specs.push_back(FFComponentSpec(parts["name"], parts["mol2"],
                                    parts.count("template") ? parts["template"] : std::string(),
                                    parts["role"]));
  }
  const std::string out_dir = path_dirname(path_abs(a.output_cif));
  const ProbeForceFieldSystem system = create_forcefield_system(
      specs, a.bond_k, a.angle_k, a.pi_dihedral_k, a.linker_dihedral_k, a.ring_improper_k,
      a.pi_improper_k, a.flat_improper_k, a.orient_improper_k, a.n_steps, a.write_every,
      a.default_radius, a.default_mass, a.nonbonded_k, a.nonbonded_cutoff, a.minimize_steps,
      out_dir);
  make_directory(out_dir);
  write_probe_forcefield_cif(a.output_cif, system);
  std::cout << "Wrote " << a.output_cif << "\n";
  std::cout << format("sites=%lu bonds=%lu angles=%lu dihedrals=%lu impropers=%lu",
                      (unsigned long)system.get_sites().size(),
                      (unsigned long)system.get_bonds().size(),
                      (unsigned long)system.get_angles().size(),
                      (unsigned long)system.get_dihedrals().size(),
                      (unsigned long)system.get_impropers().size())
            << "\n";
}

// ---- select-pairs --------------------------------------------------------------

struct SelectPairsArgs {
  std::string fps_json, rmf_file, score_set, output_path;
  std::vector<std::string> pdb_files;
  int max_pairs = 10, stride = 1;
  double err = 0.06;
  bool superpose = true, unique_only = true;
};

//! The coordinates of the beads, flat.
void append_coordinates(const std::vector<IMP::core::XYZ>& beads, std::vector<double>& out) {
  for (std::size_t k = 0; k < beads.size(); ++k) {
    const IMP::algebra::Vector3D v = beads[k].get_coordinates();
    out.push_back(v[0]);
    out.push_back(v[1]);
    out.push_back(v[2]);
  }
}

std::vector<IMP::core::XYZ> leaves_of(const IMP::atom::Hierarchy& hier) {
  std::vector<IMP::core::XYZ> beads;
  const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(hier);
  for (std::size_t k = 0; k < leaves.size(); ++k) beads.push_back(IMP::core::XYZ(leaves[k]));
  return beads;
}

void run_select_pairs(const SelectPairsArgs& a) {
  if (a.rmf_file.empty() == a.pdb_files.empty()) {
    throw CLI::ValidationError("give either --rmf or one or more --pdb");
  }
  IMP_NEW(IMP::Model, model, ());
  IMP::set_log_level(IMP::SILENT);

  // The ensemble, and the beads the RMSD is measured over. They are collected
  // before the restraint decorates the hierarchy, so the accessible volumes
  // it adds -- which move with the dye, not with the fold -- stay out of it.
  IMP::atom::Hierarchy hier;
  std::vector<IMP::core::XYZ> beads;
  std::vector<std::vector<double> > stack;   // --pdb: one flat frame per file
  std::string label;
  std::size_t n_frames = 0;
#if IMPBFF_CLI_HAS_IMP_RMF
  RMF::FileConstHandle handle;
  RMF::FrameIDs frames;
#endif
  if (!a.rmf_file.empty()) {
#if IMPBFF_CLI_HAS_IMP_RMF
    handle = RMF::open_rmf_file_read_only(a.rmf_file);
    const IMP::atom::Hierarchies hierarchies = IMP::rmf::create_hierarchies(handle, model);
    if (hierarchies.empty()) throw SubError("no hierarchies in " + a.rmf_file);
    hier = hierarchies[0];
    IMP::rmf::load_frame(handle, RMF::FrameID(0));
    beads = leaves_of(hier);
    const RMF::FrameIDs all = handle.get_root_frames();
    for (std::size_t i = 0; i < all.size(); i += static_cast<std::size_t>(a.stride)) {
      frames.push_back(all[i]);
    }
    n_frames = frames.size();
    label = format("%lu frames of %s", (unsigned long)n_frames, basename_of(a.rmf_file).c_str());
#else
    throw SubError("reading an RMF ensemble needs IMP.rmf, which this build does not link; "
                   "pass the structures as --pdb");
#endif
  } else {
    // One hierarchy per file: the restraint is built on the first, and each
    // further structure is read into its coordinates.
    hier = read_structure(a.pdb_files[0], model);
    beads = leaves_of(hier);
    std::vector<std::vector<double> > all;
    for (std::size_t f = 0; f < a.pdb_files.size(); ++f) {
      IMP_NEW(IMP::Model, other, ());
      const IMP::atom::Hierarchy h = read_structure(a.pdb_files[f], other);
      std::vector<double> xyz;
      append_coordinates(leaves_of(h), xyz);
      if (xyz.size() / 3 != beads.size()) {
        throw SubError(format("%s has %lu atoms, %s has %lu -- the structures must correspond",
                              a.pdb_files[f].c_str(), (unsigned long)(xyz.size() / 3),
                              a.pdb_files[0].c_str(), (unsigned long)beads.size()));
      }
      all.push_back(xyz);
    }
    for (std::size_t i = 0; i < all.size(); i += static_cast<std::size_t>(a.stride)) {
      stack.push_back(all[i]);
    }
    n_frames = stack.size();
    label = format("%lu PDB structures", (unsigned long)n_frames);
  }

  if (n_frames < 2) {
    throw SubError("pair selection needs an ensemble; one structure separates nothing");
  }

  IMP::Pointer<ProbeNetworkRestraint> restraint =
      new ProbeNetworkRestraint(hier, a.fps_json, "ProbeNetworkRestraint%1%", a.score_set);
  const std::vector<std::string> pair_names = restraint->get_pair_names();
  const std::map<std::string, AVPairDistanceMeasurement> used = restraint->get_used_distances();
  if (pair_names.empty()) throw SubError("no candidate pairs in " + a.fps_json);

  std::cout << "Ensemble:   " << label << "\n";
  std::cout << format("Candidates: %lu pairs", (unsigned long)pair_names.size())
            << (a.score_set.empty() ? std::string() : " from score set " + a.score_set) << "\n";

  const std::size_t n_pairs = pair_names.size();
  std::vector<double> effs;
  effs.reserve(n_frames * n_pairs);
  std::vector<double> coords;
  coords.reserve(n_frames * beads.size() * 3);
  for (std::size_t i = 0; i < n_frames; ++i) {
#if IMPBFF_CLI_HAS_IMP_RMF
    if (!a.rmf_file.empty()) IMP::rmf::load_frame(handle, frames[i]);
#endif
    if (a.rmf_file.empty()) {
      for (std::size_t k = 0; k < beads.size(); ++k) {
        beads[k].set_coordinates(IMP::algebra::Vector3D(stack[i][3 * k], stack[i][3 * k + 1],
                                                        stack[i][3 * k + 2]));
      }
    }
    const std::vector<double> e = restraint->get_pair_efficiencies();
    if (e.size() != n_pairs) {
      throw SubError(format("%lu efficiencies for %lu pairs", (unsigned long)e.size(),
                            (unsigned long)n_pairs));
    }
    effs.insert(effs.end(), e.begin(), e.end());
    append_coordinates(beads, coords);
  }

  std::string bad;
  for (std::size_t j = 0; j < n_pairs; ++j) {
    for (std::size_t i = 0; i < n_frames; ++i) {
      if (!std::isfinite(effs[i * n_pairs + j])) {
        bad += (bad.empty() ? "" : ", ") + pair_names[j];
        break;
      }
    }
  }
  if (!bad.empty()) {
    throw SubError("non-finite efficiencies for " + bad +
                   " -- the selector needs every candidate scored in every frame");
  }

  std::cout << "Pairwise RMSD...\n" << std::flush;
  double* rmsd_view = nullptr;
  int n1 = 0, n2 = 0;
  pairwise_rmsd(&coords[0], static_cast<int>(n_frames), static_cast<int>(beads.size()), 3,
                a.superpose, &rmsd_view, &n1, &n2);
  std::vector<double> rmsds(rmsd_view, rmsd_view + static_cast<std::size_t>(n1) * n2);
  std::free(rmsd_view);

  int* selected_view = nullptr;
  int n_selected = 0;
  double* decay_view = nullptr;
  int n_decay = 0;
  select_probe_pairs(&effs[0], static_cast<int>(n_frames), static_cast<int>(n_pairs), &rmsds[0],
                     n1, n2, a.err, a.max_pairs, a.unique_only, 0.99, &selected_view, &n_selected,
                     &decay_view, &n_decay);
  const std::vector<int> selected(selected_view, selected_view + n_selected);
  const std::vector<double> decay(decay_view, decay_view + n_decay);
  std::free(selected_view);
  std::free(decay_view);

  // what the expectation is before anything has been measured, so the first
  // selection has something to be a gain over
  const double start = expected_rmsd(rmsds, std::vector<double>(rmsds.size(), 0.0), 1, 0.99,
                                     static_cast<int>(n_frames));

  std::cout << "\n";
  std::cout << format("%3s  %-16s  %13s  %8s", "#", "pair", "expected RMSD", "gain") << "\n";
  std::cout << format("%3s  %-16s  %11.3f A  %8s", "-", "(no data)", start, "") << "\n";
  std::string tsv = "rank\tpair\tposition_1\tposition_2\tforster_radius\texpected_rmsd\tgain\n";
  double previous = start;
  const std::size_t n_rows = std::min(selected.size(), decay.size());
  for (std::size_t r = 0; r < n_rows; ++r) {
    const std::string& name = pair_names[static_cast<std::size_t>(selected[r])];
    const AVPairDistanceMeasurement& m = used.at(name);
    const double value = decay[r];
    const double gain = previous - value;
    std::cout << format("%3lu  %-16s  %11.3f A  %6.3f A", (unsigned long)(r + 1), name.c_str(),
                        value, gain)
              << "\n";
    tsv += format("%lu", (unsigned long)(r + 1)) + "\t" + name + "\t" + m.position_1 + "\t" +
           m.position_2 + "\t" + json_float(m.forster_radius) + "\t" + json_float(value) + "\t" +
           json_float(gain) + "\n";
    previous = value;
  }
  if (!a.output_path.empty()) {
    std::ofstream out(a.output_path.c_str(), std::ios::binary);
    if (!out) throw SubError(a.output_path + ": cannot write");
    out << tsv;
    out.close();
    std::cout << "\nwrote " << a.output_path << "\n";
  }
}

// ---- rotamer ----------------------------------------------------------------------

struct PredictArgs {
  std::string protein, donor = "AlexaFluor 488", acceptor = "AlexaFluor 594", libname_1,
                       libname_2, output_prefix = "rotamer_fret";
  std::vector<int> residues;
  std::vector<std::string> chains;
  double temperature = 300.0, r0 = 54.0, z_cutoff = 0.05;
  bool electrostatic = false, fixed_r0 = false;
  int max_frames = -1;
};

void run_predict(const PredictArgs& a) {
  if (a.residues.size() != 2) {
    throw CLI::ValidationError("--residue must be provided exactly twice");
  }
  if (!a.chains.empty() && a.chains.size() != 2) {
    throw CLI::ValidationError("--chain must be omitted or provided exactly twice");
  }
  // omitted chains match any chain: the library's empty list
  FRETRotamer fret(a.protein, a.residues, a.chains, a.donor, a.acceptor, a.libname_1, a.libname_2,
                   a.temperature, a.electrostatic, "lj", true, 0.5, 1.0, a.fixed_r0, a.r0, "",
                   a.z_cutoff, a.output_prefix, a.max_frames);
  fret.run();
  std::cout << "Wrote FRET prediction files with prefix " << py_repr(a.output_prefix) << "\n";
}

}  // namespace modelling

void add_modelling_subs(CLI::App& app) {
  // ---- av-export
  {
    std::shared_ptr<modelling::AvExportArgs> a = std::make_shared<modelling::AvExportArgs>();
    CLI::App* sub = app.add_subcommand("av-export", R"doc(Compute one accessible volume and write it in a viewer's format.)doc");
    sub->footer(R"doc(`.xyz` and `.pqr` write the point cloud (the weight rides in the fifth
column, and in PQR's charge column); `.dx` and `.mrc` write the density
grid. PyMOL and VMD open all four.)doc");
    sub->add_option("-p,--pdb", a->pdb, "Structure to label.")->required();
    sub->add_option("-c,--chain", a->chain, "Chain of the site.")->required();
    sub->add_option("-r,--residue", a->residue, "Residue number of the site.")->required();
    sub->add_option("-a,--atom", a->atom, "Attachment atom.")->capture_default_str();
    sub->add_option("-o,--output", a->output,
                    "Output file; the extension picks the format (.xyz, .pqr, .dx, .mrc).")
        ->required();
    sub->add_option("--linker-length", a->linker_length, "Linker length, A.")->capture_default_str();
    sub->add_option("--linker-width", a->linker_width, "Linker width, A.")->capture_default_str();
    sub->add_option("--radius1", a->radius1, "Dye radius, A.")->capture_default_str();
    sub->add_option("--grid", a->grid, "Simulation grid resolution, A.")->capture_default_str();
    sub->add_flag("--chain-weighting,!--no-chain-weighting", a->chain_weighting,
                  "Weight grid points by linker chain statistics (default: off). Read "
                  "okf/validation/chain_weighting.md before using it.");
    sub->callback([a] {
      set_current_sub("av-export");
      modelling::run_av_export(*a);
    });
  }

  // ---- openmm
  {
    std::shared_ptr<modelling::OpenmmArgs> a = std::make_shared<modelling::OpenmmArgs>();
    CLI::App* sub = app.add_subcommand("openmm", R"doc(Write a runnable OpenMM script with FRET restraints from an fps.json.)doc");
    sub->footer(R"doc(Each measured distance becomes a flat-bottom well between two probe
particles: zero inside the experimental error bars, harmonic outside them,
linear past that so the force is capped. The measured distances are
converted to mean-position bounds against the accessible volumes actually
built on this structure, which is not a constant offset.

The script is self-contained -- the restraint table is embedded -- and
OpenMM does not have to be installed to write it.)doc");
    sub->add_option("-j,--fps-json", a->fps_json, "fps.json labelling and distance file.")
        ->required();
    sub->add_option("-p,--pdb", a->pdb, "Structure the volumes are built on and the script loads.")
        ->required();
    sub->add_option("-s,--score-set", a->score_set, "Which distances to restrain (default: all).");
    sub->add_option("-o,--output", a->output, "Where to write the runnable script.")
        ->capture_default_str();
    sub->add_option("--json", a->json_path,
                    "Also write the restraint table as JSON, for another engine.");
    sub->add_option("--f-max", a->f_max, "Force at one error bar, kcal/mol/A, before conversion.")
        ->capture_default_str();
    sub->add_option("--tether-k", a->tether_k,
                    "Force constant holding a probe to its labelling site.")
        ->capture_default_str();
    sub->add_option("--tether-atom", a->tether_atom,
                    "Tether probes to this atom of the labelled residue (e.g. CA); empty uses the "
                    "fps.json attachment atom.");
    sub->add_option("-n,--n-steps", a->n_steps, "Steps the generated script runs.")
        ->capture_default_str();
    sub->add_option("--force-field", a->force_field, "OpenMM force-field XMLs.")
        ->capture_default_str();
    sub->callback([a] {
      set_current_sub("openmm");
      modelling::run_openmm(*a);
    });
  }

  // ---- build-system
  {
    std::shared_ptr<modelling::BuildSystemArgs> a = std::make_shared<modelling::BuildSystemArgs>();
    CLI::App* sub = app.add_subcommand("build-system", R"doc(Build a force-field system from component MOL2 files and templates.)doc");
    sub->footer(R"doc(Note this is not the same builder as
`IMP.bff.cgprobe.topology.create_probe_protein_system`: on the same two
components they agree on sites, bonds, angles and dihedrals and differ on
impropers, which this one builds (81) and that one leaves empty. See
okf/validation/impropers_are_dropped.md.)doc");
    sub->add_option("--component", a->components,
                    "Component spec: name=X,mol2=Y,template=Z,role=fixed|mobile (repeatable)")
        ->required();
    sub->add_option("--output-cif", a->output_cif, "Where the system CIF goes.")->required();
    sub->add_option("--bond-k", a->bond_k)->capture_default_str();
    sub->add_option("--angle-k", a->angle_k)->capture_default_str();
    sub->add_option("--pi-dihedral-k", a->pi_dihedral_k)->capture_default_str();
    sub->add_option("--linker-dihedral-k", a->linker_dihedral_k)->capture_default_str();
    sub->add_option("--ring-improper-k", a->ring_improper_k)->capture_default_str();
    sub->add_option("--pi-improper-k", a->pi_improper_k)->capture_default_str();
    sub->add_option("--flat-improper-k", a->flat_improper_k)->capture_default_str();
    sub->add_option("--orient-improper-k", a->orient_improper_k)->capture_default_str();
    sub->add_option("--n-steps", a->n_steps)->capture_default_str();
    sub->add_option("--write-every", a->write_every)->capture_default_str();
    sub->add_option("--default-radius", a->default_radius, "Default site radius in Angstrom.")
        ->capture_default_str();
    sub->add_option("--default-mass", a->default_mass, "Default site mass in Da.")
        ->capture_default_str();
    sub->add_option("--nonbonded-k", a->nonbonded_k)->capture_default_str();
    sub->add_option("--nonbonded-cutoff", a->nonbonded_cutoff, "Nonbonded cutoff in Angstrom.")
        ->capture_default_str();
    sub->add_option("--minimize-steps", a->minimize_steps)->capture_default_str();
    sub->callback([a] {
      set_current_sub("build-system");
      modelling::run_build_system(*a);
    });
  }

  // ---- select-pairs
  {
    std::shared_ptr<modelling::SelectPairsArgs> a = std::make_shared<modelling::SelectPairsArgs>();
    CLI::App* sub = app.add_subcommand("select-pairs", R"doc(Which FRET pairs to measure, in the order they are worth measuring.)doc");
    sub->footer(R"doc(Olga's greedy selection (Dimura et al., Nat. Commun. 11, 5394, 2020) over
the pairs an fps.json file defines and an ensemble of candidate structures.
Each step adds the pair that leaves the smallest expected RMSD between the
true structure and the one the measurements would single out; the reported
decay is that expectation after each addition.)doc");
    sub->add_option("-j,--fps-json", a->fps_json,
                    "fps.json labelling and distance file: the candidate pairs.")
        ->required();
    sub->add_option("-r,--rmf", a->rmf_file,
                    "RMF trajectory holding the ensemble of candidate structures.");
    sub->add_option("-p,--pdb", a->pdb_files,
                    "PDB files, one structure each (repeatable); alternative to --rmf.");
    sub->add_option("-s,--score-set", a->score_set,
                    "fps.json score set naming the candidate pairs (default: all).");
    sub->add_option("-n,--max-pairs", a->max_pairs, "How many pairs to select.")
        ->capture_default_str();
    sub->add_option("-e,--err", a->err,
                    "Expected absolute error of a FRET efficiency measurement.")
        ->capture_default_str();
    sub->add_option("--stride", a->stride, "Use every Nth structure of the ensemble.")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    sub->add_flag("--superpose,!--no-superpose", a->superpose,
                  "Superpose each pair of frames before measuring their RMSD (default).");
    sub->add_flag("--unique,!--repeat", a->unique_only,
                  "A pair may be selected at most once (default).");
    sub->add_option("-o,--output", a->output_path,
                    "Write the ranking to this tab-separated file.");
    sub->callback([a] {
      set_current_sub("select-pairs");
      modelling::run_select_pairs(*a);
    });
  }

  // ---- rotamer
  CLI::App* rotamer = app.add_subcommand("rotamer", R"doc(Rotamer-based cgprobe tools.)doc");
  rotamer->require_subcommand(1);
  {
    std::shared_ptr<modelling::PredictArgs> a = std::make_shared<modelling::PredictArgs>();
    CLI::App* sub = rotamer->add_subcommand("predict", R"doc(Run rotamer-based FRET prediction.)doc");
    sub->add_option("--protein", a->protein, "Protein PDB or RMF path.")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("--residue", a->residues, "Placement residue number. Repeat twice.")
        ->required();
    sub->add_option("--chain", a->chains, "Placement chain ID. Repeat twice.");
    sub->add_option("--donor", a->donor, "Donor dye name.")->capture_default_str();
    sub->add_option("--acceptor", a->acceptor, "Acceptor dye name.")->capture_default_str();
    sub->add_option("--libname-1", a->libname_1, "Donor rotamer library name or RMF path.")
        ->required();
    sub->add_option("--libname-2", a->libname_2, "Acceptor rotamer library name or RMF path.")
        ->required();
    sub->add_option("--temperature", a->temperature, "Temperature in K.")->capture_default_str();
    sub->add_flag("--electrostatic", a->electrostatic, "Include Debye-Huckel electrostatics.");
    sub->add_flag("--fixed-r0", a->fixed_r0, "Use a fixed Förster radius.");
    sub->add_option("--r0", a->r0, "Fixed R0 in Angstrom.")->capture_default_str();
    sub->add_option("--output-prefix", a->output_prefix, "Output prefix.")->capture_default_str();
    sub->add_option("--z-cutoff", a->z_cutoff, "Partition-function cutoff.")->capture_default_str();
    sub->add_option("--max-frames", a->max_frames,
                    "Maximum protein frames to process (default: all).");
    sub->callback([a] {
      set_current_sub("rotamer predict");
      modelling::run_predict(*a);
    });
  }
  {
    struct R0Args {
      std::string donor, acceptor;
      double k2 = 0.0;
    };
    std::shared_ptr<R0Args> a = std::make_shared<R0Args>();
    CLI::App* sub = rotamer->add_subcommand("r0", R"doc(Calculate a Förster radius.)doc");
    sub->add_option("--donor", a->donor, "Donor dye name.")->required();
    sub->add_option("--acceptor", a->acceptor, "Acceptor dye name.")->required();
    sub->add_option("--k2", a->k2, "Orientation factor.")->required();
    sub->callback([a] {
      set_current_sub("rotamer r0");
      std::cout << format("%.6f A", forster_radius_from_spectra(a->donor, a->acceptor, a->k2))
                << "\n";
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

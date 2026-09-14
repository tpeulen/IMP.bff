/** \file CommandLineDye.cpp
 *  \brief `imp_bff dye ...`: explicit-dye labelling, rotamer libraries and
 *         sampling.
 *
 *  Port of the `dye` group of the former `bin/imp_bff`. The engines are
 *  ProbeAttachment.h (placement, and the collision-gated DOF walk),
 *  ProbeDynamics.h (Langevin/Brownian dynamics), Linker.h (library
 *  generation), Clustering.h (the jump-count analyses), SequenceAlignment.h
 *  (fluorescent-protein detection) and RmfIO.h; this file is the grammar,
 *  the file lookup the program did, and the reports. Help texts are the
 *  Python program's docstrings, verbatim.
 *
 *  RMF trajectories are written with #IMP::bff::RmfStructureWriter (RMF's
 *  own decorators, PMI's root/chain/residue/atom shape) and not through
 *  `IMP::rmf`, which the connection layer's IMP does not carry.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <IMP/Model.h>
#include <IMP/Particle.h>
#include <IMP/atom/Atom.h>
#include <IMP/atom/Chain.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/Mass.h>
#include <IMP/atom/Residue.h>
#include <IMP/atom/mol2.h>
#include <IMP/atom/pdb.h>
#include <IMP/core/XYZ.h>
#include <IMP/core/XYZR.h>

#include <IMP/bff/Clustering.h>
#include <IMP/bff/IMPHierarchyBridge.h>
#include <IMP/bff/Linker.h>
#include <IMP/bff/ProbeAttachment.h>
#include <IMP/bff/ProbeDataPaths.h>
#include <IMP/bff/ProbeDynamics.h>
#include <IMP/bff/ProbeSampling.h>
#include <IMP/bff/RmfIO.h>
#include <IMP/bff/SequenceAlignment.h>
#include <IMP/bff/StructureTable.h>
#include <IMP/bff/internal/Text.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace dye {

//! A path as pathlib spells `Path(dir) / name`.
std::string join(const std::string& dir, const std::string& name) {
  std::string d = dir;
  while (d.size() > 1 && (d[d.size() - 1] == '/' || d[d.size() - 1] == '\\')) d.erase(d.size() - 1);
  return path_join(d, name);
}

std::string lower(std::string s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  }
  return s;
}

std::string upper(std::string s) {
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
  }
  return s;
}

//! pathlib's `stem`.
std::string stem(const std::string& path) { return path_with_suffix(basename_of(path), ""); }

//! Directory of the bundled rotamer library, as module data.
std::string rotamer_library_dir() { return get_data_path("rotamer_library"); }

//! What the program said when it could not fetch a structure.
/*! Downloading is not compiled: there is no HTTP client in the library. The
    message says where to put the file so the next run finds it. */
void report_no_download(const std::string& pdb_id, const std::string& target) {
  std::cout << "Error downloading PDB: downloading is not compiled into imp_bff; save "
            << "https://files.rcsb.org/download/" << upper(pdb_id) << ".pdb as " << target
            << "\n";
}

//! A protein PDB id or path, resolved to an existing file.
std::string resolve_protein_pdb(const std::string& pdb_id_or_path) {
  if (file_exists(pdb_id_or_path)) return pdb_id_or_path;
  const std::string pdb_path = get_structure_dir(upper(pdb_id_or_path) + ".pdb");
  if (file_exists(pdb_path)) return pdb_path;
  ensure_dir(get_structure_dir());
  report_no_download(pdb_id_or_path, pdb_path);
  throw SubError("Unable to find or download PDB file for " + pdb_id_or_path);
}

//! Where the probe's structure is: a rotamer RMF, a library PDB or a MOL2.
std::pair<std::string, std::string> find_probe_structure(const std::string& probe_name,
                                                         const std::string& linker_type) {
  const std::string templates_dir = get_template_dir("rotamer");
  static const char* const mapping[][2] = {
      {"alexa488", "A48"}, {"alexa350", "A35"}, {"alexa532", "A53"}, {"alexa568", "A56"},
      {"alexa594", "A59"}, {"alexa647", "A64"}, {"atto390", "T39"},  {"atto425", "T42"},
      {"atto465", "T46"},  {"atto488", "T48"},  {"atto495", "T49"},  {"atto520", "T52"},
      {"atto610", "T61"},  {"atto655", "T65"}};
  std::string clean;
  for (std::size_t i = 0; i < probe_name.size(); ++i) {
    const char ch = probe_name[i];
    if (ch == ' ' || ch == '-') continue;
    clean += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  std::string probe_id = probe_name;
  for (std::size_t i = 0; i < sizeof(mapping) / sizeof(mapping[0]); ++i) {
    if (clean == mapping[i][0]) probe_id = mapping[i][1];
  }

  std::vector<std::string> patterns;
  if (!linker_type.empty()) patterns.push_back(probe_id + "_" + linker_type + ".rmf3");
  patterns.push_back(probe_id + "_C1R.rmf3");
  patterns.push_back(probe_id + "_C2R.rmf3");
  patterns.push_back(probe_id + "_L1R.rmf3");
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    const std::string f = join(templates_dir, patterns[i]);
    if (file_exists(f)) return std::make_pair(f, std::string("rmf"));
  }

  const std::string lib_dir = rotamer_library_dir();
  if (file_exists(lib_dir)) {
    std::vector<std::string> pdb_patterns;
    if (!linker_type.empty()) pdb_patterns.push_back(probe_id + "_" + linker_type + ".pdb");
    pdb_patterns.push_back(probe_id + "_C1R.pdb");
    pdb_patterns.push_back(probe_id + "_C2R.pdb");
    pdb_patterns.push_back(probe_id + "_L1R.pdb");
    for (std::size_t i = 0; i < pdb_patterns.size(); ++i) {
      const std::string f = join(lib_dir, pdb_patterns[i]);
      if (file_exists(f)) return std::make_pair(f, std::string("pdb"));
    }
  }

  const std::string structures_dir = get_structure_dir();
  std::vector<std::string> mol2_patterns;
  if (!linker_type.empty()) mol2_patterns.push_back(probe_name + "_" + linker_type + ".mol2");
  mol2_patterns.push_back(probe_name + ".mol2");
  if (lower(probe_name).find("488") != std::string::npos) mol2_patterns.push_back("alexa488_r48.mol2");
  if (lower(probe_name).find("655") != std::string::npos ||
      lower(probe_name).find("atto655") != std::string::npos) {
    mol2_patterns.push_back("atto655.mol2");
  }
  for (std::size_t i = 0; i < mol2_patterns.size(); ++i) {
    const std::string f = join(structures_dir, mol2_patterns[i]);
    if (file_exists(f)) return std::make_pair(f, std::string("mol2"));
  }
  return std::make_pair(std::string(), std::string());
}

//! The MOL2 of a probe, by name; the Alexa 488 one as the last resort.
std::string find_probe_mol2(const std::string& probe_name) {
  const std::string structures_dir = get_structure_dir();
  std::string clean;
  for (std::size_t i = 0; i < probe_name.size(); ++i) {
    const char ch = probe_name[i];
    if (ch == ' ' || ch == '-') continue;
    clean += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  std::string target = probe_name;
  if (clean == "alexa488") target = "alexa488_r48";
  if (clean == "atto655") target = "atto655";
  const std::string candidates[2] = {target + ".mol2", probe_name + ".mol2"};
  for (int i = 0; i < 2; ++i) {
    const std::string f = join(structures_dir, candidates[i]);
    if (file_exists(f)) return f;
  }
  return join(structures_dir, "alexa488_r48.mol2");
}

//! A new particle decorated as a hierarchy node.
IMP::atom::Hierarchy new_node(IMP::Model* model, const std::string& name) {
  return IMP::atom::Hierarchy::setup_particle(new IMP::Particle(model, name));
}

//! Every leaf of a hierarchy as the columns an RMF writer needs, in leaf order.
StructureTable table_from_leaves(const IMP::atom::Hierarchies& leaves) {
  StructureTable table;
  const IMP::atom::ElementTable& elements = IMP::atom::get_element_table();
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    IMP::Particle* p = leaves[i].get_particle();
    const IMP::algebra::Vector3D v = IMP::core::XYZ(p).get_coordinates();
    table.xyz.push_back(v[0]);
    table.xyz.push_back(v[1]);
    table.xyz.push_back(v[2]);
    table.radius.push_back(IMP::core::XYZR::get_is_setup(p) ? IMP::core::XYZR(p).get_radius() : 1.5);
    std::string atom_name = p->get_name(), element;
    double mass = 0.0;
    if (IMP::atom::Atom::get_is_setup(p)) {
      IMP::atom::Atom atom(p);
      atom_name = atom.get_atom_type().get_string();
      if (atom_name.compare(0, 5, "HET: ") == 0) atom_name = atom_name.substr(5);
      element = elements.get_name(atom.get_element());
      mass = elements.get_mass(atom.get_element());
    }
    if (IMP::atom::Mass::get_is_setup(p)) mass = IMP::atom::Mass(p).get_mass();
    table.mass.push_back(mass);
    table.bfactor.push_back(0.0);
    table.atom_id.push_back(static_cast<int>(i) + 1);
    table.atom_name.push_back(trimmed(atom_name));
    table.element.push_back(element);
    int res_id = 0;
    std::string res_name, chain_id;
    IMP::atom::Hierarchy parent = leaves[i].get_parent();
    if (parent && IMP::atom::Residue::get_is_setup(parent)) {
      IMP::atom::Residue residue(parent);
      res_id = residue.get_index();
      res_name = residue.get_residue_type().get_string();
      IMP::atom::Hierarchy chain = parent.get_parent();
      if (chain && IMP::atom::Chain::get_is_setup(chain)) chain_id = IMP::atom::Chain(chain).get_id();
    }
    table.res_id.push_back(res_id);
    table.res_name.push_back(res_name);
    table.chain.push_back(chain_id);
  }
  return table;
}

//! The current coordinates of \p leaves, flat.
std::vector<double> leaf_coordinates(const IMP::atom::Hierarchies& leaves) {
  std::vector<double> out;
  out.reserve(leaves.size() * 3);
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const IMP::algebra::Vector3D v = IMP::core::XYZ(leaves[i]).get_coordinates();
    out.push_back(v[0]);
    out.push_back(v[1]);
    out.push_back(v[2]);
  }
  return out;
}

//! One rotamer's coordinates out of a library, flat.
std::vector<double> rotamer(const ProbeRotamerLibrary& lib, int index) {
  const std::size_t per = 3 * static_cast<std::size_t>(lib.n_atoms);
  const std::size_t start = per * static_cast<std::size_t>(index);
  if (start + per > lib.coords.size()) {
    throw SubError(format("rotamer %d is not in a library of %d", index, lib.n_rotamers));
  }
  return std::vector<double>(lib.coords.begin() + start, lib.coords.begin() + start + per);
}

void make_parent_dir(const std::string& output) {
  const std::string dir = path_dirname(path_abs(output));
  if (!dir.empty()) make_directory(dir);
}

IMP::atom::Hierarchy read_pdb(const std::string& path, IMP::Model* model, IMP::atom::PDBSelector* s) {
  return IMP::atom::read_pdb(path, model, s);
}

// ---- label -------------------------------------------------------------------

struct LabelArgs {
  std::string pdb, chain = "A", dye, linker, output;
  int residue = 0;
};

void run_label(const LabelArgs& a) {
  IMP_NEW(IMP::Model, model, ());
  std::string pdb_path, pdb_name;
  if (file_exists(a.pdb)) {
    pdb_path = a.pdb;
    pdb_name = stem(pdb_path);
  } else {
    pdb_name = upper(a.pdb);
    pdb_path = get_structure_dir(pdb_name + ".pdb");
    if (!file_exists(pdb_path)) {
      ensure_dir(get_structure_dir());
      report_no_download(a.pdb, pdb_path);
      throw SubExit(1);
    }
  }
  IMP::atom::Hierarchy protein = read_pdb(pdb_path, model, new IMP::atom::NonWaterPDBSelector());
  const std::pair<std::string, std::string> found = find_probe_structure(a.dye, a.linker);
  if (found.first.empty()) {
    std::cout << "Error: No structure found for " << a.dye << " "
              << (a.linker.empty() ? std::string("None") : a.linker) << "\n";
    throw SubExit(1);
  }
  std::cout << "Using: " << found.first << "\n";
  IMP::atom::Hierarchy dye_hier;
  if (found.second == "rmf") {
    const ProbeRotamerLibrary lib = read_rotamer_library_rmf(found.first);
    const std::string struct_stem = stem(found.first);
    const std::string probe_id = struct_stem.substr(0, struct_stem.find('_'));
    std::string base_pdb = find_probe_structure(probe_id, a.linker).first;
    if (base_pdb.empty() || !ends_with(base_pdb, ".pdb")) {
      base_pdb = join(rotamer_library_dir(), struct_stem + ".pdb");
    }
    if (file_exists(base_pdb)) {
      dye_hier = read_pdb(base_pdb, model, new IMP::atom::AllPDBSelector());
    } else {
      dye_hier = IMP::atom::read_mol2(get_structure_dir("alexa488_r48.mol2"), model);
    }
    apply_coordinates(dye_hier, rotamer(lib, 0));
  } else if (found.second == "pdb") {
    dye_hier = read_pdb(found.first, model, new IMP::atom::AllPDBSelector());
  } else {
    dye_hier = IMP::atom::read_mol2(found.first, model);
  }
  attach_probes(protein, std::vector<ProbeAttachment>(1, ProbeAttachment(dye_hier, a.chain, a.residue)),
                true);
  IMP::atom::Hierarchy root = new_node(model, "root");
  root.add_child(protein);
  root.add_child(dye_hier);
  const std::string output = a.output.empty()
                                 ? format("output/test_systems/%s_%s_%d.pdb", pdb_name.c_str(),
                                          a.dye.c_str(), a.residue)
                                 : a.output;
  make_parent_dir(output);
  IMP::atom::write_pdb(root, output);
  std::cout << "Saved to " << output << "\n";
}

// ---- build-lib ---------------------------------------------------------------

struct BuildLibArgs {
  int n_steps = 1000;
  double cluster_threshold = 0.5;
  std::string output_dir;
};

void run_build_lib(const BuildLibArgs& a) {
  const std::string output_dir = a.output_dir.empty() ? get_template_dir("rotamer") : a.output_dir;
  const std::vector<std::string> mol2_files = directory_entries(get_structure_dir(), ".mol2");
  make_directory(output_dir);
  for (std::size_t i = 0; i < mol2_files.size(); ++i) {
    const std::string name = basename_of(mol2_files[i]);
    if (name.find("1DG3") != std::string::npos) continue;
    std::string probe_name = stem(name);
    const std::size_t r48 = probe_name.find("_r48");
    if (r48 != std::string::npos) probe_name.erase(r48, 4);
    try {
      const ProbeRotamerLibrary lib =
          generate_linker_rotamers(mol2_files[i], a.n_steps, 10, 0.1, 0.02, a.cluster_threshold);
      write_rotamer_library_rmf(path_join(output_dir, probe_name + ".rmf3"), lib);
      std::cout << "Built " << probe_name << " (" << lib.n_rotamers << " rotamers)\n";
    } catch (const std::exception& e) {
      std::cout << "Failed " << probe_name << ": " << e.what() << "\n";
    }
  }
}

// ---- analyze-tc --------------------------------------------------------------

struct AnalyzeTcArgs {
  std::string input_rmf;
  double dt = 1.0;
  bool show_all = false;
};

void run_analyze_tc(const AnalyzeTcArgs& a) {
  const ProbeRotamerLibrary lib = read_rotamer_library_rmf(a.input_rmf);
  if (lib.transitions.empty()) {
    std::cout << "No transition matrix found.\n";
    return;
  }
  const std::vector<double> counts(lib.transitions.begin(), lib.transitions.end());
  const double tc = slowest_relaxation_time(counts, lib.n_rotamers, a.dt);
  std::cout << format("Slowest TC: %.2f ps", tc) << "\n";
  if (a.show_all) {
    double* view = nullptr;
    int n = 0;
    relaxation_times(counts, lib.n_rotamers, a.dt, &view, &n);
    for (int i = 0; i < n; ++i) {
      if (view[i] > 0) std::cout << format("  tau_%d: %.2f", i + 2, view[i]) << "\n";
    }
    std::free(view);
  }
}

// ---- reconstruct -------------------------------------------------------------

struct ReconstructArgs {
  std::string lib_rmf, output_rmf;
  int n_frames = 100, seed = 42;
};

void run_reconstruct(const ReconstructArgs& a) {
  const ProbeRotamerLibrary lib = read_rotamer_library_rmf(a.lib_rmf);
  // the library carries the jump counts the walk is drawn from, flat
  const std::vector<double> counts(lib.transitions.begin(), lib.transitions.end());
  const std::vector<int> indices =
      markov_state_trajectory(counts, lib.n_rotamers, lib.weights, a.n_frames, -1, a.seed);
  IMP_NEW(IMP::Model, model, ());
  IMP::atom::Hierarchy root = new_node(model, "reconstructed");
  IMP::atom::Residue res = IMP::atom::Residue::setup_particle(
      new IMP::Particle(model, "DYE"), IMP::atom::ResidueType("DYE"), 1);
  root.add_child(res);
  for (std::size_t i = 0; i < lib.atom_names.size(); ++i) {
    IMP::Particle* p = new IMP::Particle(model, lib.atom_names[i]);
    IMP::atom::Atom::setup_particle(p, IMP::atom::AtomType("C"));
    IMP::core::XYZR::setup_particle(p).set_radius(1.0);
    res.add_child(IMP::atom::Hierarchy(p));
  }
  const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(root);
  RmfStructureWriter writer(a.output_rmf, table_from_leaves(leaves), "reconstructed");
  for (std::size_t i = 0; i < indices.size(); ++i) {
    writer.append(rotamer(lib, indices[i]), format("step_%lu", (unsigned long)i));
  }
  writer.close();
  std::cout << "Saved to " << a.output_rmf << "\n";
}

// ---- sample-rotamer ----------------------------------------------------------

struct SampleRotamerArgs {
  std::string protein_pdb, chain = "A", dye, output_rmf;
  int residue = 0, n_samples = 100;
};

void run_sample_rotamer(const SampleRotamerArgs& a) {
  const std::string lib_path = find_probe_structure(a.dye, "").first;
  const ProbeRotamerLibrary lib = read_rotamer_library_rmf(lib_path);
  IMP_NEW(IMP::Model, model, ());
  IMP::atom::Hierarchy protein = read_pdb(a.protein_pdb, model, new IMP::atom::NonWaterPDBSelector());
  IMP::atom::Hierarchy dye_hier = IMP::atom::read_mol2(find_probe_mol2(a.dye), model);
  const IMP::atom::Hierarchies dye_leaves = IMP::atom::get_leaves(dye_hier);
  for (std::size_t i = 0; i < dye_leaves.size(); ++i) {
    if (!IMP::core::XYZR::get_is_setup(dye_leaves[i])) {
      IMP::core::XYZR::setup_particle(dye_leaves[i].get_particle(), 1.0);
    }
  }
  attach_probes(protein, std::vector<ProbeAttachment>(1, ProbeAttachment(dye_hier, a.chain, a.residue)),
                true);
  // `resolve_probe_site` answers CA, N, C in that order
  const IMP::ParticlesTemp site = resolve_probe_site(protein, a.chain, a.residue);
  const IMP::algebra::Vector3D ca = IMP::core::XYZ(site[0]).get_coordinates();
  const IMP::algebra::Vector3D n = IMP::core::XYZ(site[1]).get_coordinates();
  const IMP::algebra::Vector3D c = IMP::core::XYZ(site[2]).get_coordinates();

  IMP::atom::Hierarchy root = new_node(model, "root");
  root.add_child(protein);
  root.add_child(dye_hier);
  const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(root);
  RmfStructureWriter writer(a.output_rmf, table_from_leaves(leaves), "root");
  // one seed per frame keeps the run reproducible
  for (int i = 0; i < a.n_samples; ++i) {
    const int ridx = sample_weighted_index(lib.weights, 42 + i);
    apply_coordinates(dye_hier, rotamer(lib, ridx));
    place_probe_from_coords(dye_hier, ca, n, c);
    writer.append(leaf_coordinates(leaves), format("%d", i));
  }
  writer.close();
  std::cout << format("Wrote %d frames to %s", a.n_samples, a.output_rmf.c_str()) << "\n";
}

// ---- sample-dof-walk ---------------------------------------------------------

struct DofWalkArgs {
  std::string protein_pdb, chain = "A", dye, linker, output_rmf;
  int residue = 0, n_steps = 1000, seed = 42;
};

void run_dof_walk(const DofWalkArgs& a) {
  const std::string protein_path = resolve_protein_pdb(a.protein_pdb);
  IMP_NEW(IMP::Model, model, ());
  IMP::atom::Hierarchy protein = read_pdb(protein_path, model, new IMP::atom::NonWaterPDBSelector());

  // The geometry and the atoms have to be the same MOL2, in the same order:
  // `LinkerGeometry` indexes the coordinate array it was built from. The
  // residue wrapper is what an RMF needs to write it.
  const std::string mol2_path = file_exists(a.dye) ? a.dye : find_probe_mol2(a.dye);
  IMP::atom::Hierarchy raw_hier = IMP::atom::read_mol2(mol2_path, model);
  IMP::atom::Hierarchy dye_hier = new_node(model, "DYE_ROOT");
  IMP::atom::Chain d_chain = IMP::atom::Chain::setup_particle(new IMP::Particle(model, "D"), "D");
  dye_hier.add_child(d_chain);
  IMP::atom::Residue d_res = IMP::atom::Residue::setup_particle(
      new IMP::Particle(model, "DYE"), IMP::atom::ResidueType("DYE"), 1);
  d_chain.add_child(d_res);
  const IMP::atom::Hierarchies raw_leaves = IMP::atom::get_leaves(raw_hier);
  for (std::size_t i = 0; i < raw_leaves.size(); ++i) d_res.add_child(raw_leaves[i]);
  const IMP::atom::Hierarchies dye_leaves = IMP::atom::get_leaves(dye_hier);
  for (std::size_t i = 0; i < dye_leaves.size(); ++i) {
    if (!IMP::core::XYZR::get_is_setup(dye_leaves[i])) {
      IMP::core::XYZR::setup_particle(dye_leaves[i].get_particle(), 1.0);
    }
  }

  const LinkerGeometry geometry = linker_geometry_from_mol2(mol2_path);
  IMP::atom::Hierarchy root = new_node(model, "root");
  root.add_child(protein);
  root.add_child(dye_hier);

  const ProbeCollisionWalk walk =
      probe_collision_walk(protein, dye_hier, geometry, a.chain, a.residue, a.n_steps, a.seed, 10);

  const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(root);
  // the protein does not move; each saved frame is it plus the probe's walk
  std::vector<double> frame = leaf_coordinates(leaves);
  const std::size_t probe_offset = frame.size() - 3 * static_cast<std::size_t>(walk.n_atoms);
  RmfStructureWriter writer(a.output_rmf, table_from_leaves(leaves), "root");
  for (int f = 0; f < walk.n_frames; ++f) {
    const std::size_t start = 3 * static_cast<std::size_t>(walk.n_atoms) * static_cast<std::size_t>(f);
    std::copy(walk.frames.begin() + start,
              walk.frames.begin() + start + 3 * static_cast<std::size_t>(walk.n_atoms),
              frame.begin() + probe_offset);
    writer.append(frame, format("%d", (f + 1) * 10 - 1));
  }
  writer.close();
  std::cout << format("Finished: %d/%d accepted, %d clashing atoms left. Saved to %s",
                      walk.n_accepted, a.n_steps, walk.n_clashing, a.output_rmf.c_str())
            << "\n";
}

// ---- sample-langevin ---------------------------------------------------------

struct LangevinArgs {
  std::string protein_pdb, chain = "A", dye = "alexa488", integrator = "md", output_rmf;
  int residue = 0, n_steps = 20000, write_every = 100, minimize_steps = 200, seed = 42;
  double temperature = 300.0, timestep_fs = -1.0, friction_ps = 10.0, interaction_sphere = 25.0;
};

void run_langevin(const LangevinArgs& a) {
  const std::string protein_path = resolve_protein_pdb(a.protein_pdb);
  const std::string mol2 = file_exists(a.dye) ? a.dye : find_probe_mol2(a.dye);
  IMP_NEW(IMP::Model, model, ());
  IMP::atom::Hierarchy protein = read_pdb(protein_path, model, new IMP::atom::NonWaterPDBSelector());
  IMP::atom::Hierarchy dye_hier = IMP::atom::read_mol2(mol2, model);
  attach_probes(protein, std::vector<ProbeAttachment>(1, ProbeAttachment(dye_hier, a.chain, a.residue)),
                true);
  AttachedProbeDynamics sampler(protein, dye_hier, mol2, a.chain, a.residue, a.integrator,
                                a.temperature, a.timestep_fs, a.friction_ps, a.interaction_sphere,
                                10.0, a.seed);
  if (a.minimize_steps > 0) sampler.minimize(a.minimize_steps);
  const std::string out_dir = path_dirname(path_abs(a.output_rmf));
  if (!out_dir.empty()) make_directory(out_dir);
  const ProbeSimulationTrajectory traj = sampler.run(a.n_steps, a.write_every);

  // the label's frames, with the protein as the fixed backdrop; the
  // trajectory's atom order is the label hierarchy's leaves
  IMP::atom::Hierarchy root = new_node(model, "system");
  root.add_child(protein);
  root.add_child(dye_hier);
  const IMP::atom::Hierarchies label_leaves = IMP::atom::get_leaves(dye_hier);
  const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(root);
  {
    RmfStructureWriter writer(a.output_rmf, table_from_leaves(leaves), "system");
    const std::size_t per = 3 * label_leaves.size();
    for (int f = 0; f < traj.n_frames; ++f) {
      for (std::size_t k = 0; k < label_leaves.size(); ++k) {
        const std::size_t o = per * static_cast<std::size_t>(f) + 3 * k;
        IMP::core::XYZ(label_leaves[k]).set_coordinates(
            IMP::algebra::Vector3D(traj.coordinates[o], traj.coordinates[o + 1],
                                   traj.coordinates[o + 2]));
      }
      writer.append(leaf_coordinates(leaves), format("%d", f));
    }
    writer.close();
  }

  double e_sum = 0.0;
  for (std::size_t i = 0; i < traj.potential_energy.size(); ++i) e_sum += traj.potential_energy[i];
  const double e_mean = traj.potential_energy.empty()
                            ? std::numeric_limits<double>::quiet_NaN()
                            : e_sum / static_cast<double>(traj.potential_energy.size());
  std::string text = format("%s dynamics finished: %d frames, %d steps x %s fs, <E_pot> %.1f kcal/mol",
                            a.integrator.c_str(), traj.n_frames, a.n_steps,
                            json_float(traj.timestep_fs).c_str(), e_mean);
  if (a.integrator == "md") {
    double t_sum = 0.0;
    int t_n = 0;
    for (std::size_t i = 0; i < traj.kinetic_energy.size(); ++i) {
      const double t = sampler.kinetic_temperature(traj.kinetic_energy[i]);
      if (std::isnan(t)) continue;
      t_sum += t;
      ++t_n;
    }
    const double t_kin = t_n ? t_sum / t_n : std::numeric_limits<double>::quiet_NaN();
    text += format(", <T_kin> %.0f K", t_kin);
  }
  std::cout << text << ". Wrote RMF: " << a.output_rmf << "\n";
}

// ---- label-fp ----------------------------------------------------------------

struct LabelFpArgs {
  std::string pdb, output;
  std::vector<std::string> sites;
};

//! A fluorescent-protein template by name, matched case-insensitively.
std::string fp_template(const std::string& name) {
  const std::string dir = get_template_dir("fp");
  std::string path = join(dir, name + ".pdb");
  if (!file_exists(path)) {
    const std::vector<std::string> candidates = directory_entries(dir, ".pdb");
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (lower(stem(candidates[i])) == lower(name)) {
        path = candidates[i];
        break;
      }
    }
  }
  return path;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out(1);
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == sep) out.push_back(std::string());
    else out.back() += s[i];
  }
  return out;
}

void run_label_fp(const LabelFpArgs& a) {
  IMP_NEW(IMP::Model, model, ());
  std::string pdb_path, pdb_name;
  if (file_exists(a.pdb)) {
    pdb_path = a.pdb;
    pdb_name = stem(pdb_path);
  } else {
    pdb_name = upper(a.pdb);
    pdb_path = get_structure_dir(pdb_name + ".pdb");
    if (!file_exists(pdb_path)) {
      ensure_dir(get_structure_dir());
      report_no_download(a.pdb, pdb_path);
      throw SubExit(1);
    }
  }
  IMP::atom::Hierarchy protein = read_pdb(pdb_path, model, new IMP::atom::NonWaterPDBSelector());
  IMP::atom::Hierarchy root = new_node(model, "root");
  root.add_child(protein);

  for (std::size_t s = 0; s < a.sites.size(); ++s) {
    const std::vector<std::string> parts = split(a.sites[s], ':');
    if (parts.size() < 2) throw SubError("--site " + a.sites[s] + " is not residue:fp[:chain][:anchor]");
    char* end = nullptr;
    const long res_num = std::strtol(parts[0].c_str(), &end, 10);
    if (parts[0].empty() || *end != '\0') {
      throw SubError("--site " + a.sites[s] + ": '" + parts[0] + "' is not a residue number");
    }
    const std::string fp_type = parts[1];
    const std::string chain_id =
        (parts.size() > 2 && parts[2] != "N" && parts[2] != "C") ? parts[2] : std::string("A");
    std::string anchor = "N";
    if (parts.size() > 3) anchor = parts[3];
    else if (parts.size() > 2 && (parts[2] == "N" || parts[2] == "C")) anchor = parts[2];

    const std::string fp_path = fp_template(fp_type);
    if (!file_exists(fp_path)) {
      std::cout << "Error: FP template " << fp_type << " not found.\n";
      continue;
    }
    IMP::atom::Hierarchy fp_hier = read_pdb(fp_path, model, new IMP::atom::AllPDBSelector());
    int r_min = 1, r_max = 1;
    bool any = false;
    const IMP::atom::Hierarchies fp_leaves = IMP::atom::get_leaves(fp_hier);
    for (std::size_t i = 0; i < fp_leaves.size(); ++i) {
      IMP::atom::Hierarchy parent = fp_leaves[i].get_parent();
      if (!parent || !IMP::atom::Residue::get_is_setup(parent)) continue;
      const int index = IMP::atom::Residue(parent).get_index();
      r_min = any ? std::min(r_min, index) : index;
      r_max = any ? std::max(r_max, index) : index;
      any = true;
    }
    const int fp_anchor_res = upper(anchor) == "N" ? r_min : r_max;
    std::cout << "Attaching " << fp_type << " (anchor " << anchor << ") to " << chain_id << ":"
              << res_num << "...\n";
    try {
      const IMP::ParticlesTemp site = resolve_probe_site(protein, chain_id, static_cast<int>(res_num));
      // templates are renumbered to 1..N and use chain 'A'
      align_hierarchies(fp_hier, "A", fp_anchor_res, IMP::core::XYZ(site[0]).get_coordinates(),
                        IMP::core::XYZ(site[1]).get_coordinates(),
                        IMP::core::XYZ(site[2]).get_coordinates());
      root.add_child(fp_hier);
    } catch (const std::exception& e) {
      std::cout << "  Error: " << e.what() << "\n";
    }
  }
  const std::string output =
      a.output.empty() ? "output/test_systems/" + pdb_name + "_labeled_fp.pdb" : a.output;
  make_parent_dir(output);
  IMP::atom::write_pdb(root, output);
  std::cout << "Saved to " << output << "\n";
}

// ---- label-fusion ------------------------------------------------------------

struct LabelFusionArgs {
  std::string pdb_path, chain = "A", output;
};

//! One-letter sequence of a chain; empty when the chain is not there.
std::string chain_sequence(IMP::atom::Hierarchy hierarchy, const std::string& chain_id) {
  const IMP::atom::Hierarchies chains = IMP::atom::get_by_type(hierarchy, IMP::atom::CHAIN_TYPE);
  for (std::size_t i = 0; i < chains.size(); ++i) {
    if (IMP::atom::Chain(chains[i]).get_id() != chain_id) continue;
    std::string letters;
    const IMP::atom::Hierarchies residues = IMP::atom::get_by_type(chains[i], IMP::atom::RESIDUE_TYPE);
    for (std::size_t k = 0; k < residues.size(); ++k) {
      letters += IMP::atom::get_one_letter_code(IMP::atom::Residue(residues[k]).get_residue_type());
    }
    return letters;
  }
  return std::string();
}

void run_label_fusion(const LabelFusionArgs& a) {
  IMP_NEW(IMP::Model, model, ());
  IMP::atom::Hierarchy protein = read_pdb(a.pdb_path, model, new IMP::atom::NonWaterPDBSelector());
  // the sequence comes from the hierarchy IMP just built
  const std::string seq = chain_sequence(protein, a.chain);
  if (seq.empty()) {
    std::cout << "Error: Could not find sequence for chain " << a.chain << " in " << a.pdb_path << "\n";
    throw SubExit(1);
  }
  std::cout << "Analyzing chain " << a.chain << " (length " << seq.size() << ")...\n";

  const std::vector<SequenceSegment> fp_domains = get_fp_domains(seq);
  if (fp_domains.empty()) {
    std::cout << "No FP domains detected.\n";
    throw SubExit(0);
  }
  for (std::size_t i = 0; i < fp_domains.size(); ++i) {
    std::cout << format("  Found %s at %d-%d (identity: %.2f)", fp_domains[i].name.c_str(),
                        fp_domains[i].start, fp_domains[i].end, fp_domains[i].identity)
              << "\n";
  }

  std::map<int, double> plddt;
  try {
    plddt = parse_plddt_from_pdb(a.pdb_path, a.chain);
  } catch (const std::exception& e) {
    std::cout << "Warning: Could not parse pLDDT (" << e.what() << "). Using dummy values.\n";
    plddt.clear();
    for (std::size_t i = 0; i < seq.size(); ++i) plddt[static_cast<int>(i) + 1] = 100.0;
  }
  // segments define rigid cores and flexible linkers
  const std::vector<SequenceSegment> segs =
      segments_from_plddt(static_cast<int>(seq.size()), plddt, fp_domains);

  IMP::atom::Hierarchy root = new_node(model, "root");
  root.add_child(protein);
  bool labeled_any = false;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    if (segs[i].kind != "fp") continue;
    const std::string fp_name = segs[i].name;
    std::cout << "  Labeling " << fp_name << " at residue " << segs[i].start << "...\n";
    const std::string template_path = fp_template(fp_name);
    if (file_exists(template_path)) {
      IMP::atom::Hierarchy fp_hier = read_pdb(template_path, model, new IMP::atom::AllPDBSelector());
      try {
        attach_probes(protein,
                      std::vector<ProbeAttachment>(1, ProbeAttachment(fp_hier, a.chain, segs[i].start)),
                      true);
        root.add_child(fp_hier);
        labeled_any = true;
      } catch (const std::exception& e) {
        std::cout << "    Error attaching " << fp_name << ": " << e.what() << "\n";
      }
    } else {
      std::cout << "    Warning: No template found for " << fp_name << "\n";
    }
  }
  if (!labeled_any) {
    std::cout << "No domains were labeled.\n";
    throw SubExit(0);
  }
  const std::string output = a.output.empty() ? stem(a.pdb_path) + "_labeled.pdb" : a.output;
  make_parent_dir(output);
  IMP::atom::write_pdb(root, output);
  std::cout << "Wrote labeled fusion system to " << output << "\n";
}

}  // namespace dye

void add_dye_subs(CLI::App& app) {
  CLI::App* group = app.add_subcommand("dye", R"doc(cgprobe: Coarse-Grained Probe Simulations in IMP.)doc");
  group->require_subcommand(1);

  {
    std::shared_ptr<dye::LabelArgs> a = std::make_shared<dye::LabelArgs>();
    CLI::App* sub = group->add_subcommand("label", R"doc(Label a protein with a dye.)doc");
    sub->add_option("pdb_id_or_path", a->pdb, "a PDB file, or a PDB id already in the structure directory")
        ->required();
    sub->add_option("--chain", a->chain, "Chain ID")->capture_default_str();
    sub->add_option("--residue", a->residue, "Residue number")->required();
    sub->add_option("--dye", a->dye, "Dye name")->required();
    sub->add_option("--linker", a->linker, "Linker type");
    sub->add_option("--output", a->output, "Output PDB path");
    sub->callback([a] {
      set_current_sub("dye label");
      dye::run_label(*a);
    });
  }
  {
    std::shared_ptr<dye::BuildLibArgs> a = std::make_shared<dye::BuildLibArgs>();
    CLI::App* sub = group->add_subcommand("build-lib", R"doc(Batch generate rotamer libraries.)doc");
    sub->add_option("--n-steps", a->n_steps, "Sampling steps per dye")->capture_default_str();
    sub->add_option("--cluster-threshold", a->cluster_threshold, "Cluster RMSD, A")
        ->capture_default_str();
    sub->add_option("--output-dir", a->output_dir,
                    "Where the libraries go (default: the rotamer template directory)");
    sub->callback([a] {
      set_current_sub("dye build-lib");
      dye::run_build_lib(*a);
    });
  }
  {
    std::shared_ptr<dye::AnalyzeTcArgs> a = std::make_shared<dye::AnalyzeTcArgs>();
    CLI::App* sub = group->add_subcommand("analyze-tc", R"doc(Calculate rotational correlation times.)doc");
    sub->add_option("input_rmf", a->input_rmf, "a rotamer library RMF")
        ->required()
        ->check(CLI::ExistingPath);
    sub->add_option("--dt", a->dt, "Timestep (ps)")->capture_default_str();
    sub->add_flag("--show-all", a->show_all, "Print every relaxation time.");
    sub->callback([a] {
      set_current_sub("dye analyze-tc");
      dye::run_analyze_tc(*a);
    });
  }
  {
    std::shared_ptr<dye::ReconstructArgs> a = std::make_shared<dye::ReconstructArgs>();
    CLI::App* sub = group->add_subcommand("reconstruct", R"doc(Reconstruct trajectory from kinetic library.)doc");
    sub->add_option("--lib-rmf", a->lib_rmf, "a rotamer library RMF with jump counts")->required();
    sub->add_option("--n-frames", a->n_frames, "Frames to draw")->capture_default_str();
    sub->add_option("--output-rmf", a->output_rmf, "The trajectory to write")->required();
    sub->add_option("--seed", a->seed, "Random seed")->capture_default_str();
    sub->callback([a] {
      set_current_sub("dye reconstruct");
      dye::run_reconstruct(*a);
    });
  }
  {
    std::shared_ptr<dye::SampleRotamerArgs> a = std::make_shared<dye::SampleRotamerArgs>();
    CLI::App* sub = group->add_subcommand("sample-rotamer", R"doc(Library-based sampling with clash detection.)doc");
    sub->add_option("--protein-pdb", a->protein_pdb, "The protein")->required();
    sub->add_option("--chain", a->chain, "Chain ID")->capture_default_str();
    sub->add_option("--residue", a->residue, "Residue number")->required();
    sub->add_option("--dye", a->dye, "Dye name")->required();
    sub->add_option("--n-samples", a->n_samples, "Frames to draw")->capture_default_str();
    sub->add_option("--output-rmf", a->output_rmf, "The trajectory to write")->required();
    sub->callback([a] {
      set_current_sub("dye sample-rotamer");
      dye::run_sample_rotamer(*a);
    });
  }
  {
    std::shared_ptr<dye::DofWalkArgs> a = std::make_shared<dye::DofWalkArgs>();
    CLI::App* sub = group->add_subcommand("sample-dof-walk", R"doc(Collision-gated Metropolis walk over the linker's internal DOFs.)doc");
    sub->footer(R"doc(Not Langevin dynamics: no forces, friction or temperature -- a proposal is
accepted whenever the dye does not clash. Real Langevin/Brownian dynamics
is ``sample-langevin`` (IMP.bff.cgprobe.sampling).)doc");
    sub->add_option("--protein-pdb", a->protein_pdb, "The protein")->required();
    sub->add_option("--chain", a->chain, "Chain ID")->capture_default_str();
    sub->add_option("--residue", a->residue, "Residue number")->required();
    sub->add_option("--dye", a->dye, "Dye name or MOL2 path")->required();
    sub->add_option("--linker", a->linker, "Linker type");
    sub->add_option("--n-steps", a->n_steps, "Proposals to make")->capture_default_str();
    sub->add_option("--output-rmf", a->output_rmf, "The trajectory to write")->required();
    sub->add_option("--seed", a->seed, "Random seed")->capture_default_str();
    sub->callback([a] {
      set_current_sub("dye sample-dof-walk");
      dye::run_dof_walk(*a);
    });
  }
  {
    std::shared_ptr<dye::LangevinArgs> a = std::make_shared<dye::LangevinArgs>();
    CLI::App* sub = group->add_subcommand("sample-langevin", R"doc(Langevin (md) or Brownian (bd) dynamics of an explicit dye attached at a residue.)doc");
    sub->footer(R"doc(Real stochastic dynamics on the dye force field (bonds, angles, dihedrals,
repulsive LJ) with a soft-sphere repulsion against the protein around the
site; the dye's backbone anchor stays on the residue. Thermodynamic pins:
test/cgprobe/test_langevin_sampler.py.)doc");
    sub->add_option("--protein-pdb", a->protein_pdb, "The protein")->required();
    sub->add_option("--chain", a->chain, "Chain ID")->capture_default_str();
    sub->add_option("--residue", a->residue, "Residue number")->required();
    sub->add_option("--dye", a->dye, "Dye name or MOL2 path.")->capture_default_str();
    sub->add_option("--integrator", a->integrator,
                    "md: Langevin-thermostat molecular dynamics; bd: Brownian dynamics.")
        ->check(CLI::IsMember({"md", "bd"}))
        ->capture_default_str();
    sub->add_option("--temperature", a->temperature, "K")->capture_default_str();
    sub->add_option("--timestep-fs", a->timestep_fs,
                    "Negative picks it per integrator: 2 fs (md), 0.5 fs (bd).")
        ->capture_default_str();
    sub->add_option("--friction-ps", a->friction_ps, "Langevin friction (1/ps), md only.")
        ->capture_default_str();
    sub->add_option("--n-steps", a->n_steps, "Steps")->capture_default_str();
    sub->add_option("--write-every", a->write_every, "Keep a frame this often")->capture_default_str();
    sub->add_option("--minimize-steps", a->minimize_steps, "Minimisation before the run")
        ->capture_default_str();
    sub->add_option("--interaction-sphere", a->interaction_sphere,
                    "Protein atoms within this of the site are obstacles, A")
        ->capture_default_str();
    sub->add_option("--output-rmf", a->output_rmf, "The trajectory to write")->required();
    sub->add_option("--seed", a->seed, "Random seed")->capture_default_str();
    sub->callback([a] {
      set_current_sub("dye sample-langevin");
      dye::run_langevin(*a);
    });
  }
  {
    std::shared_ptr<dye::LabelFpArgs> a = std::make_shared<dye::LabelFpArgs>();
    CLI::App* sub = group->add_subcommand("label-fp", R"doc(Label a protein with one or more Fluorescent Proteins (FPs).)doc");
    sub->add_option("pdb_id_or_path", a->pdb, "a PDB file, or a PDB id already in the structure directory")
        ->required();
    sub->add_option("--site", a->sites,
                    "Attachment site in format residue:fp[:chain][:anchor] (e.g., 6:eGFP:A:C)")
        ->required();
    sub->add_option("--output", a->output, "Output PDB path");
    sub->callback([a] {
      set_current_sub("dye label-fp");
      dye::run_label_fp(*a);
    });
  }
  {
    std::shared_ptr<dye::LabelFusionArgs> a = std::make_shared<dye::LabelFusionArgs>();
    CLI::App* sub = group->add_subcommand("label-fusion", R"doc(Automatically detect and label FPs in a fusion protein.)doc");
    sub->add_option("pdb_path", a->pdb_path, "the fusion protein's structure")->required();
    sub->add_option("--chain", a->chain, "Chain to analyze")->capture_default_str();
    sub->add_option("--output", a->output, "Output labeled PDB");
    sub->callback([a] {
      set_current_sub("dye label-fusion");
      dye::run_label_fusion(*a);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

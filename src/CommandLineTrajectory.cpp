/** \file CommandLineTrajectory.cpp
 *  \brief `imp_bff probe-pdb2cif`, `imp_bff traj2bcif`, `imp_bff traj2drot`:
 *         structure and trajectory conversions.
 *
 *  Ports of `bin/imp_bff_probe_pdb2cif`, `bin/imp_bff_traj2bcif` and
 *  `bin/imp_bff_traj2drot`. The work is the library's (the BinaryCIF writer
 *  and the XTC reader are in TrajectoryIO.h); what lives here is the grammar, the file
 *  discovery the scripts did (`<stem>.pdb` and `<stem>_weights.txt` beside a
 *  library) and the round-trip self-check.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <IMP/bff/Clustering.h>
#include <IMP/bff/ProbeRotamerLibrary.h>
#include <IMP/bff/StructureIO.h>
#include <IMP/bff/TrajectoryIO.h>
#include <IMP/bff/internal/Text.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace trajectory {

//! Round-trip tolerance of the lossless rung, Angstrom. float32 grids put
//! the floor at ~1e-5 A for a 30 A molecule; anything above this means the
//! reconstruction, not the storage, is wrong.
const double LOSSLESS_TOL_A = 1e-4;

//! Round-trip tolerance of the compact rung, in multiples of the grid step.
//! A dihedral error travels down the tree, so the bound is not the step
//! itself; 40x is what the shipped corpus needs with headroom.
const double COMPACT_TOL_STEPS = 40.0;

long file_size(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 ? static_cast<long>(st.st_size) : 0L;
}

std::string dirname_of(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

//! `name` without its last suffix, as pathlib's `stem`.
std::string stem_of(const std::string& path) {
  const std::string name = basename_of(path);
  const std::size_t dot = name.find_last_of('.');
  return dot == std::string::npos || dot == 0 ? name : name.substr(0, dot);
}

//! A text file of whitespace-separated numbers, as weights.
std::vector<double> read_weights(const std::string& path) {
  std::ifstream in(path.c_str());
  if (!in) throw SubError(path + ": cannot read weights");
  std::vector<double> out;
  double v = 0.0;
  while (in >> v) out.push_back(v);
  return out;
}

double* data_of(std::vector<double>& v) { return v.empty() ? nullptr : &v[0]; }

//! Atom names, elements and residue names of a template PDB, in file order.
/*! The element column (77-78) is authoritative when present -- bond
    perception depends on it, and guessing from a name calls CL chlorine or
    carbon depending on the day. Without it, the name's first letter. */
void parse_template(const std::string& path, std::vector<std::string>& names,
                    std::vector<std::string>& elements,
                    std::vector<std::string>& resnames) {
  std::ifstream in(path.c_str());
  if (!in) throw SubError(path + ": cannot read");
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    if (line.compare(0, 4, "ATOM") != 0 && line.compare(0, 6, "HETATM") != 0) continue;
    const std::string name = line.size() > 12 ? trimmed(line.substr(12, 4)) : std::string();
    names.push_back(name);
    resnames.push_back(line.size() > 17 ? trimmed(line.substr(17, 3)) : std::string());
    std::string element = line.size() > 76 ? trimmed(line.substr(76, 2)) : std::string();
    if (element.empty()) element = name.empty() ? std::string("C") : name.substr(0, 1);
    elements.push_back(element);
  }
  if (names.empty()) throw SubError(path + ": no ATOM/HETATM records");
}

//! `<stem>.pdb` beside a `<stem>_cutoff<N>.<ext>` library, if there is one.
std::string template_for(const std::string& src) {
  std::string stem = stem_of(src);
  const std::size_t cut = stem.find("_cutoff");
  if (cut != std::string::npos) stem = stem.substr(0, cut);
  const std::string candidate = dirname_of(src) + stem + ".pdb";
  return file_exists(candidate) ? candidate : std::string();
}

//! `<stem>_weights.txt` beside a library, if there is one.
std::string weights_for(const std::string& src) {
  const std::string candidate = dirname_of(src) + stem_of(src) + "_weights.txt";
  return file_exists(candidate) ? candidate : std::string();
}

//! The coordinates of a `.drot.pto` library, flat.
std::vector<double> library_coords(const ProbeRotamerLibrary& lib) {
  double* view = nullptr;
  int n = 0;
  lib.get_coords(&view, &n);
  std::vector<double> out(view, view + n);
  std::free(view);
  return out;
}

//! `n_frames * n_atoms * 3` coordinates from a .bcif, .dcd, .xtc or .drot
//! trajectory, in Angstrom.
std::vector<double> read_frames(const std::string& src, int n_atoms, int* n_frames,
                                const std::string& top = std::string()) {
  std::vector<double> xyz;
  if (ends_with(src, ".xtc")) {
    // XTC stores nanometres; read_xtc hands back Angstrom
    const int in_file = read_xtc_n_atoms(src);
    if (in_file != n_atoms) {
      throw SubError(basename_of(src) + format(": %d atoms per frame, but ", in_file) +
                     basename_of(top) + format(" has %d", n_atoms));
    }
    double* view = nullptr;
    int n_flat = 0;
    read_xtc(src, -1, &view, &n_flat);
    xyz.assign(view, view + n_flat);
    std::free(view);
  } else if (ends_with(src, ".drot") || ends_with(src, ".drot.pto")) {
    const ProbeRotamerLibrary lib = read_probe_rotamer_drot(src);
    xyz = library_coords(lib);
  } else {
    double* view = nullptr;
    int n_flat = 0;
    read_trajectory(src, n_atoms, -1, &view, &n_flat);
    xyz.assign(view, view + n_flat);
    std::free(view);
  }
  const std::size_t per_frame = 3 * static_cast<std::size_t>(n_atoms);
  if (xyz.empty() || xyz.size() % per_frame != 0) {
    throw SubError(basename_of(src) + ": " + format("%lu", (unsigned long)xyz.size()) +
                   " coordinates is not a whole number of frames of " +
                   format("%d", n_atoms) + " atoms");
  }
  *n_frames = static_cast<int>(xyz.size() / per_frame);
  return xyz;
}

//! Leader-cluster raw frames; keep the leaders, weight them by population.
void leader_cluster(std::vector<double>& xyz, int* n_frames, int n_atoms,
                    double threshold, std::vector<double>& weights) {
  int* leaders = nullptr;
  int n_leaders = 0;
  cluster_frames_leader(data_of(xyz), *n_frames, n_atoms, 3, threshold, &leaders,
                        &n_leaders);
  int* labels = nullptr;
  int n_labels = 0;
  assign_frames_to_clusters(data_of(xyz), *n_frames, n_atoms, 3, leaders, n_leaders,
                            &labels, &n_labels);
  std::vector<double> counts(static_cast<std::size_t>(n_leaders), 0.0);
  for (int i = 0; i < n_labels; ++i) {
    if (labels[i] >= 0 && labels[i] < n_leaders) counts[labels[i]] += 1.0;
  }
  const std::size_t per_frame = 3 * static_cast<std::size_t>(n_atoms);
  std::vector<double> reduced;
  reduced.reserve(static_cast<std::size_t>(n_leaders) * per_frame);
  for (int i = 0; i < n_leaders; ++i) {
    const std::size_t f = static_cast<std::size_t>(leaders[i]) * per_frame;
    reduced.insert(reduced.end(), xyz.begin() + f, xyz.begin() + f + per_frame);
  }
  xyz.swap(reduced);
  weights.swap(counts);
  *n_frames = n_leaders;
  std::free(leaders);
  std::free(labels);
}

struct Traj2DrotArgs {
  std::vector<std::string> positional;
  std::string top, weights, all, bundle;
  double cluster = -1.0, grid = -1.0, grid_deg = 0.01;
  bool no_verify = false;
};

//! One trajectory to one `.drot.pto`, round-trip checked by default.
void convert(const std::string& src, const std::string& dst, std::string top,
             const std::string& weights_path, double cluster_a, double grid_a,
             double grid_deg, bool verify) {
  std::vector<std::string> names, elements, resnames;
  int n_frames = 0;
  std::vector<double> xyz;
  if (top.empty() && ends_with(src, ".drot.pto")) {
    // a library carries its own template; nothing has to sit beside it
    const ProbeRotamerLibrary lib = read_probe_rotamer_drot(src);
    names = lib.atom_names;
    elements = lib.elements;
    resnames = lib.resnames;
    xyz = library_coords(lib);
    n_frames = lib.n_rotamers;
  } else {
    if (top.empty()) top = template_for(src);
    if (top.empty()) {
      throw SubError(basename_of(src) + ": no --top given and no template PDB beside it");
    }
    parse_template(top, names, elements, resnames);
    xyz = read_frames(src, static_cast<int>(names.size()), &n_frames, top);
  }
  const int n_atoms = static_cast<int>(names.size());

  std::vector<double> weights;
  if (!weights_path.empty()) weights = read_weights(weights_path);
  if (cluster_a >= 0.0) leader_cluster(xyz, &n_frames, n_atoms, cluster_a, weights);
  if (weights.empty()) weights.assign(static_cast<std::size_t>(n_frames), 1.0);
  if (weights.size() != static_cast<std::size_t>(n_frames)) {
    throw SubError(basename_of(src) + ": " + format("%lu", (unsigned long)weights.size()) +
                   " weights for " + format("%d", n_frames) +
                   " frames -- pass --cluster to build weights from populations");
  }

  ProbeRotamerDrotEncoding encoding;
  const bool compact = grid_a >= 0.0;
  if (compact) {
    encoding.lossless = false;
    encoding.grid_a = grid_a;
    encoding.grid_deg = grid_deg;
  }
  write_probe_rotamer_drot(dst, data_of(xyz), static_cast<int>(xyz.size()), names,
                           elements, resnames, data_of(weights),
                           static_cast<int>(weights.size()), encoding);

  std::string line = basename_of(dst) + ": " + format("%d rotamers, %d atoms, %ld bytes ",
                                                      n_frames, n_atoms, file_size(dst)) +
                     (compact ? "(" + json_float(grid_a) + " A grid)" : std::string("(lossless)"));
  if (verify) {
    const ProbeRotamerLibrary lib = read_probe_rotamer_drot(dst);
    const std::vector<double> back = library_coords(lib);
    if (back.size() != xyz.size()) {
      throw SubError(basename_of(dst) + ": read back " +
                     format("%lu", (unsigned long)back.size()) + " coordinates, wrote " +
                     format("%lu", (unsigned long)xyz.size()));
    }
    double err = 0.0;
    for (std::size_t i = 0; i < back.size(); ++i) {
      err = (std::max)(err, std::fabs(back[i] - xyz[i]));
    }
    const double tol = compact ? COMPACT_TOL_STEPS * grid_a : LOSSLESS_TOL_A;
    if (err > tol) {
      throw SubError(basename_of(dst) + format(": round trip differs by %.3e A (tolerance %.3e)",
                                               err, tol));
    }
    if (lib.atom_names != names) {
      throw SubError(basename_of(dst) + ": atom names did not survive the round trip");
    }
    line += format("  verified to %.2e A", err);
  }
  std::cout << line << "\n";
}

//! Re-encode a whole library directory: every `*_cutoff*.bcif` beside its
//! template, to `<stem>.drot.pto`. Keeps going past a failure.
int convert_all(const std::string& directory, double grid_a, double grid_deg, bool verify) {
  std::vector<std::string> sources;
  const std::vector<std::string> bcif = directory_entries(directory, ".bcif");
  for (std::size_t i = 0; i < bcif.size(); ++i) {
    if (basename_of(bcif[i]).find("_cutoff") != std::string::npos) sources.push_back(bcif[i]);
  }
  if (sources.empty()) throw SubError(directory + ": no *_cutoff*.bcif libraries");
  int failed = 0;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    const std::string& src = sources[i];
    const std::string dst = dirname_of(src) + stem_of(src) + ".drot.pto";
    try {
      convert(src, dst, template_for(src), weights_for(src), -1.0, grid_a, grid_deg, verify);
    } catch (const std::exception& e) {
      std::cerr << basename_of(src) << ": FAILED -- " << e.what() << "\n";
      ++failed;
    }
  }
  std::cout << (sources.size() - failed) << "/" << sources.size() << " written\n";
  return failed ? 1 : 0;
}

//! Several one-library containers into one family container.
/*! A container operation only: the payloads cross verbatim under names
    prefixed `<library>/`, and a `drot.catalog` object lists what is inside.
    Nothing is re-encoded, so bundling is a copy. */
void bundle(const std::vector<std::string>& sources, const std::string& out) {
  if (sources.empty()) throw SubError("--bundle needs at least one source container");
  std::vector<std::string> names;
  long total = 0;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    std::string name = basename_of(sources[i]);
    const std::size_t drot = name.find(".drot");
    if (drot != std::string::npos) name = name.substr(0, drot);
    names.push_back(name);
    total += file_size(sources[i]);
  }
  std::vector<std::string> sorted_names = names;
  std::sort(sorted_names.begin(), sorted_names.end());
  if (std::adjacent_find(sorted_names.begin(), sorted_names.end()) != sorted_names.end()) {
    throw SubError("two sources would take the same library name");
  }
  write_probe_rotamer_drot_bundle(sources, names, out);
  std::vector<std::string> listed = probe_rotamer_drot_catalog(out);
  std::sort(listed.begin(), listed.end());
  if (listed != sorted_names) throw SubError(out + ": the catalog does not list what went in");
  const long size = file_size(out);
  std::cout << basename_of(out)
            << format(": %lu libraries, %ld bytes (%.3fx the %ld bytes of the separate files)",
                      (unsigned long)names.size(), size,
                      total > 0 ? static_cast<double>(size) / total : 0.0, total)
            << "\n";
}

// ---- traj2bcif ------------------------------------------------------------

//! pathlib's `suffix.lower()`.
std::string lower_suffix(const std::string& path) {
  std::string s = path_suffix(path);
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  }
  return s;
}

//! What `traj2bcif` reads: coordinates in Angstrom from a DCD or an XTC.
std::vector<double> load_trajectory(const std::string& path, const std::string& top,
                                    int* n_frames, int* n_atoms) {
  double* view = nullptr;
  int n_flat = 0;
  const std::string suffix = lower_suffix(path);
  if (suffix == ".dcd") {
    *n_atoms = read_dcd_header(path).n_atoms;
    read_dcd(path, -1, &view, &n_flat);          // already A
  } else if (suffix == ".xtc") {
    if (top.empty()) throw SubError(".xtc needs --top (a .gro/.pdb topology)");
    *n_atoms = read_xtc_n_atoms(path);
    read_xtc(path, -1, &view, &n_flat);          // nm -> A inside
  } else if (suffix == ".trr") {
    throw SubError(".trr is not read by the compiled program (the Python one went through "
                   "mdtraj); convert it to .xtc or .dcd first");
  } else {
    throw SubError("unsupported trajectory: " + path);
  }
  std::vector<double> xyz(view, view + n_flat);
  std::free(view);
  const std::size_t per_frame = 3 * static_cast<std::size_t>(*n_atoms);
  *n_frames = per_frame ? static_cast<int>(xyz.size() / per_frame) : 0;
  return xyz;
}

//! One DCD/XTC to one `.bcif`, round-trip checked by default.
void convert_bcif(const std::string& src, const std::string& dst, const std::string& top,
                  double grid_a, bool verify) {
  int n_frames = 0, n_atoms = 0;
  std::vector<double> xyz = load_trajectory(src, top, &n_frames, &n_atoms);
  const long n_bytes = write_bcif_trajectory(dst, data_of(xyz), n_frames, n_atoms, 3, grid_a);
  const double before = static_cast<double>(file_size(src));
  const double coords = static_cast<double>(xyz.size());
  std::string line = basename_of(src) +
                     format(": %d x %d x 3  %6.2f MB -> %6.2f MB  (%.2f B/coord, %.0f %%)",
                            n_frames, n_atoms, before / 1e6, n_bytes / 1e6, n_bytes / coords,
                            n_bytes / before * 100);
  if (verify) {
    double* view = nullptr;
    int n_back = 0;
    read_bcif_trajectory(dst, n_atoms, "_rotamer_coord", &view, &n_back);
    std::vector<double> back(view, view + n_back);
    std::free(view);
    if (back.size() != xyz.size()) {
      throw SubError(basename_of(src) + format(": shape (%lu,) != (%lu,)",
                                               (unsigned long)back.size(),
                                               (unsigned long)xyz.size()));
    }
    double err = 0.0;
    for (std::size_t i = 0; i < xyz.size(); ++i) {
      const double want = grid_a <= 0.0
                              ? static_cast<double>(static_cast<float>(xyz[i]))
                              : std::nearbyint(xyz[i] / grid_a) * grid_a;
      err = (std::max)(err, std::fabs(back[i] - want));
    }
    if (err > 1e-9) {
      throw SubError(basename_of(src) + format(": round trip differs by %.3e A", err));
    }
    line += grid_a <= 0.0 ? std::string("  verified bit-exact (float32)")
                          : "  verified exact on the " + json_float(grid_a) + " A grid";
  }
  std::cout << line << "\n";
}

struct Traj2BcifArgs {
  std::vector<std::string> positional;
  std::string top, all;
  double grid = -1.0;
  bool no_verify = false;
};

}  // namespace trajectory

void add_trajectory_subs(CLI::App& app) {
  // ---- traj2bcif ------------------------------------------------------------
  {
    std::shared_ptr<trajectory::Traj2BcifArgs> a = std::make_shared<trajectory::Traj2BcifArgs>();
    CLI::App* sub =
        app.add_subcommand("traj2bcif", "Convert a trajectory (DCD or XTC) to BinaryCIF.");
    sub->footer(
        "BinaryCIF is this package's trajectory format. The shipped rotamer libraries are\n"
        "stored with it losslessly: float32 in, float32 out, 4.01 bytes per coordinate\n"
        "against DCD's 4.31 -- 7 % smaller. What the format buys is that the C++ side reads\n"
        "the libraries directly, through the ihm C parser IMP already vendors.\n\n"
        "Quantising is an option, not the default. Fixed-point at 0.001 A halves the files,\n"
        "but it moves transition-dipole directions enough to shift the FRETpredict reference\n"
        "values past their tolerance -- and 0.001, 0.005 and 0.01 A all cost exactly 2 bytes\n"
        "per coordinate, so a coarser grid buys nothing and loses accuracy.\n\n"
        "Usage:\n"
        "  imp_bff traj2bcif lib.dcd lib.bcif            # lossless, the default\n"
        "  imp_bff traj2bcif lib.dcd lib.bcif --grid 0.001\n"
        "  imp_bff traj2bcif --all data/rotamer_library\n"
        "  imp_bff traj2bcif traj.xtc traj.bcif --top conf_ed.gro\n\n"
        "XTC is read by the compiled reader (nm -> A); .trr is not read.");
    sub->add_option("sources", a->positional, "src (a .dcd or .xtc) and dst (the .bcif to write)");
    sub->add_option("--top", a->top, "topology for an XTC (.gro/.pdb)");
    sub->add_option("--grid", a->grid,
                    "quantise to this grid in A. Omit for lossless float32, which is the default "
                    "and is already smaller than DCD. Note that 0.001, 0.005 and 0.01 all cost "
                    "exactly 2 bytes per coordinate for these libraries -- the size is set by the "
                    "integer type, not the grid, so a coarser grid buys nothing.");
    sub->add_option("--all", a->all, "convert every .dcd in DIR next to its source");
    sub->add_flag("--no-verify", a->no_verify, "skip the round-trip check");
    sub->callback([a] {
      set_current_sub("traj2bcif");
      const bool verify = !a->no_verify;
      if (a->positional.size() > 2) {
        throw CLI::ValidationError("unrecognized arguments: " + a->positional[2]);
      }
      if (!a->all.empty()) {
        const std::vector<std::string> dcds = directory_entries(a->all, ".dcd");
        for (std::size_t i = 0; i < dcds.size(); ++i) {
          trajectory::convert_bcif(dcds[i], path_with_suffix(dcds[i], ".bcif"), "", a->grid,
                                   verify);
        }
        return;
      }
      if (a->positional.size() < 2) {
        throw CLI::ValidationError("give src and dst, or --all DIR");
      }
      trajectory::convert_bcif(a->positional[0], a->positional[1], a->top, a->grid, verify);
    });
  }

  // ---- probe-pdb2cif -------------------------------------------------------
  {
    struct Args {
      std::string pdb, cif, probe_id;
    };
    std::shared_ptr<Args> a = std::make_shared<Args>();
    CLI::App* sub = app.add_subcommand(
        "probe-pdb2cif", "Convert a probe PDB to an mmCIF structure file.");
    sub->alias("pdb2cif");
    sub->add_option("pdb", a->pdb, "input PDB")->required();
    sub->add_option("cif", a->cif, "output mmCIF")->required();
    sub->add_option("--probe-id", a->probe_id,
                    "chemical component id to record; empty takes it from the file");
    sub->callback([a] {
      set_current_sub("probe-pdb2cif");
      convert_pdb_to_cif(a->pdb, a->cif, a->probe_id);
    });
  }

  // ---- traj2drot ------------------------------------------------------------
  {
    std::shared_ptr<trajectory::Traj2DrotArgs> a =
        std::make_shared<trajectory::Traj2DrotArgs>();
    CLI::App* sub = app.add_subcommand(
        "traj2drot", "Convert a trajectory to a .drot.pto rotamer library, or bundle libraries.");
    sub->add_option("sources", a->positional,
                    "a .bcif, .dcd, .xtc or .drot trajectory and the library to write "
                    "(conventionally <stem>.drot.pto); with --bundle, the source containers");
    sub->add_option("--top", a->top,
                    "template PDB: atom names, elements, residue names. A .drot.pto source carries its own; otherwise defaults to "
                    "<stem>.pdb beside the source.");
    sub->add_option("--weights", a->weights,
                    "one weight per frame, as the shipped <stem>_cutoff<N>_weights.txt");
    sub->add_option("--cluster", a->cluster,
                    "leader-cluster the frames at this RMSD (A) first; the leaders become "
                    "the rotamers and the cluster populations the weights");
    sub->add_option("--grid", a->grid,
                    "compact rung: int16 grids at this step (A) for base coordinates and "
                    "bond lengths. Omit for the lossless float32 default -- the pins depend on it.");
    sub->add_option("--grid-deg", a->grid_deg, "angle step of the compact rung")
        ->capture_default_str();
    sub->add_option("--all", a->all, "re-encode every *_cutoff*.bcif in DIR to .drot.pto");
    sub->add_option("--bundle", a->bundle,
                    "bundle the given .drot.pto containers into one family container OUT, "
                    "each filed under its stem");
    sub->add_flag("--no-verify", a->no_verify, "skip the round-trip check");
    sub->callback([a] {
      set_current_sub("traj2drot");
      const bool verify = !a->no_verify;
      if (!a->bundle.empty()) {
        trajectory::bundle(a->positional, a->bundle);
        return;
      }
      if (!a->all.empty()) {
        if (trajectory::convert_all(a->all, a->grid, a->grid_deg, verify) != 0) {
          throw SubExit(1);  // each failure was already reported
        }
        return;
      }
      if (a->positional.size() < 2) {
        throw CLI::ValidationError("give src and dst, --all DIR, or --bundle OUT sources...");
      }
      trajectory::convert(a->positional[0], a->positional[1], a->top, a->weights, a->cluster,
                          a->grid, a->grid_deg, verify);
    });
  }
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

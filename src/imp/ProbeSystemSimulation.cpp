/**
 * \file src/imp/ProbeSystemSimulation.cpp
 * \brief The `imp_bff simulate` runner: an explicit dye on a force-field
 *        system, sampled by MD, rigid-body Monte Carlo, or both.
 *
 * A port of the Python runner that lived in `bin/imp_bff` (run_dye_simulation
 * and its helpers). Its kernels were already C++ (`ProbePotentialRestraints.h`:
 * the bonded, steric and Go restraint builders and the placement search);
 * what moved here is the driving -- loading, deciding what moves, the
 * rigid-body Monte Carlo step, the MC/MD alternation and the multi-restart
 * loop -- with the output files spelled as the Python spelled them.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/ProbeSystemSimulation.h>

#include <IMP/bff/MolecularGraph.h>
#include <IMP/bff/Mol2IO.h>
#include <IMP/bff/ProbeForceFieldCIF.h>
#include <IMP/bff/ProbeLibrary.h>
#include <IMP/bff/ProbePotentialRestraints.h>
#include <IMP/bff/internal/Text.h>

#include <IMP/Model.h>
#include <IMP/Particle.h>
#include <IMP/log.h>
#include <IMP/random.h>
#include <IMP/algebra/Rotation3D.h>
#include <IMP/algebra/vector_generators.h>
#include <IMP/atom/Atom.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/CenterOfMass.h>
#include <IMP/atom/LangevinThermostatOptimizerState.h>
#include <IMP/atom/Mass.h>
#include <IMP/atom/MolecularDynamics.h>
#include <IMP/atom/bond_decorators.h>
#include <IMP/atom/mol2.h>
#include <IMP/atom/pdb.h>
#include <IMP/core/ConjugateGradients.h>
#include <IMP/core/DistanceRestraint.h>
#include <IMP/core/Harmonic.h>
#include <IMP/core/RestraintsScoringFunction.h>
#include <IMP/core/XYZR.h>
#include <IMP/core/utility.h>
#include <IMP/rmf/atom_io.h>
#include <IMP/rmf/frames.h>
#include <RMF/FileHandle.h>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <stdlib.h>
#include <unistd.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace probe_system_simulation {

std::string fmt(const char* f, ...) {
  va_list args;
  va_start(args, f);
  char buf[4096];
  std::vsnprintf(buf, sizeof(buf), f, args);
  va_end(args);
  return buf;
}

void fail(const std::string& message) { IMP_THROW(message, IMP::ValueException); }

// ---- paths, as os.path spells them ----------------------------------------

//! A path separator on any platform this may read a path from -- a project
//! written on Windows and read here, or vice versa, still has one meaning.
bool is_sep(char c) { return c == '/' || c == '\\'; }

//! `p` starts with a drive letter, e.g. "C:". Recognized unconditionally,
//! not only under `_WIN32`: a foreign absolute path must never be treated as
//! relative and glued onto a local one, which produces a path that opens
//! nothing and reads like garbage (`/D:\a\...\C:\Users\...`).
bool has_drive(const std::string& p) {
  return p.size() > 1 && std::isalpha(static_cast<unsigned char>(p[0])) &&
         p[1] == ':';
}

std::vector<std::string> split_parts(const std::string& path) {
  std::vector<std::string> parts;
  std::string part;
  for (std::size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || is_sep(path[i])) {
      if (part == "..") {
        if (!parts.empty()) parts.pop_back();
      } else if (!part.empty() && part != ".") {
        parts.push_back(part);
      }
      part.clear();
    } else {
      part += path[i];
    }
  }
  return parts;
}

std::string cwd() {
  char buf[4096];
#ifdef _WIN32
  const char* c = _getcwd(buf, sizeof(buf));
#else
  const char* c = getcwd(buf, sizeof(buf));
#endif
  return c ? std::string(c) : std::string(".");
}

bool is_abs(const std::string& p) {
  if (p.empty()) return false;
  if (is_sep(p[0])) return true;
  return has_drive(p);
}

//! os.path.abspath
std::string abspath(const std::string& path) {
  const std::string full = is_abs(path) ? path : cwd() + "/" + path;
  const std::vector<std::string> parts = split_parts(full);
  if (parts.empty()) return std::string("/");
  std::string out;
  std::size_t i = 0;
  // A drive letter leads the path directly ("C:/..."), never after a slash
  // ("/C:/..." is not a path anything on Windows will open).
  if (has_drive(parts[0])) {
    out = parts[0];
    i = 1;
  }
  for (; i < parts.size(); ++i) out += "/" + parts[i];
  return out.empty() ? std::string("/") : out;
}

std::string join(const std::string& a, const std::string& b) {
  if (a.empty() || is_abs(b)) return b;
  return is_sep(a[a.size() - 1]) ? a + b : a + "/" + b;
}

std::string dirname(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  if (slash == std::string::npos) return std::string();
  return slash == 0 ? std::string("/") : path.substr(0, slash);
}

//! pathlib's non-strict resolve: symlinks resolved along the part that exists.
std::string resolve(const std::string& path) {
  std::string rest = abspath(path);
  std::string head = rest, tail;
  while (true) {
#ifndef _WIN32
    char buf[PATH_MAX];
    if (realpath(head.c_str(), buf) != nullptr) {
      return tail.empty() ? std::string(buf) : join(std::string(buf), tail);
    }
#endif
    // A drive root ("C:") is Windows' "/": realpath does not exist there, so
    // this is where the walk stops and abspath's answer is the final one.
    if (head.empty() || head == "/" || has_drive(head)) return rest;
    const std::size_t slash = head.find_last_of("/\\");
    if (slash == std::string::npos) return rest;
    const std::string leaf = head.substr(slash + 1);
    tail = tail.empty() ? leaf : leaf + "/" + tail;
    head = slash == 0 ? std::string("/") : head.substr(0, slash);
  }
}

void make_dirs(const std::string& path) { internal::make_directory(path); }

// ---- the system -------------------------------------------------------------

const std::vector<std::string>& group_ids(const ProbeForceFieldSystem& system,
                                          const std::string& group) {
  const std::map<std::string, std::vector<std::string> >& g = system.get_groups();
  std::map<std::string, std::vector<std::string> >::const_iterator it = g.find(group);
  if (it == g.end()) {
    fail("unknown group " + group);
    static const std::vector<std::string> none;
    return none;
  }
  return it->second;
}

bool has_group(const ProbeForceFieldSystem& system, const std::string& group) {
  return system.get_groups().count(group) > 0;
}

//! Python's repr of a sorted list of strings: ['a', 'b'].
std::string list_repr(const std::set<std::string>& values) {
  std::string out = "[";
  for (std::set<std::string>::const_iterator it = values.begin(); it != values.end(); ++it) {
    if (it != values.begin()) out += ", ";
    out += "'" + *it + "'";
  }
  return out + "]";
}

std::string component_for_group(const ProbeForceFieldSystem& system, const std::string& group) {
  const std::vector<std::string>& idv = group_ids(system, group);
  const std::set<std::string> ids(idv.begin(), idv.end());
  std::set<std::string> comps;
  for (std::size_t i = 0; i < system.get_sites().size(); ++i) {
    const FFSite& s = system.get_sites()[i];
    if (ids.count(s.id)) comps.insert(s.component);
  }
  if (comps.empty()) fail("group " + group + " has no sites");
  if (comps.size() != 1) fail("group " + group + " spans multiple components: " + list_repr(comps));
  return *comps.begin();
}

std::string infer_fixed_component(const ProbeForceFieldSystem& system) {
  const std::vector<std::string> fixed = system.get_fixed_components();
  if (!fixed.empty()) return fixed[0];
  fail("Could not infer fixed component; define role: fixed in system");
  return std::string();
}

std::string infer_mobile_group(const ProbeForceFieldSystem& system) {
  const std::vector<std::string> mobile = system.get_mobile_components();
  if (mobile.empty()) fail("No component with role 'mobile' found in system");
  const std::string comp = mobile[0];
  if (has_group(system, comp + "_all")) return comp + "_all";
  if (has_group(system, comp)) return comp;
  fail("Could not infer mobile group for " + comp + "; define '" + comp + "_all' in system");
  return std::string();
}

std::string atom_name_of(IMP::Particle* p) {
  std::string s = IMP::atom::Atom(p).get_atom_type().get_string();
  std::size_t at;
  while ((at = s.find("HET: ")) != std::string::npos) s.erase(at, 5);
  while ((at = s.find("HET:")) != std::string::npos) s.erase(at, 4);
  return internal::trimmed(s);
}

//! The loaded structures and the site particles, in the system's site order.
struct Loaded {
  IMP::atom::Hierarchy root;
  std::vector<std::string> site_ids;              // insertion order
  std::map<std::string, IMP::Particle*> site;     // id -> particle
  std::map<std::string, std::string> site_atom_names;
};

Loaded load_hierarchies(IMP::Model* model, const ProbeForceFieldSystem& system,
                        const std::string& system_cif, const std::set<std::string>& only) {
  const std::string base = dirname(abspath(system_cif));
  Loaded out;
  out.root = IMP::atom::Hierarchy::setup_particle(new IMP::Particle(model, "root"));

  std::map<std::string, std::map<int, IMP::Particle*> > atom_serial;
  std::map<std::string, std::map<int, std::string> > atom_name_serial;

  const std::map<std::string, FFComponent>& components = system.get_components();
  for (std::map<std::string, FFComponent>::const_iterator it = components.begin();
       it != components.end(); ++it) {
    const std::string& comp = it->first;
    if (!only.count(comp)) continue;
    std::string path = !it->second.mol2_path.empty() ? it->second.mol2_path : it->second.pdb_path;
    if (path.empty()) fail("Component " + comp + " has no mol2 or pdb path in system CIF");
    if (!is_abs(path)) path = abspath(join(base, path));
    const bool is_mol2 = path.size() >= 5 && path.compare(path.size() - 5, 5, ".mol2") == 0;
    IMP::atom::Hierarchy hier;
    {
      const IMP::LogLevel old = IMP::get_log_level();
      IMP::set_log_level(IMP::SILENT);
      try {
        hier = is_mol2 ? IMP::atom::read_mol2(path, model)
                       : IMP::atom::read_pdb(path, model, new IMP::atom::AllPDBSelector());
      } catch (...) {
        IMP::set_log_level(old);
        throw;
      }
      IMP::set_log_level(old);
    }
    hier->set_name(comp);
    out.root.add_child(hier);

    // IMP maps TRIPOS types C.3 -> C3 and loses the real name; read it
    std::map<int, std::string> mol2_names;
    if (is_mol2) mol2_names = read_mol2_atom_names(path);

    std::map<int, IMP::Particle*>& amap = atom_serial[comp];
    std::map<int, std::string>& nmap = atom_name_serial[comp];
    const IMP::atom::Hierarchies atoms = IMP::atom::get_by_type(hier, IMP::atom::ATOM_TYPE);
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      IMP::Particle* a = atoms[i].get_particle();
      const int idx = IMP::atom::Atom(a).get_input_index();
      amap[idx] = a;
      std::map<int, std::string>::const_iterator m = mol2_names.find(idx);
      nmap[idx] = (m != mol2_names.end() && !m->second.empty()) ? m->second : atom_name_of(a);
    }
  }

  const std::vector<FFSite>& sites = system.get_sites();
  for (std::size_t i = 0; i < sites.size(); ++i) {
    const FFSite& s = sites[i];
    if (!only.count(s.component)) continue;
    std::map<int, IMP::Particle*>& amap = atom_serial[s.component];
    const int serial = s.site_serial;
    std::map<int, IMP::Particle*>::const_iterator found = amap.find(serial);
    if (found == amap.end()) {
      fail(fmt("%s: serial %d missing from component %s pdb", s.id.c_str(), serial,
               s.component.c_str()));
    }
    IMP::Particle* p = found->second;
    if (IMP::core::XYZR::get_is_setup(p)) {
      IMP::core::XYZR(p).set_radius(s.radius);
    } else {
      IMP::core::XYZR::setup_particle(p, s.radius);
    }
    if (!IMP::atom::Mass::get_is_setup(p)) {
      IMP::atom::Mass::setup_particle(p, s.mass);
    } else {
      IMP::atom::Mass(p).set_mass(s.mass);
    }
    if (!IMP::atom::Bonded::get_is_setup(p)) IMP::atom::Bonded::setup_particle(p);
    if (!out.site.count(s.id)) out.site_ids.push_back(s.id);
    out.site[s.id] = p;
    std::map<int, std::string>::const_iterator n = atom_name_serial[s.component].find(serial);
    out.site_atom_names[s.id] = n == atom_name_serial[s.component].end() ? std::string() : n->second;
  }
  return out;
}

IMP::ParticleIndexes indexes_of(const Loaded& loaded) {
  IMP::ParticleIndexes out;
  for (std::size_t i = 0; i < loaded.site_ids.size(); ++i) {
    out.push_back(loaded.site.find(loaded.site_ids[i])->second->get_index());
  }
  return out;
}

std::set<std::string> mobile_ring_site_ids(const ProbeForceFieldSystem& system,
                                           const std::map<std::string, std::string>& names,
                                           const std::string& mobile_component) {
  const std::map<std::string, std::vector<std::string> > full = system.get_bonded_neighbors();
  std::set<std::string> mobile_ids;
  for (std::size_t i = 0; i < system.get_sites().size(); ++i) {
    if (system.get_sites()[i].component == mobile_component) {
      mobile_ids.insert(system.get_sites()[i].id);
    }
  }
  std::set<std::pair<std::string, std::string> > bonds;
  for (std::set<std::string>::const_iterator sid = mobile_ids.begin(); sid != mobile_ids.end();
       ++sid) {
    std::map<std::string, std::vector<std::string> >::const_iterator nb = full.find(*sid);
    if (nb == full.end()) continue;
    for (std::size_t k = 0; k < nb->second.size(); ++k) {
      const std::string& other = nb->second[k];
      if (!mobile_ids.count(other)) continue;
      bonds.insert(*sid <= other ? std::make_pair(*sid, other) : std::make_pair(other, *sid));
    }
  }
  const LabelledGraph graph(std::vector<std::pair<std::string, std::string> >(bonds.begin(),
                                                                             bonds.end()));
  const std::vector<std::vector<std::string> > cycles = graph.get_rings(8);
  std::set<std::string> ring;
  for (std::size_t i = 0; i < cycles.size(); ++i) {
    if (cycles[i].size() >= 5 && cycles[i].size() <= 7) ring.insert(cycles[i].begin(), cycles[i].end());
  }
  std::set<std::string> extra;
  for (std::set<std::string>::const_iterator sid = ring.begin(); sid != ring.end(); ++sid) {
    const std::vector<std::string> nbs = graph.get_neighbors(*sid);
    for (std::size_t k = 0; k < nbs.size(); ++k) {
      std::map<std::string, std::string>::const_iterator n = names.find(nbs[k]);
      if (n != names.end() && !n->second.empty() && n->second[0] == 'H') extra.insert(nbs[k]);
    }
  }
  ring.insert(extra.begin(), extra.end());
  return ring;
}

std::set<std::string> fixed_flex_ids(const ProbeForceFieldSystem& system, const std::string& mode,
                                     const std::string& fixed_name) {
  if (mode != "flex") return std::set<std::string>();
  const std::map<std::string, std::vector<std::string> >& g = system.get_groups();
  std::map<std::string, std::vector<std::string> >::const_iterator it = g.find(fixed_name + "_flex");
  if (it == g.end()) return std::set<std::string>();
  return std::set<std::string>(it->second.begin(), it->second.end());
}

//! Which sites MD integrates; sets `coordinates are optimized` on each.
void configure_movable(const ProbeForceFieldSystem& system, const Loaded& loaded,
                       const std::string& fixed_flex_mode,
                       const std::vector<std::string>& md_fixed_site_ids) {
  std::set<std::string> fixed;
  const std::vector<std::string>& fixed_groups = system.get_fixed_groups();
  for (std::size_t i = 0; i < fixed_groups.size(); ++i) {
    const std::vector<std::string>& ids = group_ids(system, fixed_groups[i]);
    fixed.insert(ids.begin(), ids.end());
  }
  // flex mode releases the atoms of any `_flex` group from the broad fixed set
  if (fixed_flex_mode == "flex") {
    const std::map<std::string, std::vector<std::string> >& g = system.get_groups();
    for (std::map<std::string, std::vector<std::string> >::const_iterator it = g.begin();
         it != g.end(); ++it) {
      const std::string& gid = it->first;
      if (gid.size() >= 5 && gid.compare(gid.size() - 5, 5, "_flex") == 0) {
        for (std::size_t k = 0; k < it->second.size(); ++k) fixed.erase(it->second[k]);
      }
    }
  }
  fixed.insert(md_fixed_site_ids.begin(), md_fixed_site_ids.end());
  for (std::size_t i = 0; i < loaded.site_ids.size(); ++i) {
    IMP::core::XYZ(loaded.site.find(loaded.site_ids[i])->second)
        .set_coordinates_are_optimized(!fixed.count(loaded.site_ids[i]));
  }
}

//! The system with its component paths made absolute against the CIF.
ProbeForceFieldSystem absolutize_components(const ProbeForceFieldSystem& system,
                                            const std::string& system_cif) {
  const std::string base_dir = dirname(resolve(system_cif));
  std::map<std::string, FFComponent> comps;
  const std::map<std::string, FFComponent>& src = system.get_components();
  for (std::map<std::string, FFComponent>::const_iterator it = src.begin(); it != src.end(); ++it) {
    FFComponent c;
    c.role = it->second.role;
    c.mol2_path = it->second.mol2_path;
    c.pdb_path = it->second.pdb_path;
    if (!c.mol2_path.empty() && !is_abs(c.mol2_path)) c.mol2_path = resolve(join(base_dir, c.mol2_path));
    if (!c.pdb_path.empty() && !is_abs(c.pdb_path)) c.pdb_path = resolve(join(base_dir, c.pdb_path));
    comps[it->first] = c;
  }
  ProbeForceFieldSystem out = system;
  out.set_components(comps);
  return out;
}

void write_system_copy(const ProbeForceFieldSystem& system, const std::string& system_cif,
                       const std::string& out_dir, const std::string& output_root) {
  const ProbeForceFieldSystem sys_out = absolutize_components(system, system_cif);
  write_probe_forcefield_cif(join(out_dir, "system.cif"), sys_out);

  // `systems/` beside the nearest `trajs` ancestor of the output root, else
  // inside the output root -- a lexical walk, as pathlib's parents are
  std::string anchor;
  std::string p = output_root;
  while (true) {
    while (p.size() > 1 && p[p.size() - 1] == '/') p.erase(p.size() - 1);
    const std::size_t slash = p.find_last_of('/');
    const std::string name = slash == std::string::npos ? p : p.substr(slash + 1);
    if (name == "trajs") {
      anchor = p;
      break;
    }
    if (slash == std::string::npos || p == "/") break;
    p = slash == 0 ? std::string("/") : p.substr(0, slash);
  }
  std::string systems_dir;
  if (!anchor.empty()) {
    const std::size_t slash = anchor.find_last_of('/');
    systems_dir = join(slash == std::string::npos ? std::string() : (slash == 0 ? "/" : anchor.substr(0, slash)),
                       "systems");
  } else {
    systems_dir = join(output_root, "systems");
  }
  make_dirs(systems_dir);
  const std::string name = system.get_name().empty() ? std::string("ff_system") : system.get_name();
  write_probe_forcefield_cif(join(systems_dir, name + ".system.cif"), sys_out);
}

// ---- sampling ----------------------------------------------------------------

typedef std::vector<IMP::Particle*> Ps;

IMP::algebra::Vector3D centroid(const Ps& ps) {
  IMP::core::XYZs xyzs;
  for (std::size_t i = 0; i < ps.size(); ++i) xyzs.push_back(IMP::core::XYZ(ps[i]));
  return IMP::core::get_centroid(xyzs);
}

double gauss(double sigma) {
  boost::normal_distribution<double> n(0.0, sigma);
  return n(IMP::random_number_generator);
}

double uniform01() {
  boost::uniform_real<double> u(0.0, 1.0);
  return u(IMP::random_number_generator);
}

//! Rigid-body Metropolis moves of each group as a whole; returns (accepted, proposed).
/*! IMP's MolecularDynamics cannot integrate rigid-body members, so no
    RigidBody is made: each move translates and rotates the group's atoms
    about their centroid and is tested against \p mc_sf. */
std::pair<int, int> rb_mc_step(const std::vector<Ps>& groups, IMP::ScoringFunction* mc_sf,
                               double mc_temp, int mc_steps, double max_trans,
                               double max_rot_rad) {
  const double kT = mc_temp;
  const double t_scale = max_trans / 3.0;
  const double r_scale = max_rot_rad / 3.0;
  int n_acc = 0, n_prop = 0;
  for (int step = 0; step < mc_steps; ++step) {
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const Ps& ps = groups[g];
      if (ps.empty()) continue;
      std::vector<IMP::algebra::Vector3D> old(ps.size());
      double sx = 0, sy = 0, sz = 0;
      for (std::size_t i = 0; i < ps.size(); ++i) {
        old[i] = IMP::core::XYZ(ps[i]).get_coordinates();
        sx += old[i][0];
        sy += old[i][1];
        sz += old[i][2];
      }
      const double n = static_cast<double>(ps.size());
      const IMP::algebra::Vector3D com(sx / n, sy / n, sz / n);
      const double tx = gauss(t_scale), ty = gauss(t_scale), tz = gauss(t_scale);
      const IMP::algebra::Vector3D trans(tx, ty, tz);
      const double angle = gauss(r_scale);
      const IMP::algebra::Vector3D axis = IMP::algebra::get_random_vector_on_unit_sphere();
      const double half = angle / 2.0, s = std::sin(half);
      const IMP::algebra::Rotation3D rot(
          IMP::algebra::Vector4D(std::cos(half), axis[0] * s, axis[1] * s, axis[2] * s));
      const double before = mc_sf->evaluate(false);
      for (std::size_t i = 0; i < ps.size(); ++i) {
        IMP::core::XYZ(ps[i]).set_coordinates(rot.get_rotated(old[i] - com) + com + trans);
      }
      const double after = mc_sf->evaluate(false);
      const double delta = after - before;
      ++n_prop;
      if (delta <= 0 || uniform01() < std::exp(-delta / kT)) {
        ++n_acc;
      } else {
        for (std::size_t i = 0; i < ps.size(); ++i) IMP::core::XYZ(ps[i]).set_coordinates(old[i]);
      }
    }
  }
  return std::make_pair(n_acc, n_prop);
}

IMP::Pointer<IMP::atom::MolecularDynamics> make_md(IMP::Model* model, IMP::ScoringFunction* sf,
                                                  const IMP::ParticlesTemp& movable,
                                                  double temperature_k, double friction_ps,
                                                  double timestep_fs, bool assign) {
  IMP_NEW(IMP::atom::MolecularDynamics, md, (model));
  md->set_scoring_function(sf);
  md->set_maximum_time_step(timestep_fs);
  if (assign) md->assign_velocities(temperature_k);
  if (!movable.empty()) {
    IMP_NEW(IMP::atom::LangevinThermostatOptimizerState, thermo,
            (model, IMP::get_indexes(movable), temperature_k, friction_ps));
    thermo->set_period(1);
    md->add_optimizer_state(thermo);
  }
  return md;
}

void write_initial_rmf(const std::string& out_dir, IMP::atom::Hierarchy root) {
  RMF::FileHandle fh = RMF::create_rmf_file(join(out_dir, "initial.0.rmf3"));
  IMP::rmf::add_hierarchies(fh, IMP::atom::Hierarchies(1, root));
  IMP::rmf::save_frame(fh, "init");
}

std::ofstream open_stat(const std::string& path) {
  std::ofstream stat(path.c_str());
  if (!stat) fail("cannot write " + path);
  return stat;
}

void run_simple_md(IMP::Model* model, IMP::atom::Hierarchy root, IMP::ScoringFunction* sf,
                   const std::string& out_dir, const IMP::ParticlesTemp& movable, int n_steps,
                   int write_every, double temperature_k, double friction_ps, double timestep_fs,
                   int log_every_frames, std::ostream& out) {
  IMP::Pointer<IMP::atom::MolecularDynamics> md =
      make_md(model, sf, movable, temperature_k, friction_ps, timestep_fs, true);
  write_initial_rmf(out_dir, root);
  RMF::FileHandle rmf = RMF::create_rmf_file(join(join(out_dir, "rmfs"), "0.rmf3"));
  IMP::rmf::add_hierarchies(rmf, IMP::atom::Hierarchies(1, root));

  const int n_frames = std::max(1, n_steps / std::max(1, write_every));
  std::ofstream stat = open_stat(join(out_dir, "stat.0.out"));
  stat << "frame\tscore\tkinetic_energy\n";
  IMP::rmf::save_frame(rmf, "init");
  const double init_score = sf->evaluate(false);
  stat << fmt("init\t%.6f\t%.6f\n", init_score, md->get_kinetic_energy());
  for (int frame = 0; frame < n_frames; ++frame) {
    md->optimize(write_every);
    const double score = sf->evaluate(false);
    const double ke = md->get_kinetic_energy();
    IMP::rmf::save_frame(rmf, fmt("%d", frame));
    stat << fmt("%d\t%.6f\t%.6f\n", frame, score, ke);
    if (frame % std::max(1, log_every_frames) == 0) {
      out << fmt("  frame %5d/%d score=%.2f KE=%.2f", frame, n_frames, score, ke) << "\n";
    }
  }
}

std::vector<Ps> particles_of_groups(const std::vector<std::vector<std::string> >& groups,
                                    const Loaded& loaded) {
  std::vector<Ps> out;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    Ps ps;
    for (std::size_t i = 0; i < groups[g].size(); ++i) {
      std::map<std::string, IMP::Particle*>::const_iterator it = loaded.site.find(groups[g][i]);
      if (it == loaded.site.end()) fail("KeyError: '" + groups[g][i] + "'");
      ps.push_back(it->second);
    }
    out.push_back(ps);
  }
  return out;
}

double mean_distance(const Ps& ps, const std::vector<IMP::algebra::Vector3D>& ref,
                     const IMP::algebra::Vector3D& shift_now,
                     const IMP::algebra::Vector3D& shift_ref, bool relative) {
  double sum = 0;
  for (std::size_t i = 0; i < ps.size(); ++i) {
    IMP::algebra::Vector3D c = IMP::core::XYZ(ps[i]).get_coordinates();
    if (relative) c = c - shift_now;
    (void)shift_ref;
    sum += IMP::algebra::get_distance(c, ref[i]);
  }
  return sum / static_cast<double>(ps.size());
}

void run_alternating_rb_mc_md(IMP::Model* model, IMP::ScoringFunction* mc_sf,
                              IMP::ScoringFunction* full_sf, IMP::atom::Hierarchy root,
                              const std::string& out_dir, const Loaded& loaded,
                              const IMP::ParticlesTemp& movable,
                              const std::vector<std::vector<std::string> >& rb_groups_in,
                              int md_steps, int md_steps_per_block, int mc_steps_per_block,
                              double temperature_k, double friction_ps, double timestep_fs,
                              double rb_max_trans, double rb_max_rot_deg, double mc_temp,
                              int log_every_frames, const std::set<std::string>& flex_ids,
                              const Ps& ring_ps, const Ps& linker_ps, std::ostream& out) {
  std::vector<std::vector<std::string> > rb_groups;
  for (std::size_t g = 0; g < rb_groups_in.size(); ++g) {
    if (!rb_groups_in[g].empty()) rb_groups.push_back(rb_groups_in[g]);
  }
  if (rb_groups.empty()) IMP_THROW("No rb_groups in system mmCIF", IMP::ValueException);
  const std::vector<Ps> rb_ps = particles_of_groups(rb_groups, loaded);

  IMP::Pointer<IMP::atom::MolecularDynamics> md =
      make_md(model, full_sf, movable, temperature_k, friction_ps, timestep_fs, true);
  write_initial_rmf(out_dir, root);
  RMF::FileHandle rmf = RMF::create_rmf_file(join(join(out_dir, "rmfs"), "0.rmf3"));
  IMP::rmf::add_hierarchies(rmf, IMP::atom::Hierarchies(1, root));

  const int n_frames = std::max(1, md_steps / std::max(1, md_steps_per_block));

  std::set<IMP::Particle*> rb_unique;
  for (std::size_t g = 0; g < rb_ps.size(); ++g) rb_unique.insert(rb_ps[g].begin(), rb_ps[g].end());
  const Ps rb_particles(rb_unique.begin(), rb_unique.end());
  const IMP::algebra::Vector3D rb_com0 =
      rb_particles.empty() ? IMP::algebra::Vector3D(0, 0, 0) : centroid(rb_particles);

  Ps flex_ps;
  for (std::set<std::string>::const_iterator it = flex_ids.begin(); it != flex_ids.end(); ++it) {
    std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(*it);
    if (p != loaded.site.end()) flex_ps.push_back(p->second);
  }
  std::vector<IMP::algebra::Vector3D> flex0;
  for (std::size_t i = 0; i < flex_ps.size(); ++i) {
    flex0.push_back(IMP::core::XYZ(flex_ps[i]).get_coordinates());
  }

  // internal linker displacement: each linker atom relative to the ring centroid
  const bool have_ring = !ring_ps.empty();
  IMP::algebra::Vector3D ring_com0(0, 0, 0);
  std::vector<IMP::algebra::Vector3D> linker_rel0;
  if (have_ring) {
    ring_com0 = centroid(ring_ps);
    for (std::size_t i = 0; i < linker_ps.size(); ++i) {
      linker_rel0.push_back(IMP::core::XYZ(linker_ps[i]).get_coordinates() - ring_com0);
    }
  }

  out << fmt("    Alternating RB-MC/MD: frames=%d, mc_steps=%d, md_steps=%d, rb_groups=%d",
             n_frames, mc_steps_per_block, md_steps_per_block, static_cast<int>(rb_groups.size()))
      << "\n";

  std::ofstream stat = open_stat(join(out_dir, "stat.0.out"));
  stat << "frame\tscore\tkinetic_energy\n";
  IMP::rmf::save_frame(rmf, "init");
  stat << fmt("init\t%.6f\t%.6f\n", full_sf->evaluate(false), md->get_kinetic_energy());

  const double max_rot_rad = rb_max_rot_deg * 3.141592653589793 / 180.0;
  for (int frame = 0; frame < n_frames; ++frame) {
    const std::pair<int, int> acc =
        rb_mc_step(rb_ps, mc_sf, mc_temp, mc_steps_per_block, rb_max_trans, max_rot_rad);
    md->optimize(md_steps_per_block);
    const double score = full_sf->evaluate(false);
    const double ke = md->get_kinetic_energy();
    const IMP::algebra::Vector3D rb_com = rb_particles.empty() ? rb_com0 : centroid(rb_particles);
    const double rb_com_disp = IMP::algebra::get_distance(rb_com, rb_com0);

    double flex_disp = 0.0;
    if (!flex_ps.empty()) {
      flex_disp = mean_distance(flex_ps, flex0, IMP::algebra::Vector3D(0, 0, 0),
                                IMP::algebra::Vector3D(0, 0, 0), false);
    }
    double linker_disp = 0.0;
    if (have_ring && !linker_ps.empty() && !linker_rel0.empty()) {
      linker_disp = mean_distance(linker_ps, linker_rel0, centroid(ring_ps), ring_com0, true);
    }

    IMP::rmf::save_frame(rmf, fmt("%d", frame));
    stat << fmt("%d\t%.6f\t%.6f\n", frame, score, ke);
    if (frame % std::max(1, log_every_frames) == 0) {
      out << fmt("  frame %5d/%d score=%.2f KE=%.2f rbCOM=%.3fA linkerInt=%.3fA so3Disp=%.3fA "
                 "mcAcc=%d/%d",
                 frame, n_frames, score, ke, rb_com_disp, linker_disp, flex_disp, acc.first,
                 acc.second)
          << "\n";
    }
  }
}

//! Guest and host particles for a placement search.
void placement_sets(const ProbeForceFieldSystem& system, const Loaded& loaded,
                    const std::string& guest_group, const std::string& host_component,
                    IMP::ParticleIndexes& host, IMP::ParticleIndexes& guest) {
  const std::vector<std::string>& mobile = group_ids(system, guest_group);
  const std::set<std::string> mobile_set(mobile.begin(), mobile.end());
  host.clear();
  guest.clear();
  if (has_group(system, host_component + "_all")) {
    const std::vector<std::string>& fixed = group_ids(system, host_component + "_all");
    for (std::size_t i = 0; i < fixed.size(); ++i) {
      std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(fixed[i]);
      if (p == loaded.site.end()) fail("KeyError: '" + fixed[i] + "'");
      host.push_back(p->second->get_index());
    }
  }
  if (host.empty()) {
    for (std::size_t i = 0; i < loaded.site_ids.size(); ++i) {
      if (!mobile_set.count(loaded.site_ids[i])) {
        host.push_back(loaded.site.find(loaded.site_ids[i])->second->get_index());
      }
    }
  }
  for (std::size_t i = 0; i < mobile.size(); ++i) {
    std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(mobile[i]);
    if (p == loaded.site.end()) fail("KeyError: '" + mobile[i] + "'");
    guest.push_back(p->second->get_index());
  }
}

void run_multi_restart(IMP::Model* model, IMP::ScoringFunction* mc_sf,
                       IMP::ScoringFunction* full_sf, IMP::atom::Hierarchy root,
                       const std::string& out_dir, const Loaded& loaded,
                       const IMP::ParticlesTemp& movable,
                       const std::vector<std::vector<std::string> >& rb_groups, int n_restarts,
                       int md_steps, int md_steps_per_block, int mc_steps_per_block,
                       double mc_temp, double rb_max_trans, double rb_max_rot_deg,
                       double temperature_k, double friction_ps, double timestep_fs,
                       const ProbeForceFieldSystem& system, const std::string& guest_group,
                       double init_distance_a, int init_trials, int init_seed,
                       const std::string& host_component, std::ostream& out) {
  out << fmt("Running %d independent restarts for multi_restart mode.", n_restarts) << "\n";
  make_dirs(out_dir);
  const std::string final_rmf_path = join(join(out_dir, "rmfs"), "0.rmf3");
  make_dirs(dirname(final_rmf_path));
  RMF::FileHandle rmf = RMF::create_rmf_file(final_rmf_path);
  IMP::rmf::add_hierarchies(rmf, IMP::atom::Hierarchies(1, root));

  // the state every restart starts from
  const IMP::ParticleIndexes all = model->get_particle_indexes();
  std::vector<std::pair<IMP::ParticleIndex, IMP::algebra::Vector3D> > initial;
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (IMP::core::XYZ::get_is_setup(model, all[i])) {
      initial.push_back(std::make_pair(all[i], IMP::core::XYZ(model, all[i]).get_coordinates()));
    }
  }

  std::ofstream stat = open_stat(join(out_dir, "stat.0.out"));
  stat << "frame\tscore\tkinetic_energy\n";
  IMP::Pointer<IMP::atom::MolecularDynamics> md =
      make_md(model, full_sf, movable, temperature_k, friction_ps, timestep_fs, false);
  const std::vector<Ps> rb_ps = particles_of_groups(rb_groups, loaded);
  const double max_rot_rad = rb_max_rot_deg * 3.141592653589793 / 180.0;

  for (int i = 0; i < n_restarts; ++i) {
    for (std::size_t k = 0; k < initial.size(); ++k) {
      IMP::core::XYZ(model, initial[k].first).set_coordinates(initial[k].second);
    }
    IMP::ParticleIndexes host, guest;
    placement_sets(system, loaded, guest_group, host_component, host, guest);
    place_guest_by_score(mc_sf, model, host, guest, init_distance_a, init_trials, init_seed + i);
    md->assign_velocities(temperature_k);

    const int n_blocks = std::max(1, md_steps / std::max(1, md_steps_per_block));
    for (int b = 0; b < n_blocks; ++b) {
      rb_mc_step(rb_ps, mc_sf, mc_temp, mc_steps_per_block, rb_max_trans, max_rot_rad);
      md->optimize(md_steps_per_block);
      if ((b + 1) % 10 == 0) {
        IMP::rmf::save_frame(rmf, fmt("restart_%d_block_%d", i, b));
        if ((b + 1) % 50 == 0) {
          out << fmt("    [Restart %d Block %d/%d] score=%.2f", i + 1, b + 1, n_blocks,
                     full_sf->evaluate(false))
              << std::endl;
        }
      }
    }
    const double score = full_sf->evaluate(false);
    const double ke = md->get_kinetic_energy();
    IMP::rmf::save_frame(rmf, fmt("%d", i));
    stat << fmt("%d\t%.6f\t%.6f\n", i, score, ke);
    stat.flush();
    if ((i + 1) % 100 == 0 || i == 0) {
      out << fmt("  [Restart %d/%d] score=%.2f KE=%.2f", i + 1, n_restarts, score, ke) << std::endl;
    }
  }
  out << "Meta-sampling complete. Final frames saved to " << final_rmf_path << "\n";
}

}  // namespace probe_system_simulation

std::vector<std::string> probe_system_paths(const std::string& system_cif,
                                            const std::string& systems_dir) {
  if (!system_cif.empty()) return std::vector<std::string>(1, system_cif);
  if (systems_dir.empty()) {
    IMP_THROW("Provide --system-cif or --systems-dir", IMP::ValueException);
  }
  std::vector<std::string> paths;
  if (!internal::file_exists(systems_dir)) {
    IMP_THROW("cannot list " << systems_dir, IMP::IOException);
  }
  std::vector<std::string> files, dirs;
  internal::directory_listing(systems_dir, files, dirs);
  for (std::size_t i = 0; i < files.size(); ++i) {
    const std::string& name = files[i];
    if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".cif") == 0) {
      paths.push_back(probe_system_simulation::join(systems_dir, name));
    }
  }
  if (paths.empty()) IMP_THROW("No .cif files found in " << systems_dir, IMP::ValueException);
  return paths;
}

void run_probe_system_simulation(const std::string& system_cif,
                                 const ProbeSystemSimulationOptions& o, std::ostream& out) {
  namespace pss = probe_system_simulation;
  typedef std::vector<std::string> Ids;

  ProbeForceFieldSystem system = read_forcefield_cif(system_cif);
  const std::string problem = system.get_inconsistency();
  if (!problem.empty()) pss::fail(problem);
  if (!std::isnan(o.friction_ps) || !std::isnan(o.temperature_k)) {
    FFSampling s = system.get_sampling();
    if (!std::isnan(o.friction_ps)) s.friction_ps = o.friction_ps;
    if (!std::isnan(o.temperature_k)) s.temperature_K = o.temperature_k;
    system.set_sampling(s);
  }
  const std::string output_root =
      o.output_root.empty() ? pss::join(pss::join(pss::cwd(), "output"), "trajs") : o.output_root;

  IMP_NEW(IMP::Model, model, ());
  std::string mobile_group = o.mobile_group;
  if (mobile_group.empty()) {
    mobile_group = pss::infer_mobile_group(system);
    out << "Inferred mobile group: " << mobile_group << "\n";
  }
  const std::string mobile_component = pss::component_for_group(system, mobile_group);
  const std::string fixed_component = pss::infer_fixed_component(system);
  std::set<std::string> only;
  only.insert(fixed_component);
  only.insert(mobile_component);
  const pss::Loaded loaded = pss::load_hierarchies(model, system, system_cif, only);
  const IMP::ParticleIndexes particles = pss::indexes_of(loaded);

  // the bonded terms and the steric subset a rigid move feels
  IMP::Restraints restraints =
      create_probe_restraints(model, system, loaded.site_ids, particles, false);
  IMP::Restraints softsphere;
  IMP::Pointer<IMP::Restraint> steric =
      create_steric_restraint(model, system, loaded.site_ids, particles);
  if (steric) softsphere.push_back(steric);
  restraints.insert(restraints.end(), softsphere.begin(), softsphere.end());

  const std::string run_name = fixed_component + "_" + mobile_component;
  const std::string out_dir = pss::join(output_root, run_name + "_imp");
  pss::make_dirs(pss::join(out_dir, "rmfs"));

  // native contacts: the mobile component everywhere, the fixed one only
  // where flex mode released atoms
  std::map<std::string, std::string> names;
  for (std::size_t i = 0; i < loaded.site_ids.size(); ++i) {
    names[loaded.site_ids[i]] = loaded.site_atom_names.find(loaded.site_ids[i])->second;
  }
  IMP::Restraints go = create_go_restraints(model, system, loaded.site_ids, particles, names,
                                            mobile_component, Ids(), o.go_mobile_k, o.go_cutoff);
  const std::set<std::string> flex_ids =
      pss::fixed_flex_ids(system, o.fixed_flex_mode, fixed_component);
  if (!flex_ids.empty()) {
    const IMP::Restraints fixed_go =
        create_go_restraints(model, system, loaded.site_ids, particles, names, fixed_component,
                             Ids(flex_ids.begin(), flex_ids.end()), o.go_fixed_k, o.go_cutoff);
    go.insert(go.end(), fixed_go.begin(), fixed_go.end());
  }

  IMP::Pointer<IMP::Restraint> com_pull;
  if (o.com_pull_k > 0) {
    const Ids fixed_ids = pss::has_group(system, fixed_component + "_all")
                              ? pss::group_ids(system, fixed_component + "_all")
                              : Ids();
    const Ids& mobile_ids = pss::group_ids(system, mobile_group);
    IMP::ParticleIndexes fixed_ps, mobile_ps;
    for (std::size_t i = 0; i < fixed_ids.size(); ++i) {
      std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(fixed_ids[i]);
      if (p != loaded.site.end()) fixed_ps.push_back(p->second->get_index());
    }
    for (std::size_t i = 0; i < mobile_ids.size(); ++i) {
      std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(mobile_ids[i]);
      if (p != loaded.site.end()) mobile_ps.push_back(p->second->get_index());
    }
    if (!fixed_ps.empty() && !mobile_ps.empty()) {
      IMP::atom::CenterOfMass c1 =
          IMP::atom::CenterOfMass::setup_particle(new IMP::Particle(model), fixed_ps);
      IMP::atom::CenterOfMass c2 =
          IMP::atom::CenterOfMass::setup_particle(new IMP::Particle(model), mobile_ps);
      IMP_NEW(IMP::core::Harmonic, h, (0.0, o.com_pull_k));
      com_pull = new IMP::core::DistanceRestraint(model, h, c1.get_particle_index(),
                                                  c2.get_particle_index());
      com_pull->set_name("com_pull");
    }
  }

  IMP::Restraints all = restraints;
  all.insert(all.end(), go.begin(), go.end());
  if (com_pull) all.push_back(com_pull);

  IMP_NEW(IMP::core::RestraintsScoringFunction, sf, (all));
  IMP_NEW(IMP::core::RestraintsScoringFunction, mc_sf, (softsphere.empty() ? all : softsphere));

  if (o.init_placement) {
    IMP::ParticleIndexes host, guest;
    pss::placement_sets(system, loaded, mobile_group, fixed_component, host, guest);
    int seed = o.init_seed;
    for (std::size_t i = 0; i < run_name.size(); ++i) {
      seed += static_cast<int>(i + 1) * static_cast<unsigned char>(run_name[i]);
    }
    const double best =
        place_guest_by_score(mc_sf, model, host, guest, o.init_distance_a, o.init_trials, seed);
    out << pss::fmt("Initial score-based placement: distance=%.1fA trials=%d best_score=%.2f",
                    o.init_distance_a, o.init_trials, best)
        << std::endl;
  }

  FFSampling sampling = system.get_sampling();
  if (o.md_steps >= 0) sampling.n_steps = o.md_steps;
  if (o.write_every >= 0) sampling.write_every = o.write_every;
  const int n_steps = sampling.n_steps;
  const int write_every = sampling.write_every;
  const double temperature_k = sampling.temperature_K;
  const double friction_ps = sampling.friction_ps;
  const double timestep_fs = sampling.timestep_fs;
  const int minimize_steps = sampling.minimize_steps;

  const Ids mobile_all = pss::group_ids(system, mobile_group);
  const std::set<std::string> mobile_all_set(mobile_all.begin(), mobile_all.end());
  const std::set<std::string> ring_ids =
      pss::mobile_ring_site_ids(system, loaded.site_atom_names, mobile_component);

  // The rigid body always covers the whole mobile component, so every atom
  // rides an MC move; md_fixed groups say which atoms MD holds still.
  std::vector<Ids> rb_groups(1, mobile_all);
  Ids md_fixed;
  const std::map<std::string, Ids>& md_fixed_cfg = system.get_md_fixed_groups();
  for (std::map<std::string, Ids>::const_iterator it = md_fixed_cfg.begin();
       it != md_fixed_cfg.end(); ++it) {
    for (std::size_t k = 0; k < it->second.size(); ++k) {
      if (mobile_all_set.count(it->second[k])) md_fixed.push_back(it->second[k]);
    }
  }
  std::sort(md_fixed.begin(), md_fixed.end());
  if (md_fixed.empty() && o.freeze_mobile_rings) {
    for (std::set<std::string>::const_iterator it = mobile_all_set.begin();
         it != mobile_all_set.end(); ++it) {
      if (ring_ids.count(*it)) md_fixed.push_back(*it);
    }
  }
  pss::configure_movable(system, loaded, o.fixed_flex_mode, md_fixed);

  std::vector<Ids> nonempty;
  for (std::size_t g = 0; g < rb_groups.size(); ++g) {
    if (!rb_groups[g].empty()) nonempty.push_back(rb_groups[g]);
  }
  rb_groups = nonempty.empty() ? std::vector<Ids>(1, mobile_all) : nonempty;
  std::set<std::string> rb_site_ids;
  for (std::size_t g = 0; g < rb_groups.size(); ++g) {
    rb_site_ids.insert(rb_groups[g].begin(), rb_groups[g].end());
  }

  std::set<std::string> unknown;
  for (std::size_t i = 0; i < md_fixed.size(); ++i) {
    if (!loaded.site.count(md_fixed[i])) unknown.insert(md_fixed[i]);
  }
  if (!unknown.empty()) {
    std::string list;
    for (std::set<std::string>::const_iterator it = unknown.begin(); it != unknown.end(); ++it) {
      list += (list.empty() ? "" : ", ") + *it;
    }
    pss::fail("mmCIF md_fixed ids not in system: " + list);
  }

  IMP::ParticlesTemp movable;
  std::set<std::string> movable_ids;
  for (std::size_t i = 0; i < loaded.site_ids.size(); ++i) {
    IMP::Particle* p = loaded.site.find(loaded.site_ids[i])->second;
    if (IMP::core::XYZ(p).get_coordinates_are_optimized()) {
      movable.push_back(p);
      movable_ids.insert(loaded.site_ids[i]);
    }
  }
  std::size_t mobile_linker_movable = 0;
  for (std::set<std::string>::const_iterator it = movable_ids.begin(); it != movable_ids.end();
       ++it) {
    if (mobile_all_set.count(*it) && !ring_ids.count(*it)) ++mobile_linker_movable;
  }
  out << pss::fmt("DOF: movable=%d / %d (freeze_mobile_rings=%s, fixed_flex_mode=%s)",
                  static_cast<int>(movable.size()), static_cast<int>(loaded.site_ids.size()),
                  o.freeze_mobile_rings ? "True" : "False", o.fixed_flex_mode.c_str())
      << std::endl;
  out << pss::fmt("DOF detail: rb_groups=%d rb_sites=%d md_fixed_sites=%d "
                  "mobile_linker_movable=%d fixed_flex_movable=%d",
                  static_cast<int>(rb_groups.size()), static_cast<int>(rb_site_ids.size()),
                  static_cast<int>(md_fixed.size()), static_cast<int>(mobile_linker_movable),
                  static_cast<int>(flex_ids.size()))
      << std::endl;

  if (minimize_steps > 0 && !movable.empty()) {
    IMP_NEW(IMP::core::ConjugateGradients, cg, (model));
    cg->set_scoring_function(sf);
    cg->optimize(minimize_steps);
  }

  pss::write_system_copy(system, system_cif, out_dir, output_root);

  out << "\n"
      << pss::fmt("%s: %d steps, write_every=%d, mode=%s", run_name.c_str(), n_steps, write_every,
                  o.sampling_mode.c_str())
      << "\n";
  out << pss::fmt("Restraints: total=%d softsphere=%d go=%d", static_cast<int>(all.size()),
                  static_cast<int>(softsphere.size()), static_cast<int>(go.size()))
      << std::endl;

  // particles for the internal displacement report
  const std::set<std::string> md_fixed_set(md_fixed.begin(), md_fixed.end());
  pss::Ps ring_ps, linker_ps;
  for (std::size_t i = 0; i < md_fixed.size(); ++i) {
    std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(md_fixed[i]);
    if (p != loaded.site.end()) ring_ps.push_back(p->second);
  }
  for (std::size_t i = 0; i < mobile_all.size(); ++i) {
    std::map<std::string, IMP::Particle*>::const_iterator p = loaded.site.find(mobile_all[i]);
    if (p == loaded.site.end() || md_fixed_set.count(mobile_all[i])) continue;
    const std::string& name = loaded.site_atom_names.find(mobile_all[i])->second;
    if (!name.empty() && name[0] == 'H') continue;
    linker_ps.push_back(p->second);
  }

  if (o.sampling_mode == "hybrid_md_mc") {
    pss::run_alternating_rb_mc_md(model, mc_sf, sf, loaded.root, out_dir, loaded, movable,
                                  rb_groups, n_steps, write_every, o.mc_steps, temperature_k,
                                  friction_ps, timestep_fs, o.rb_max_translation,
                                  o.rb_max_rotation_deg, o.mc_temperature, o.log_every_frames,
                                  flex_ids, ring_ps, linker_ps, out);
  } else if (o.sampling_mode == "multi_restart") {
    pss::run_multi_restart(model, mc_sf, sf, loaded.root, out_dir, loaded, movable, rb_groups,
                           o.n_restarts, n_steps, write_every, o.mc_steps, o.mc_temperature,
                           o.rb_max_translation, o.rb_max_rotation_deg, temperature_k, friction_ps,
                           timestep_fs, system, mobile_group, o.init_distance_a, o.init_trials,
                           o.init_seed, fixed_component, out);
  } else {
    pss::run_simple_md(model, loaded.root, sf, out_dir, movable, n_steps, write_every,
                       temperature_k, friction_ps, timestep_fs, o.log_every_frames, out);
  }
  out << std::flush;
}

IMPBFF_END_NAMESPACE

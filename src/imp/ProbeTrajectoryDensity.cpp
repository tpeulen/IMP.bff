/**
 *  \file ProbeTrajectoryDensity.cpp
 *  \brief Where a probe went over a trajectory.
 *
 *  Port of `analyze_dye_density` and its helpers from the Python program
 *  `bin/imp_bff` (`analyze-trajectories`). Two things the Python did not do and
 *  this does: it read the component template and the system CIF's sites as
 *  dictionaries (`template.get("features")`, `site.get("component")`), which
 *  the wrapped C++ values stopped being, so the command failed on its first
 *  template; and it resolved its default output and trajectory directories
 *  beside the program file, which a compiled program does not have -- they are
 *  relative to the working directory here.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ProbeTrajectoryDensity.h>

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/ProbeComponentTemplate.h>
#include <IMP/bff/ProbeDataPaths.h>
#include <IMP/bff/ProbeForceFieldCIF.h>
#include <IMP/bff/ProbeLibrary.h>
#include <IMP/bff/TrajectoryAnalysis.h>
#include <IMP/bff/internal/NumpyCompat.h>
#include <IMP/bff/internal/Text.h>

#include <IMP/Model.h>
#include <IMP/atom/Atom.h>
#include <IMP/atom/Hierarchy.h>
#include <IMP/atom/Mass.h>
#include <IMP/core/XYZR.h>
#include <IMP/em/DensityMap.h>
#include <IMP/em/SampledDensityMap.h>
#include <IMP/em/converters.h>
#include <IMP/log.h>

#if !defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF)
#include <IMP/rmf/frames.h>
#include <IMP/rmf/atom_io.h>
#include <RMF/FileConstHandle.h>
#include <RMF/ID.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sys/stat.h>

IMPBFF_BEGIN_NAMESPACE

namespace probe_trajectory_density {

namespace nc = internal::numpy_compat;

std::string strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

std::vector<double> coordinates(const IMP::ParticlesTemp& atoms) {
  std::vector<double> out;
  out.reserve(3 * atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const IMP::algebra::Vector3D c = IMP::core::XYZ(atoms[i]).get_coordinates();
    out.push_back(c[0]);
    out.push_back(c[1]);
    out.push_back(c[2]);
  }
  return out;
}

bool is_dir(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR;
}

//! pathlib's `Path(a) / b`, spelled as `str()` spells it.
std::string join(const std::string& a, const std::string& b) {
  if (a.empty() || a == ".") return b;
  if (!b.empty() && b[0] == '/') return b;
  return a[a.size() - 1] == '/' ? a + b : a + "/" + b;
}

//! `str(Path(p))`: repeated and trailing slashes and `.` components folded.
std::string pathlib_str(const std::string& p) {
  if (p.empty()) return ".";
  const bool absolute = p[0] == '/';
  std::vector<std::string> parts;
  std::string part;
  for (std::size_t i = 0; i <= p.size(); ++i) {
    if (i == p.size() || p[i] == '/') {
      if (!part.empty() && part != ".") parts.push_back(part);
      part.clear();
    } else {
      part += p[i];
    }
  }
  std::string out = absolute ? "/" : "";
  for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? "/" : "") + parts[i];
  if (out.empty()) return ".";
  return out;
}

//! The subdirectories of \p dir, names only, sorted.
std::vector<std::string> subdirectories(const std::string& dir) {
  std::vector<std::string> out;
  const std::vector<std::string> entries = internal::directory_entries(dir, "");
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (is_dir(entries[i])) {
      const std::size_t slash = entries[i].find_last_of('/');
      out.push_back(slash == std::string::npos ? entries[i] : entries[i].substr(slash + 1));
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

double dot3(const double* a, const double* b) { return nc::dot_fma(a, b, 3); }

}  // namespace probe_trajectory_density

std::string probe_atom_name(IMP::Particle* atom,
                            const std::map<int, std::string>& site_name_lookup) {
  const bool is_atom = IMP::atom::Atom::get_is_setup(atom);
  if (is_atom && !site_name_lookup.empty()) {
    const int idx = IMP::atom::Atom(atom).get_input_index();
    std::map<int, std::string>::const_iterator hit = site_name_lookup.find(idx);
    if (hit != site_name_lookup.end()) return hit->second;
    hit = site_name_lookup.find(idx + 1);
    if (hit != site_name_lookup.end()) return hit->second;
  }
  std::string nm;
  if (is_atom) {
    const std::string at = IMP::atom::Atom(atom).get_atom_type().get_string();
    nm = probe_trajectory_density::strip(probe_trajectory_density::replace_all(
            probe_trajectory_density::replace_all(at, "HET: ", ""), "HET:", ""));
  } else {
    nm = atom->get_name();
  }
  const std::size_t slash = nm.find_last_of('/');
  if (slash != std::string::npos) nm = nm.substr(slash + 1);
  return nm;
}

std::vector<double> probe_fixed_axis(const IMP::ParticlesTemp& atoms,
                                     const std::map<int, std::string>& site_name_lookup,
                                     const std::vector<double>& ref_axis,
                                     const std::string& axis_element) {
  namespace ptd = probe_trajectory_density;
  namespace nc = internal::numpy_compat;
  const std::vector<double> coords = ptd::coordinates(atoms);
  std::vector<double> out = point_cloud_principal_axis(coords, true);
  if (out.empty()) {
    IMP_THROW("the fixed component has no atoms", ValueException);
  }
  double* axis = &out[3];
  const double* center = &out[0];
  if (!axis_element.empty()) {
    std::vector<double> marker;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      const std::string nm = probe_atom_name(atoms[i], site_name_lookup);
      if (nm.compare(0, axis_element.size(), axis_element) == 0) {
        marker.push_back(coords[3 * i]);
        marker.push_back(coords[3 * i + 1]);
        marker.push_back(coords[3 * i + 2]);
      }
    }
    if (!marker.empty()) {
      const std::size_t n = marker.size() / 3;
      double v[3];
      for (int k = 0; k < 3; ++k) {
        v[k] = nc::sequential_sum(&marker[k], n, 3) / static_cast<double>(n) - center[k];
      }
      if (ptd::dot3(axis, v) < 0) {
        for (int k = 0; k < 3; ++k) axis[k] = -axis[k];
      }
    }
  }
  if (ref_axis.size() == 3 && ptd::dot3(axis, &ref_axis[0]) < 0) {
    for (int k = 0; k < 3; ++k) axis[k] = -axis[k];
  }
  return out;
}

std::vector<double> probe_long_axis(const IMP::ParticlesTemp& atoms,
                                    const std::vector<double>& ref_axis) {
  namespace ptd = probe_trajectory_density;
  if (atoms.size() < 2) return std::vector<double>();
  const std::vector<double> out = point_cloud_principal_axis(ptd::coordinates(atoms), false);
  std::vector<double> axis(out.begin() + 3, out.end());
  if (ref_axis.size() == 3 && ptd::dot3(&axis[0], &ref_axis[0]) < 0) {
    for (int k = 0; k < 3; ++k) axis[k] = -axis[k];
  }
  return axis;
}

void write_probe_region_density(const std::vector<double>& points, const std::string& path,
                                double resolution, double voxel_size) {
  IMP_NEW(IMP::Model, model, ());
  IMP::ParticlesTemp particles;
  for (std::size_t i = 0; i + 2 < points.size(); i += 3) {
    IMP::Particle* p = new IMP::Particle(model);
    IMP::core::XYZR xyzr = IMP::core::XYZR::setup_particle(p);
    xyzr.set_coordinates(IMP::algebra::Vector3D(points[i], points[i + 1], points[i + 2]));
    xyzr.set_radius(1.0);
    IMP::atom::Mass::setup_particle(p, 1.0);
    particles.push_back(p);
  }
  if (particles.empty()) {
    IMP_THROW("No points available for " << path, ValueException);
  }
  IMP::Pointer<IMP::em::SampledDensityMap> density =
          IMP::em::particles2density(particles, resolution, voxel_size);
  density->std_normalize();
  // the writer by suffix (`.mrc`), made inside IMP.em: a MRCReaderWriter built
  // here and destroyed there owns a filebuf across two C++ runtimes whenever
  // this library and IMP were compiled against different libc++ builds
  IMP::em::write_map(density, path);
}

#if !defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF)

namespace probe_trajectory_density {

struct Regions {
  std::vector<std::string> order;
  std::map<std::string, std::vector<ComponentTemplate::FeatureAtom> > atoms;
};

Regions load_regions(const std::string& template_path) {
  const ComponentTemplate tmpl = read_component_template_cif(template_path);
  const std::map<std::string, std::string> colors = region_features(tmpl);
  Regions out;
  for (std::map<std::string, ComponentTemplate::Feature>::const_iterator it =
               tmpl.features.begin();
       it != tmpl.features.end(); ++it) {
    if (!colors.count(it->first)) continue;
    out.order.push_back(it->first);
    out.atoms[it->first] = it->second.atoms;
  }
  return out;
}

std::map<int, std::string> load_site_name_lookup(const std::string& run_dir,
                                                 const std::string& component) {
  std::map<int, std::string> lookup;
  const std::string system_cif = join(run_dir, "system.cif");
  if (!internal::file_exists(system_cif)) return lookup;
  const ProbeForceFieldSystem system = read_forcefield_cif(system_cif);
  const std::vector<FFSite>& sites = system.get_sites();
  for (std::size_t i = 0; i < sites.size(); ++i) {
    if (sites[i].component == component) lookup[sites[i].site_serial] = sites[i].atom_name;
  }
  return lookup;
}

std::map<std::string, IMP::ParticlesTemp> build_name_map(
        const IMP::ParticlesTemp& atoms, const std::map<int, std::string>& lookup,
        std::vector<std::string>& key_order) {
  bool use_enum = false;
  if (!lookup.empty()) {
    use_enum = true;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      if (IMP::atom::Atom::get_is_setup(atoms[i]) &&
          IMP::atom::Atom(atoms[i]).get_input_index() != -1) {
        use_enum = false;
        break;
      }
    }
  }
  std::map<std::string, IMP::ParticlesTemp> out;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    std::string nm;
    if (use_enum) {
      std::map<int, std::string>::const_iterator hit = lookup.find(static_cast<int>(i) + 1);
      nm = hit != lookup.end() ? hit->second : probe_atom_name(atoms[i]);
    } else {
      nm = probe_atom_name(atoms[i], lookup);
    }
    if (!out.count(nm)) key_order.push_back(nm);
    out[nm].push_back(atoms[i]);
  }
  return out;
}

bool is_alpha(const std::string& s) {
  if (s.empty()) return false;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (!std::isalpha(static_cast<unsigned char>(s[i]))) return false;
  }
  return true;
}

std::map<std::string, IMP::ParticlesTemp> collect_region_atoms(
        const IMP::ParticlesTemp& mobile_atoms, const Regions& regions,
        const std::map<int, std::string>& lookup, std::ostream& log) {
  std::vector<std::string> key_order;
  const std::map<std::string, IMP::ParticlesTemp> by_name =
          build_name_map(mobile_atoms, lookup, key_order);
  std::map<std::string, IMP::ParticlesTemp> selected;
  for (std::size_t r = 0; r < regions.order.size(); ++r) {
    const std::string& region = regions.order[r];
    const std::vector<ComponentTemplate::FeatureAtom>& specs = regions.atoms.find(region)->second;
    IMP::ParticlesTemp atoms;
    std::vector<std::string> missing;
    for (std::size_t s = 0; s < specs.size(); ++s) {
      const std::string& name = specs[s].name;
      const int occ = specs[s].occurrence;
      IMP::ParticlesTemp matches;
      std::map<std::string, IMP::ParticlesTemp>::const_iterator hit = by_name.find(name);
      if (hit != by_name.end()) matches = hit->second;
      if (matches.empty()) {
        // generic name + letters (O3 -> O3A, O3B), in sorted order
        std::vector<std::string> candidates;
        for (std::map<std::string, IMP::ParticlesTemp>::const_iterator it = by_name.begin();
             it != by_name.end(); ++it) {
          if (it->first.compare(0, name.size(), name) == 0 &&
              is_alpha(it->first.substr(name.size()))) {
            candidates.push_back(it->first);
          }
        }
        std::sort(candidates.begin(), candidates.end());
        for (std::size_t c = 0; c < candidates.size(); ++c) {
          const IMP::ParticlesTemp& more = by_name.find(candidates[c])->second;
          matches.insert(matches.end(), more.begin(), more.end());
        }
      }
      if (matches.empty()) {
        missing.push_back(name);
        continue;
      }
      const int idx = occ - 1;
      atoms.push_back((idx >= 0 && idx < static_cast<int>(matches.size())) ? matches[idx]
                                                                           : matches[0]);
    }
    selected[region] = atoms;
    if (!missing.empty()) {
      log << "    Warning: " << region << " missing " << missing.size() << " atoms: ";
      for (std::size_t i = 0; i < missing.size(); ++i) log << (i ? ", " : "") << missing[i];
      log << "\n";
    }
  }
  return selected;
}

std::string resolve_rmf_path(const std::string& traj_root, const std::string& system_name) {
  const std::string base = join(traj_root, system_name);
  const char* const candidates[] = {"rmfs/0.rmf3", "traj.rmf3", "traj_init.rmf3"};
  for (int i = 0; i < 3; ++i) {
    const std::string p = join(base, candidates[i]);
    if (internal::file_exists(p)) return p;
  }
  IMP_THROW("No RMF found for " << system_name << " under " << traj_root, IOException);
}

//! np.median: the middle of a sorted copy, or the mean of the two middles.
double numpy_median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2) return v[n / 2];
  return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double numpy_std(const std::vector<double>& v) {
  const double n = static_cast<double>(v.size());
  const double mean = nc::pairwise_sum(v) / n;
  std::vector<double> sq(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double d = v[i] - mean;
    sq[i] = d * d;
  }
  return std::sqrt(nc::pairwise_sum(sq) / n);
}

void analyze_one(const std::vector<std::string>& traj_roots,
                 const ProbeTrajectoryDensityOptions& o, const std::string& mobile,
                 const std::string& template_path, std::ostream& log) {
  log << "\nAnalyzing " << mobile << "\n";
  log << "  Template: " << template_path << "\n";
  const Regions regions = load_regions(template_path);

  long total_used = 0, total_available = 0;
  std::map<std::string, std::vector<double> > all_points, radial, axial, xy;
  for (std::size_t r = 0; r < regions.order.size(); ++r) {
    all_points[regions.order[r]];
    radial[regions.order[r]];
    axial[regions.order[r]];
    xy[regions.order[r]];
  }
  std::vector<double> axis_ref, pi_axis_ref;
  std::vector<int> o_frame;
  std::vector<double> o_angle, o_abs, o_cos;
  int global_frame_index = 0;

  for (std::size_t t = 0; t < traj_roots.size(); ++t) {
    const std::string& traj_root = traj_roots[t];
    std::string fixed_name = o.fixed_name;
    std::string run_dir;
    if (!o.system_name.empty()) {
      run_dir = o.system_name;
    } else if (!fixed_name.empty()) {
      run_dir = fixed_name + "_" + mobile + "_imp";
    } else {
      const std::string suffix = "_" + mobile + "_imp";
      const std::vector<std::string> dirs = subdirectories(traj_root);
      for (std::size_t i = 0; i < dirs.size(); ++i) {
        if (dirs[i].size() >= suffix.size() &&
            dirs[i].compare(dirs[i].size() - suffix.size(), suffix.size(), suffix) == 0) {
          run_dir = dirs[i];
          break;
        }
      }
      if (run_dir.empty()) {
        IMP_THROW("No run directory ending with '" << suffix << "' found under " << traj_root,
                  IOException);
      }
    }
    const std::string rmf_path = resolve_rmf_path(traj_root, run_dir);
    log << "  RMF: " << rmf_path << "\n";

    RMF::FileConstHandle fh = RMF::open_rmf_file_read_only(rmf_path);
    IMP_NEW(IMP::Model, model, ());
    IMP::set_check_level(IMP::NONE);
    const IMP::atom::Hierarchies roots = IMP::rmf::create_hierarchies(fh, model);
    if (roots.empty()) {
      IMP_THROW("No hierarchies found in " << rmf_path, IOException);
    }
    const IMP::atom::Hierarchy root = roots[0];
    IMP::atom::Hierarchy fixed_hier, mobile_hier;
    std::string found;
    for (unsigned int c = 0; c < root.get_number_of_children(); ++c) {
      const IMP::atom::Hierarchy child = root.get_child(c);
      const std::string nm = child->get_name();
      found += std::string(found.empty() ? "" : "', '") + nm;
      if (nm == mobile) {
        mobile_hier = child;
      } else if (fixed_name.empty() || nm == fixed_name) {
        fixed_hier = child;
      }
    }
    if (!fixed_hier || !mobile_hier) {
      IMP_THROW("Could not find fixed/mobile hierarchies in "
                        << rmf_path << "; found "
                        << (root.get_number_of_children() ? "['" + found + "']" : std::string("[]")),
                IOException);
    }
    if (fixed_name.empty()) fixed_name = fixed_hier->get_name();

    IMP::ParticlesTemp fixed_atoms, mobile_atoms;
    {
      IMP::atom::Hierarchies f = IMP::atom::get_by_type(fixed_hier, IMP::atom::ATOM_TYPE);
      IMP::atom::Hierarchies m = IMP::atom::get_by_type(mobile_hier, IMP::atom::ATOM_TYPE);
      for (std::size_t i = 0; i < f.size(); ++i) fixed_atoms.push_back(f[i].get_particle());
      for (std::size_t i = 0; i < m.size(); ++i) mobile_atoms.push_back(m[i].get_particle());
    }
    const std::string run_path = join(traj_root, run_dir);
    const std::map<int, std::string> fixed_lookup = load_site_name_lookup(run_path, fixed_name);
    const std::map<int, std::string> mobile_lookup = load_site_name_lookup(run_path, mobile);
    if (fixed_atoms.empty()) {
      IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(fixed_hier);
      for (std::size_t i = 0; i < leaves.size(); ++i) fixed_atoms.push_back(leaves[i].get_particle());
    }
    if (mobile_atoms.empty()) {
      IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(mobile_hier);
      for (std::size_t i = 0; i < leaves.size(); ++i) mobile_atoms.push_back(leaves[i].get_particle());
    }

    std::map<std::string, IMP::ParticlesTemp> region_atoms =
            collect_region_atoms(mobile_atoms, regions, mobile_lookup, log);

    // the probe's pi system: every region but the linker, each atom once
    IMP::ParticlesTemp pi_atoms;
    std::set<IMP::ParticleIndex> seen;
    for (std::size_t r = 0; r < regions.order.size(); ++r) {
      if (regions.order[r] == "linker") continue;
      const IMP::ParticlesTemp& atoms = region_atoms[regions.order[r]];
      for (std::size_t i = 0; i < atoms.size(); ++i) {
        if (seen.insert(atoms[i]->get_index()).second) pi_atoms.push_back(atoms[i]);
      }
    }

    const int n_frames_total = static_cast<int>(fh.get_number_of_frames());
    const int n_frames = o.max_frames <= 0 ? n_frames_total : (std::min)(n_frames_total, o.max_frames);
    total_available += n_frames_total;
    total_used += n_frames;

    for (int fi = 0; fi < n_frames; ++fi) {
      IMP::rmf::load_frame(fh, RMF::FrameID(fi));
      const std::vector<double> fixed = probe_fixed_axis(fixed_atoms, fixed_lookup, axis_ref,
                                                         o.axis_element);
      const double center[3] = {fixed[0], fixed[1], fixed[2]};
      const double axis[3] = {fixed[3], fixed[4], fixed[5]};
      if (axis_ref.empty()) axis_ref.assign(axis, axis + 3);

      const std::vector<double> pi_axis = probe_long_axis(pi_atoms, pi_axis_ref);
      if (!pi_axis.empty()) {
        if (pi_axis_ref.empty()) pi_axis_ref = pi_axis;
        double cos_theta = dot3(&pi_axis[0], axis);
        cos_theta = (std::max)(-1.0, (std::min)(1.0, cos_theta));
        const double abs_cos = std::fabs(cos_theta);
        const double angle = std::acos((std::max)(-1.0, (std::min)(1.0, abs_cos))) *
                             (180.0 / 3.141592653589793);
        o_frame.push_back(global_frame_index);
        o_angle.push_back(angle);
        o_abs.push_back(abs_cos);
        o_cos.push_back(cos_theta);
      }
      ++global_frame_index;

      for (std::size_t r = 0; r < regions.order.size(); ++r) {
        const std::string& region = regions.order[r];
        const IMP::ParticlesTemp& atoms = region_atoms[region];
        for (std::size_t i = 0; i < atoms.size(); ++i) {
          const IMP::algebra::Vector3D c = IMP::core::XYZ(atoms[i]).get_coordinates();
          double rel[3] = {c[0] - center[0], c[1] - center[1], c[2] - center[2]};
          all_points[region].push_back(rel[0]);
          all_points[region].push_back(rel[1]);
          all_points[region].push_back(rel[2]);
          radial[region].push_back(nc::norm3(rel));
          const double zproj = dot3(rel, axis);
          axial[region].push_back(zproj);
          double perp[3];
          for (int k = 0; k < 3; ++k) perp[k] = rel[k] - zproj * axis[k];
          xy[region].push_back(nc::norm3(perp));
        }
      }
    }
  }
  log << "  Frames used: " << total_used << "/" << total_available << "\n";

  const std::string mobile_out = join(o.output_dir, mobile);
  internal::make_directory(mobile_out);

  for (std::size_t r = 0; r < regions.order.size(); ++r) {
    const std::string& region = regions.order[r];
    const std::string mrc = join(mobile_out, "occupancy_" + region + ".mrc");
    write_probe_region_density(all_points[region], mrc, o.resolution, o.voxel_size);
    log << "  Wrote grid: " << mrc << "\n";
    const std::string hist = join(mobile_out, "radial_" + region + ".csv");
    write_radial_histogram(radial[region], hist, o.bin_width);
    log << "  Wrote radial histogram: " << hist << "\n";
  }

  std::vector<std::vector<double> > axial_values, xy_values;
  for (std::size_t r = 0; r < regions.order.size(); ++r) {
    axial_values.push_back(axial[regions.order[r]]);
    xy_values.push_back(xy[regions.order[r]]);
  }
  const std::string axis_csv = join(mobile_out, "axis_z_profile_regions.csv");
  write_binned_profile(axis_csv, regions.order, axial_values, o.bin_width, false);
  log << "  Wrote axis profile CSV: " << axis_csv << "\n";
  // a distance from the axis is measured from the axis: the bins start at zero
  const std::string xy_csv = join(mobile_out, "axis_xy_profile_regions.csv");
  write_binned_profile(xy_csv, regions.order, xy_values, o.bin_width, true);
  log << "  Wrote xy profile CSV: " << xy_csv << "\n";
  const std::string orientation_csv = join(mobile_out, "axis_mobile_vs_fixed_orientation.csv");
  write_orientation_profile(orientation_csv, o_frame, o_angle, o_abs, o_cos);
  log << "  Wrote orientation CSV: " << orientation_csv << "\n";

  if (!axis_ref.empty()) {
    namespace nc = internal::numpy_compat;
    const std::string axis_def = join(mobile_out, "fixed_axis_definition.json");
    nc::PyJson summary = nc::PyJson::object();
    if (!o_angle.empty()) {
      summary.set("n_frames", nc::PyJson::integer(static_cast<long long>(o_angle.size())));
      summary.set("mean_angle_deg",
                  nc::PyJson::number(nc::pairwise_sum(o_angle) / static_cast<double>(o_angle.size())));
      summary.set("median_angle_deg", nc::PyJson::number(numpy_median(o_angle)));
      summary.set("std_angle_deg", nc::PyJson::number(numpy_std(o_angle)));
      summary.set("min_angle_deg",
                  nc::PyJson::number(*std::min_element(o_angle.begin(), o_angle.end())));
      summary.set("max_angle_deg",
                  nc::PyJson::number(*std::max_element(o_angle.begin(), o_angle.end())));
    }
    nc::PyJson axis_list = nc::PyJson::array();
    for (int k = 0; k < 3; ++k) axis_list.push(nc::PyJson::number(axis_ref[k]));
    nc::PyJson doc = nc::PyJson::object();
    doc.set("axis_unit_vector", axis_list);
    doc.set("note", nc::PyJson::string("Axis from smallest-variance PCA direction of "
                                       "fixed-component atoms; oriented toward axis_element "
                                       "centroid."));
    doc.set("mobile_vs_fixed_orientation", summary);
    std::ofstream out(axis_def.c_str());
    if (!out) {
      IMP_THROW("cannot write " << axis_def, IOException);
    }
    out << doc.dumps(2);
    out.close();
    log << "  Wrote axis definition: " << axis_def << "\n";
  }
}

}  // namespace probe_trajectory_density

void analyze_probe_trajectories(const ProbeTrajectoryDensityOptions& options, std::ostream& log) {
  namespace ptd = probe_trajectory_density;
  std::vector<std::string> roots;
  if (options.combine_runs) {
    const std::string runs_root = ptd::pathlib_str(
            options.runs_root.empty() ? ptd::join(options.traj_root, "runs") : options.runs_root);
    const std::vector<std::string> dirs = ptd::subdirectories(runs_root);
    for (std::size_t i = 0; i < dirs.size(); ++i) {
      if (dirs[i].compare(0, 3, "run") == 0) roots.push_back(ptd::join(runs_root, dirs[i]));
    }
    if (roots.empty()) {
      IMP_THROW("No run directories found under " << runs_root, IOException);
    }
  } else {
    roots.push_back(ptd::pathlib_str(options.traj_root));
  }
  ProbeTrajectoryDensityOptions o = options;
  o.output_dir = ptd::pathlib_str(options.output_dir);

  std::vector<std::string> templates;
  for (std::size_t i = 0; i < o.mobiles.size(); ++i) {
    templates.push_back(i < o.mobile_template_cifs.size()
                                ? ptd::pathlib_str(o.mobile_template_cifs[i])
                                : ptd::pathlib_str(ptd::join(get_template_dir(),
                                                             o.mobiles[i] + ".template.cif")));
  }
  for (std::size_t i = 0; i < o.mobiles.size(); ++i) {
    if (!internal::file_exists(templates[i])) {
      IMP_THROW("Template CIF not found for mobile component '"
                        << o.mobiles[i] << "': " << templates[i]
                        << "\nPass --mobile-template-cif explicitly.",
                ValueException);
    }
    ptd::analyze_one(roots, o, o.mobiles[i], templates[i], log);
  }
}

#endif

IMPBFF_END_NAMESPACE

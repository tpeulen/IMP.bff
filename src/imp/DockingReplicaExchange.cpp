/**
 *  \file DockingReplicaExchange.cpp
 *  \brief PMI's replica-exchange macro on one replica, and docking from
 *         independent starts -- `imp_bff dock` and `imp_bff dock-errors`.
 *
 *  These were the two docking workflows of the Python program `bin/imp_bff`:
 *  `run_replica_exchange_docking` drove `IMP.pmi.macros.ReplicaExchange`, and
 *  `estimate_errors` a multiprocessing pool of `dock_minimize` trials. Both are
 *  here, call for call where it matters -- the same IMP objects draw the same
 *  random numbers in the same order, so a seeded run reproduces the Python one
 *  -- and file for file where a user reads the output (PMI's run directory
 *  layout, its stat2 files, its best-scoring PDBs and PSF).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/Docking.h>
#include <IMP/bff/DockingPrecision.h>
#include <IMP/bff/internal/CommandLineSubs.h>
#include <IMP/bff/internal/Text.h>

#include <IMP/Model.h>
#include <IMP/RestraintSet.h>
#include <IMP/algebra/Transformation3D.h>
#include <IMP/algebra/vector_generators.h>
#include <IMP/atom/Atom.h>
#include <IMP/atom/Chain.h>
#include <IMP/atom/Copy.h>
#include <IMP/atom/Molecule.h>
#include <IMP/atom/Residue.h>
#include <IMP/atom/Selection.h>
#include <IMP/atom/pdb.h>
#include <IMP/core/Gaussian.h>
#include <IMP/core/GridClosePairsFinder.h>
#include <IMP/core/MonteCarlo.h>
#include <IMP/core/RigidBodyMover.h>
#include <IMP/core/SerialMover.h>
#include <IMP/core/XYZR.h>
#include <IMP/core/provenance.h>
#include <IMP/core/rigid_bodies.h>
#include <IMP/random.h>

//! The trajectories are IMP.rmf files. The module build always has IMP.rmf; a
//! core+imp build that links it says so with IMPBFF_WITH_IMP_RMF, and one that
//! does not docks without writing them.
#if !defined(IMPBFF_STANDALONE) || defined(IMPBFF_WITH_IMP_RMF)
#  define IMPBFF_DOCK_HAS_IMP_RMF 1
#  include <IMP/rmf/atom_io.h>
#  include <IMP/rmf/frames.h>
#  include <RMF/FileHandle.h>
#else
#  define IMPBFF_DOCK_HAS_IMP_RMF 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace dock_rex {

using internal::cli::format;

//! Python's str() of a float: repr, with nan/inf spelled as Python spells them.
std::string py_float(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return internal::cli::json_float(v);
}

//! Python's repr() of a str.
std::string py_str(const std::string& s) {
    const bool has_single = s.find('\'') != std::string::npos;
    const bool has_double = s.find('"') != std::string::npos;
    const char quote = (has_single && !has_double) ? '"' : '\'';
    std::string out(1, quote);
    for (std::size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\\') {
            out += "\\\\";
        } else if (c == static_cast<unsigned char>(quote)) {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20 || c == 0x7f) {
            out += format("\\x%02x", c);
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + quote;
}

std::string key(std::size_t i) { return format("%lu", static_cast<unsigned long>(i)); }

//! `IMP.pmi.get_molecule_name_and_copy`.
std::string molecule_name_and_copy(IMP::atom::Hierarchy h) {
    return IMP::atom::get_molecule_name(h) + "." +
           format("%d", IMP::atom::get_copy_index(h));
}

//! One row of `Output.get_particle_infos_for_pdb_writing`.
struct ParticleInfo {
    IMP::algebra::Vector3D xyz;
    IMP::atom::AtomType atom_type;
    IMP::atom::ResidueType residue_type;
    std::string chain;
    int residue_index;
    double radius;
};

//! The PDB and PSF half of `IMP.pmi.output.Output`, for one hierarchy.
class PmiPdbWriter {
public:
    explicit PmiPdbWriter(IMP::atom::Hierarchy root) : root_(root) {
        // `_init_dictchain`: one chain id per molecule; a repeated id gets a
        // numeric suffix, and the last molecule of a name wins
        std::set<std::string> seen;
        const IMP::atom::Hierarchies molecules =
                IMP::atom::get_by_type(root, IMP::atom::MOLECULE_TYPE);
        for (std::size_t i = 0; i < molecules.size(); ++i) {
            std::string chid = IMP::atom::Chain(molecules[i]).get_id();
            if (chid == std::string(1, '\0')) chid = " ";
            if (seen.count(chid)) {
                std::cerr << "StructureWarning: Duplicate chain ID '" << chid
                          << "' encountered" << std::endl;
                for (int suffix = 1;; ++suffix) {
                    const std::string candidate = chid + format("%d", suffix);
                    if (!seen.count(candidate)) {
                        chid = candidate;
                        break;
                    }
                }
            }
            seen.insert(chid);
            chains_[molecule_name_and_copy(molecules[i])] = chid;
        }
    }

    std::vector<ParticleInfo> particle_infos() const {
        std::vector<ParticleInfo> out;
        IMP::ParticlesTemp ps;
        if (IMP::core::XYZR::get_is_setup(root_) ||
            root_.get_number_of_children() != 0) {
            IMP::atom::Selection sel(root_);
            sel.set_resolution(0);
            ps = sel.get_selected_particles();
        }
        std::map<std::string, std::set<int> > residues_seen;
        for (std::size_t n = 0; n < ps.size(); ++n) {
            IMP::Particle* p = ps[n];
            const std::string protname =
                    molecule_name_and_copy(IMP::atom::Hierarchy(p));
            std::set<int>& seen = residues_seen[protname];
            ParticleInfo info;
            if (IMP::atom::Atom::get_is_setup(p)) {
                IMP::atom::Residue residue(
                        IMP::atom::Atom(p).get_parent().get_particle());
                info.residue_type = residue.get_residue_type();
                info.residue_index = residue.get_index();
                info.atom_type = IMP::atom::Atom(p).get_atom_type();
                seen.insert(info.residue_index);
            } else if (IMP::atom::Residue::get_is_setup(p)) {
                IMP::atom::Residue residue(p);
                if (seen.count(residue.get_index())) continue;
                seen.insert(residue.get_index());
                info.residue_type = residue.get_residue_type();
                info.residue_index = residue.get_index();
                info.atom_type = IMP::atom::AT_CA;
            } else {
                continue;  // a docking assembly carries no beads
            }
            info.xyz = IMP::core::XYZ(p).get_coordinates();
            info.radius = IMP::core::XYZR(p).get_radius();
            std::map<std::string, std::string>::const_iterator c =
                    chains_.find(protname);
            info.chain = c == chains_.end() ? std::string(" ") : c->second;
            out.push_back(info);
        }
        std::stable_sort(out.begin(), out.end(),
                         [](const ParticleInfo& a, const ParticleInfo& b) {
                             if (a.chain.size() != b.chain.size())
                                 return a.chain.size() < b.chain.size();
                             if (a.chain != b.chain) return a.chain < b.chain;
                             return a.residue_index < b.residue_index;
                         });
        return out;
    }

    //! `Output.write_pdb` (`_write_pdb_internal`, not centred).
    void write_pdb(const std::string& path) const {
        const std::vector<ParticleInfo> infos = particle_infos();
        std::ofstream out(path.c_str());
        for (std::size_t n = 0; n < infos.size(); ++n) {
            const ParticleInfo& i = infos[n];
            out << IMP::atom::get_pdb_string(
                    i.xyz, static_cast<int>(n + 1), i.atom_type, i.residue_type,
                    i.chain.empty() ? ' ' : i.chain[0], i.residue_index, ' ', 1.00,
                    i.radius);
        }
        out << "ENDMDL\n";
    }

    //! `Output.write_psf`.
    void write_psf(const std::string& path) const {
        const std::vector<ParticleInfo> infos = particle_infos();
        std::ofstream out(path.c_str());
        out << "PSF CMAP CHEQ\n" << infos.size() << " !NATOM\n";
        std::map<std::string, std::vector<std::pair<int, int> > > by_chain;
        for (std::size_t n = 0; n < infos.size(); ++n) {
            const ParticleInfo& p = infos[n];
            const int atom_index = static_cast<int>(n + 1);
            out << format("%8d %-4s %-4s %-4s %-4s %-4s%14.6f%14.6f%8d%14.6f%14.6f",
                          atom_index, p.chain.c_str(),
                          format("%d", p.residue_index).c_str(),
                          ("\"" + p.residue_type.get_string() + "\"").c_str(), "C",
                          "C", 1.0, 0.0, 0, 0.0, 0.0)
                << "\n";
            by_chain[p.chain].push_back(std::make_pair(atom_index, p.residue_index));
        }
        std::vector<std::pair<int, int> > pairs;
        for (std::map<std::string, std::vector<std::pair<int, int> > >::iterator it =
                     by_chain.begin();
             it != by_chain.end(); ++it) {
            std::vector<std::pair<int, int> > ls = it->second;
            std::stable_sort(ls.begin(), ls.end(),
                             [](const std::pair<int, int>& a,
                                const std::pair<int, int>& b) {
                                 return a.second < b.second;
                             });
            for (std::size_t i = 0; i + 1 < ls.size(); ++i) {
                pairs.push_back(std::make_pair(ls[i].first, ls[i + 1].first));
            }
        }
        out << pairs.size() << " !NBOND: bonds\n";
        for (std::size_t i = 0; i < pairs.size(); i += 4) {
            for (std::size_t k = i; k < pairs.size() && k < i + 4; ++k) {
                out << format("%8d%8d", pairs[k].first, pairs[k].second);
            }
            out << "\n";
        }
    }

private:
    IMP::atom::Hierarchy root_;
    std::map<std::string, std::string> chains_;
};

//! `Output.init_pdb_best_scoring` + `write_pdb_best_scoring`, one prefix.
class BestScoringPdbs {
public:
    BestScoringPdbs(const PmiPdbWriter* writer, const std::string& prefix, int n_best,
                    const std::string& score_file)
        : writer_(writer), prefix_(prefix), n_best_(n_best), score_file_(score_file) {
        write_scores();
        for (int i = 0; i < n_best_; ++i) {
            std::ofstream touch(name(i).c_str());
        }
    }

    std::string name(int i) const { return prefix_ + "." + format("%d", i) + ".pdb"; }

    void add(double score) {
        if (static_cast<int>(scores_.size()) < n_best_) {
            scores_.push_back(score);
            std::stable_sort(scores_.begin(), scores_.end());
            const int index = index_of(score);
            for (int i = static_cast<int>(scores_.size()) - 2; i >= index; --i) {
                if (internal::file_exists(name(i + 1))) std::remove(name(i + 1).c_str());
                std::rename(name(i).c_str(), name(i + 1).c_str());
            }
            writer_->write_pdb(name(index));
        } else if (!scores_.empty() && score < scores_.back()) {
            scores_.push_back(score);
            std::stable_sort(scores_.begin(), scores_.end());
            scores_.pop_back();
            const int index = index_of(score);
            for (int i = static_cast<int>(scores_.size()) - 1; i >= index; --i) {
                std::rename(name(i).c_str(), name(i + 1).c_str());
            }
            std::remove(name(n_best_).c_str());
            writer_->write_pdb(name(index));
        }
        write_scores();
    }

private:
    int index_of(double score) const {
        for (std::size_t i = 0; i < scores_.size(); ++i) {
            if (scores_[i] == score) return static_cast<int>(i);
        }
        return 0;
    }

    void write_scores() const {
        std::ofstream out(score_file_.c_str());
        out << "self.best_score_list=[";
        for (std::size_t i = 0; i < scores_.size(); ++i) {
            out << (i ? ", " : "") << py_float(scores_[i]);
        }
        out << "]\n";
    }

    const PmiPdbWriter* writer_;
    std::string prefix_;
    int n_best_;
    std::string score_file_;
    std::vector<double> scores_;
};

typedef std::vector<std::pair<std::string, std::string> > Items;

//! A Python dict repr in the order given, and PMI's trailing " \n".
std::string stat_line(const Items& items) {
    std::string out = "{";
    for (std::size_t i = 0; i < items.size(); ++i) {
        out += (i ? ", " : "") + items[i].first + ": " + items[i].second;
    }
    return out + "} \n";
}

//! `time.process_time()`.
double process_seconds() { return static_cast<double>(std::clock()) / CLOCKS_PER_SEC; }

//! The header of a stat2 file. PMI's also carried `STAT2HEADER_ENVIRON`, the
//! entire process environment (tokens and keys included); that is left out.
std::string stat_header(const std::vector<std::string>& keys) {
    Items items;
    items.push_back(std::make_pair(py_str("STAT2HEADER"), py_str("STAT2HEADER")));
    items.push_back(std::make_pair(
            py_str("STAT2HEADER_IMP_VERSIONS"),
            py_str("{'IMP_VERSION': " + py_str(IMP::get_module_version()) +
                   ", 'IMP_BFF_VERSION': " + py_str(IMP::bff::get_module_version()) +
                   "}")));
    for (std::size_t i = 0; i < keys.size(); ++i) {
        items.push_back(std::make_pair(key(i), py_str(keys[i])));
    }
    return stat_line(items);
}

void append(const std::string& path, const std::string& text) {
    std::ofstream f(path.c_str(), std::ios::app);
    f << text;
}

//! Every file under \p directory ending in one of \p suffixes, in the order
//! `os.walk` visits them: a directory's own files, sorted, then its
//! subdirectories.
void walk_files(const std::string& directory, const std::vector<std::string>& suffixes,
                std::vector<std::string>& out) {
    std::vector<std::string> files, dirs;
    internal::directory_listing(directory, files, dirs);
    for (std::size_t i = 0; i < files.size(); ++i) {
        for (std::size_t k = 0; k < suffixes.size(); ++k) {
            if (internal::ends_with(files[i], suffixes[k])) {
                out.push_back(internal::cli::path_join(directory, files[i]));
                break;
            }
        }
    }
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        walk_files(internal::cli::path_join(directory, dirs[i]), suffixes, out);
    }
}

}  // namespace dock_rex

void shuffle_configuration(IMP::atom::Hierarchy root, double max_translation,
                           double max_rotation, bool avoid_collision_rb,
                           double cutoff, int niterations) {
    // `get_rbs_and_beads` over the leaves, in leaf order
    const IMP::atom::Hierarchies leaves = IMP::atom::get_leaves(root);
    std::vector<IMP::core::RigidBody> bodies;
    std::set<IMP::ParticleIndex> seen, all_idxs, excluded;
    IMP::ParticlesTemp beads;
    IMP::Model* model = root.get_model();
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        IMP::Particle* p = leaves[i].get_particle();
        if (IMP::core::RigidMember::get_is_setup(p)) {
            IMP::core::RigidBody rb = IMP::core::RigidMember(p).get_rigid_body();
            if (seen.insert(rb.get_particle_index()).second) bodies.push_back(rb);
        } else if (IMP::core::NonRigidMember::get_is_setup(p)) {
            IMP::core::RigidBody rb = IMP::core::NonRigidMember(p).get_rigid_body();
            if (seen.insert(rb.get_particle_index()).second) bodies.push_back(rb);
            beads.push_back(p);
        } else {
            beads.push_back(p);
        }
        if (IMP::core::XYZ::get_is_setup(p)) all_idxs.insert(p->get_index());
        if (IMP::core::Gaussian::get_is_setup(p)) excluded.insert(p->get_index());
    }
    if (bodies.empty() && beads.empty()) {
        IMP_THROW("Could not find any particles in the hierarchy", ValueException);
    }
    if (bodies.empty()) {
        std::cout << "shuffle_configuration: rigid bodies were not initialized"
                  << std::endl;
    }
    for (std::set<IMP::ParticleIndex>::const_iterator it = excluded.begin();
         it != excluded.end(); ++it) {
        all_idxs.erase(*it);
    }
    IMP_NEW(IMP::core::GridClosePairsFinder, finder, ());
    finder->set_distance(cutoff);

    std::cout << "shuffling " << bodies.size() << " rigid bodies" << std::endl;
    for (std::size_t b = 0; b < bodies.size(); ++b) {
        IMP::core::RigidBody rb = bodies[b];
        IMP::ParticleIndexes rb_idxs, other_idxs;
        if (avoid_collision_rb) {
            const IMP::ParticleIndexes members = rb.get_member_particle_indexes();
            std::set<IMP::ParticleIndex> mine;
            for (std::size_t k = 0; k < members.size(); ++k) {
                if (!excluded.count(members[k])) {
                    mine.insert(members[k]);
                    rb_idxs.push_back(members[k]);
                }
            }
            for (std::set<IMP::ParticleIndex>::const_iterator it = all_idxs.begin();
                 it != all_idxs.end(); ++it) {
                if (!mine.count(*it)) other_idxs.push_back(*it);
            }
        }
        int niter = 0;
        while (niter < niterations) {
            const IMP::algebra::Transformation3D t =
                    IMP::algebra::get_random_local_transformation(
                            rb.get_coordinates(), max_translation, max_rotation);
            IMP::core::transform(rb, t);
            if (!(avoid_collision_rb && !other_idxs.empty())) break;
            model->update();
            if (finder->get_close_pairs(model, other_idxs, rb_idxs).empty()) break;
            ++niter;
            if (niter == niterations) {
                IMP_THROW("tried the maximum number of iterations to avoid "
                          "collisions, increase the distance cutoff",
                          ValueException);
            }
        }
    }
    std::cout << "shuffling " << beads.size() << " flexible beads" << std::endl;
    for (std::size_t i = 0; i < beads.size(); ++i) {
        IMP::Particle* fb = beads[i];
        if (!IMP::core::XYZ::get_is_setup(fb)) continue;
        const IMP::algebra::Transformation3D t =
                IMP::algebra::get_random_local_transformation(
                        IMP::core::XYZ(fb).get_coordinates(), max_translation,
                        max_rotation);
        if (IMP::core::NonRigidMember::get_is_setup(fb)) {
            IMP::core::NonRigidMember memb(fb);
            memb.set_internal_coordinates(
                    t.get_transformed(memb.get_internal_coordinates()));
        } else {
            IMP::core::XYZ d(fb);
            IMP::core::transform(d, t);
        }
    }
}

DockingResult dock_replica_exchange(const std::vector<std::string>& pdb_paths,
                                    const std::string& fps_json_path,
                                    const std::string& output_dir,
                                    const DockingParameters& params,
                                    const std::string& initial_poses) {
    using dock_rex::format;
    using dock_rex::key;
    using dock_rex::py_float;
    using dock_rex::py_str;
    internal::make_directory(output_dir);
    DockingAssembly assembly = create_docking_assembly(
            pdb_paths, fps_json_path, params.score_set, params.mean_position_restraint,
            params.ev_weight, params.sigma_da);
    IMP::Model* model = assembly.get_model();
    IMP::atom::Hierarchy root = assembly.get_root();

    // PMI scores what is registered on the model: one set holding the
    // assembly's restraints
    IMP_NEW(IMP::RestraintSet, all_restraints, (model, "All PMI restraints"));
    all_restraints->add_restraint(assembly.get_restraints());

    // `dict(zip(body_of_pdb, rigid_bodies))`: a key keeps its first place and
    // its last value
    const IMP::core::RigidBodies rbs = assembly.get_rigid_bodies();
    const std::vector<int> body_of_pdb = assembly.get_body_of_pdb();
    std::vector<std::pair<int, IMP::ParticleIndex> > bodies;
    for (std::size_t i = 0; i < rbs.size() && i < body_of_pdb.size(); ++i) {
        bool replaced = false;
        for (std::size_t k = 0; k < bodies.size(); ++k) {
            if (bodies[k].first == body_of_pdb[i]) {
                bodies[k].second = rbs[i].get_particle_index();
                replaced = true;
            }
        }
        if (!replaced) {
            bodies.push_back(std::make_pair(body_of_pdb[i], rbs[i].get_particle_index()));
        }
    }
    IMP::core::MonteCarloMovers movers;
    for (std::size_t k = 0; k < bodies.size(); ++k) {
        if (bodies[k].first == params.fixed_body) continue;
        movers.push_back(new IMP::core::RigidBodyMover(
                model, bodies[k].second, params.max_translation, params.max_rotation));
    }
    if (movers.empty() && !bodies.empty()) {
        // a single body: let it move, so the docking means something
        movers.push_back(new IMP::core::RigidBodyMover(
                model, bodies[0].second, params.max_translation, params.max_rotation));
    }

    if (!initial_poses.empty()) apply_poses(assembly, initial_poses);
    if (initial_poses.empty() && params.shuffle_max_translation > 0) {
        try {
            shuffle_configuration(root, params.shuffle_max_translation);
        } catch (const std::exception&) {
            // a failed collision-free placement was passed over, as before
        }
    }

    // ---- the macro --------------------------------------------------------
    IMP_NEW(IMP::core::SerialMover, serial, (movers));
    IMP_NEW(IMP::core::MonteCarlo, mc, (model));
    mc->set_scoring_function(all_restraints);
    mc->set_return_best(false);
    mc->set_score_moved(false);
    mc->set_kt(params.mc_temperature);
    mc->add_mover(serial);
    // One replica: its temperature ladder is the minimum temperature, 1.0,
    // and the replica sets the sampler to it. --temperature never reached it.
    const double replica_temperature = 1.0;
    mc->set_kt(replica_temperature);

    const std::string globaldir = internal::cli::path_join(output_dir, "") + "/";
    const std::string rmf_dir = globaldir + "rmfs/";
    const std::string pdb_dir = globaldir + "pdbs/";
    internal::make_directory(output_dir + "/rmfs");
    internal::make_directory(output_dir + "/pdbs");
    double stopwatch = dock_rex::process_seconds();

    model->update();
    const std::string low_temp_stat_file = globaldir + "stat.0.out";
    std::vector<std::string> stat_keys;
    stat_keys.push_back("Total_Score");
    for (std::size_t k = 0; k < movers.size(); ++k) {
        stat_keys.push_back("MonteCarlo_Acceptance_" + movers[k]->get_name() + "_" +
                            key(k));
    }
    stat_keys.push_back("MonteCarlo_Temperature");
    stat_keys.push_back("MonteCarlo_Nframe");
    stat_keys.push_back("ReplicaExchange_SwapSuccessRatio");
    stat_keys.push_back("ReplicaExchange_MinTempFrequency");
    stat_keys.push_back("ReplicaExchange_MaxTempFrequency");
    stat_keys.push_back("ReplicaExchange_CurrentTemp");
    stat_keys.push_back("Stopwatch_None_delta_seconds");
    stat_keys.push_back("rmf_file");
    stat_keys.push_back("rmf_frame_index");
    {
        std::ofstream f(low_temp_stat_file.c_str());
        f << dock_rex::stat_header(stat_keys);
    }
    const std::string replica_stat_file = globaldir + "stat_replica.0.out";
    {
        std::vector<std::string> keys;
        keys.push_back("ReplicaExchange_SwapSuccessRatio");
        keys.push_back("ReplicaExchange_MinTempFrequency");
        keys.push_back("ReplicaExchange_MaxTempFrequency");
        keys.push_back("ReplicaExchange_CurrentTemp");
        keys.push_back("score");
        std::ofstream f(replica_stat_file.c_str());
        f << dock_rex::stat_header(keys);
    }

    const dock_rex::PmiPdbWriter pdb_writer(root);
    std::unique_ptr<dock_rex::BestScoringPdbs> best;
    if (params.n_best > 0) {
        best.reset(new dock_rex::BestScoringPdbs(&pdb_writer, pdb_dir + "/model",
                                                 params.n_best,
                                                 globaldir + "best.scores.rex.py"));
        pdb_writer.write_psf(pdb_dir + "/model.psf");
    }

#if IMPBFF_DOCK_HAS_IMP_RMF
    {
        RMF::FileHandle initial = RMF::create_rmf_file(globaldir + "initial.0.rmf3");
        IMP::rmf::add_hierarchies(initial, IMP::atom::Hierarchies(1, root));
        IMP::rmf::save_frame(initial);
    }
#endif

    // `_add_provenance`: what sampled this, recorded in the hierarchy
    {
        const IMP::ParticleIndex pi = model->add_particle("sampling");
        IMP::core::SampleProvenance sp = IMP::core::SampleProvenance::setup_particle(
                model, pi, "Monte Carlo", params.n_frames, params.mc_steps);
        sp.set_number_of_replicas(1);
        // IMP.core's `add_imp_provenance` / `add_software_provenance` are Python
        const std::pair<std::string, std::string> software[2] = {
                std::make_pair(std::string("Integrative Modeling Platform (IMP)"),
                               IMP::get_module_version()),
                std::make_pair(std::string("IMP.bff"), IMP::bff::get_module_version())};
        const char* locations[2] = {"https://integrativemodeling.org",
                                    "https://github.com/fluorescence-tools/imp.bff"};
        for (int i = 0; i < 2; ++i) {
            const IMP::ParticleIndex spi = model->add_particle("software");
            IMP::core::SoftwareProvenance prov =
                    IMP::core::SoftwareProvenance::setup_particle(
                            model, spi, software[i].first, software[i].second,
                            locations[i]);
            IMP::core::add_provenance(model, root.get_particle_index(), prov);
        }
        IMP::core::add_provenance(model, root.get_particle_index(), sp);
    }

    const std::string rmfname = rmf_dir + "/0.rmf3";
#if IMPBFF_DOCK_HAS_IMP_RMF
    RMF::FileHandle production = RMF::create_rmf_file(rmfname);
    IMP::rmf::add_hierarchies(production, IMP::atom::Hierarchies(1, root));
#endif

    int ntimes_at_low_temp = 0;
    for (int i = 0; i < params.n_frames; ++i) {
        const double score = mc->optimize(
                static_cast<unsigned int>(params.mc_steps * movers.size()));
        model->update();
        std::cout << "--- frame " << i << " score " << py_float(score) << " "
                  << std::endl;
        std::cout << "--- writing coordinates" << std::endl;
        if (best) best->add(score);
#if IMPBFF_DOCK_HAS_IMP_RMF
        IMP::rmf::save_frame(production);
        production.flush();
#endif

        // stat.0.out: the sampler, the replica and the stopwatch answer first,
        // then the total score, then the extra labels
        dock_rex::Items row;
        std::size_t k = 1;
        for (std::size_t m = 0; m < movers.size(); ++m, ++k) {
            const unsigned proposed = movers[m]->get_number_of_proposed();
            const double rate =
                    proposed ? static_cast<double>(movers[m]->get_number_of_accepted()) /
                                       proposed
                             : 0.0;
            row.push_back(std::make_pair(key(k), py_str(py_float(rate))));
        }
        row.push_back(std::make_pair(key(k++), py_str(py_float(mc->get_kt()))));
        row.push_back(std::make_pair(key(k++), py_str(format("%d", i))));
        for (int r = 0; r < 3; ++r) row.push_back(std::make_pair(key(k++), py_str("0")));
        row.push_back(std::make_pair(key(k++), py_str(py_float(replica_temperature))));
        const double now = dock_rex::process_seconds();
        row.push_back(std::make_pair(key(k++), py_str(py_float(now - stopwatch))));
        stopwatch = now;
        row.push_back(std::make_pair(
                "0", py_str(py_float(all_restraints->evaluate(false)))));
        row.push_back(std::make_pair(key(k++), py_str(rmfname)));
        row.push_back(std::make_pair(key(k++), format("%d", ntimes_at_low_temp)));
        dock_rex::append(low_temp_stat_file, dock_rex::stat_line(row));
        ++ntimes_at_low_temp;

        dock_rex::Items replica_row;
        for (int r = 0; r < 3; ++r) {
            replica_row.push_back(std::make_pair(key(r), py_str("0")));
        }
        replica_row.push_back(std::make_pair("3", py_str(py_float(replica_temperature))));
        replica_row.push_back(std::make_pair("4", py_float(score)));
        dock_rex::append(replica_stat_file, dock_rex::stat_line(replica_row));
    }
    std::cout << "closing production rmf files" << std::endl;
#if IMPBFF_DOCK_HAS_IMP_RMF
    production = RMF::FileHandle();
#endif

    // ---- after the macro ---------------------------------------------------
    const double total = assembly.evaluate();
    const std::vector<PairDistance> pairs = collect_pair_distances(
            assembly.get_network(), assembly.get_mean_position(), assembly.get_sigma_da());
    const std::string score_csv = internal::cli::path_join(output_dir, "scores.csv");
    write_score_csv(score_csv, total, pairs);

    std::vector<std::string> rmfs, pdbs, rmf_suffix, pdb_suffix;
    rmf_suffix.push_back(".rmf3");
    rmf_suffix.push_back(".rmf");
    pdb_suffix.push_back(".pdb");
    dock_rex::walk_files(output_dir, rmf_suffix, rmfs);
    dock_rex::walk_files(output_dir, pdb_suffix, pdbs);
    std::sort(pdbs.begin(), pdbs.end());

    DockingResult result;
    result.score = total;
    result.n_avs = static_cast<int>(assembly.get_network()->get_used_avs().size());
    result.n_distances = static_cast<int>(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) result.pairs.push_back(pairs[i]);
    result.output_dir = output_dir;
    result.rmf_file = rmfs.empty() ? std::string() : rmfs.front();
    result.best_pdbs = pdbs;
    result.score_csv = score_csv;
    result.extra = "{\"n_frames\": " + format("%d", params.n_frames) +
                   ", \"n_movers\": " + key(movers.size()) + "}";
    result.poses = capture_poses(assembly);
    return result;
}

namespace dock_rex {

//! One trial of `imp_bff dock-errors`, seeded by its index.
DockingTrial run_trial(int trial, const std::string& directory,
                       const std::vector<std::string>& pdb_paths,
                       const std::string& fps_json_path,
                       const DockingParameters& params) {
    IMP::random_number_generator.seed(static_cast<unsigned>(trial + 1));
    const DockingResult r = dock_minimize(pdb_paths, fps_json_path, directory, params);
    DockingTrial t;
    t.trial = trial;
    t.score = r.score;
    t.n_distances = r.n_distances;
    t.output_dir = directory;
    t.best_pdb = r.best_pdbs.empty() ? std::string() : r.best_pdbs[0];
    t.score_csv = r.score_csv;
    t.stat_file = internal::cli::path_join(directory, "convergence.csv");
    return t;
}

#ifndef _WIN32
//! A trial as the child process hands it back: score (exact), count, paths.
std::string encode_trial(const DockingTrial& t) {
    std::ostringstream out;
    out << format("%.17g", t.score) << "\n"
        << t.n_distances << "\n"
        << t.best_pdb << "\n"
        << t.score_csv << "\n";
    return out.str();
}

bool decode_trial(const std::string& text, DockingTrial& t) {
    std::istringstream in(text);
    std::string score, count;
    if (!std::getline(in, score) || !std::getline(in, count) ||
        !std::getline(in, t.best_pdb) || !std::getline(in, t.score_csv)) {
        return false;
    }
    t.score = std::strtod(score.c_str(), nullptr);
    t.n_distances = std::atoi(count.c_str());
    return true;
}

//! The trials across up to \p n_workers forked processes. False when a process
//! cannot be started or a trial fails -- the caller then runs them all again,
//! serially, as the Python pool's fallback did.
bool run_trials_forked(const std::vector<std::string>& dirs,
                       const std::vector<std::string>& pdb_paths,
                       const std::string& fps_json_path,
                       const DockingParameters& params, int n_workers,
                       std::vector<DockingTrial>& out) {
    const int n = static_cast<int>(dirs.size());
    std::vector<DockingTrial> results(static_cast<std::size_t>(n));
    std::map<pid_t, std::pair<int, int> > running;  // pid -> (trial, read end)
    int next = 0;
    bool ok = true;
    std::cout << std::flush;
    std::cerr << std::flush;
    while ((ok && next < n) || !running.empty()) {
        while (ok && next < n && static_cast<int>(running.size()) < n_workers) {
            int fds[2];
            if (pipe(fds) != 0) {
                ok = false;
                break;
            }
            const pid_t pid = fork();
            if (pid < 0) {
                close(fds[0]);
                close(fds[1]);
                ok = false;
                break;
            }
            if (pid == 0) {
                close(fds[0]);
                int code = 0;
                try {
                    const std::string line = encode_trial(
                            run_trial(next, dirs[next], pdb_paths, fps_json_path, params));
                    if (write(fds[1], line.data(), line.size()) !=
                        static_cast<ssize_t>(line.size())) {
                        code = 1;
                    }
                } catch (...) {
                    code = 1;
                }
                close(fds[1]);
                std::cout << std::flush;
                _exit(code);
            }
            close(fds[1]);
            running[pid] = std::make_pair(next, fds[0]);
            ++next;
        }
        if (running.empty()) break;
        int status = 0;
        const pid_t done = waitpid(-1, &status, 0);
        if (done < 0) return false;
        std::map<pid_t, std::pair<int, int> >::iterator it = running.find(done);
        if (it == running.end()) continue;
        std::string text;
        char buf[4096];
        ssize_t got;
        while ((got = read(it->second.second, buf, sizeof(buf))) > 0) {
            text.append(buf, static_cast<std::size_t>(got));
        }
        close(it->second.second);
        DockingTrial& t = results[static_cast<std::size_t>(it->second.first)];
        t.trial = it->second.first;
        t.output_dir = dirs[static_cast<std::size_t>(it->second.first)];
        t.stat_file = internal::cli::path_join(t.output_dir, "convergence.csv");
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || !decode_trial(text, t)) {
            ok = false;
        }
        running.erase(it);
    }
    if (!ok) return false;
    out = results;
    return true;
}
#endif

}  // namespace dock_rex

DockingSpread dock_from_independent_starts(const std::vector<std::string>& pdb_paths,
                                           const std::string& fps_json_path,
                                           const std::string& output_dir,
                                           const DockingParameters& params,
                                           int n_trials, int n_workers) {
    internal::make_directory(output_dir);
    std::vector<std::string> dirs;
    for (int i = 0; i < n_trials; ++i) {
        dirs.push_back(internal::cli::path_join(output_dir,
                                                dock_rex::format("trial_%03d", i)));
    }
    if (n_workers <= 0) {
        const int cpus = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        n_workers = std::min(n_trials, cpus);
    }
    n_workers = std::max(1, n_workers);

    DockingSpread spread;
    spread.n_trials = n_trials;
    bool parallel_done = false;
#ifndef _WIN32
    if (n_workers > 1 && n_trials > 1) {
        parallel_done = dock_rex::run_trials_forked(dirs, pdb_paths, fps_json_path,
                                                    params, n_workers, spread.trials);
    }
#endif
    spread.n_workers = parallel_done ? n_workers : 1;
    if (!parallel_done) {
        spread.trials.clear();
        for (int i = 0; i < n_trials; ++i) {
            spread.trials.push_back(
                    dock_rex::run_trial(i, dirs[i], pdb_paths, fps_json_path, params));
        }
    }

    std::vector<double> finite;
    for (std::size_t i = 0; i < spread.trials.size(); ++i) {
        const double s = spread.trials[i].score;
        if (s == s) finite.push_back(s);
    }
    spread.best_trial = -1;
    if (!finite.empty()) {
        // `min(details, key=score)`: the first minimum in trial order; a NaN is
        // never less than anything, but a leading NaN is kept until beaten
        double best_score = 0;
        for (std::size_t i = 0; i < spread.trials.size(); ++i) {
            if (spread.best_trial < 0 || spread.trials[i].score < best_score) {
                spread.best_trial = spread.trials[i].trial;
                best_score = spread.trials[i].score;
            }
        }
        double sum = 0;
        for (std::size_t i = 0; i < finite.size(); ++i) sum += finite[i];
        spread.score_mean = sum / finite.size();
        if (finite.size() > 1) {
            double var = 0;
            for (std::size_t i = 0; i < finite.size(); ++i) {
                var += (finite[i] - spread.score_mean) * (finite[i] - spread.score_mean);
            }
            spread.score_std = std::sqrt(var / finite.size());
        }
    } else {
        spread.score_mean = std::numeric_limits<double>::quiet_NaN();
    }

    // model precision: superpose each run's best model on the fixed body
    std::vector<std::string> best_pdbs;
    for (std::size_t i = 0; i < spread.trials.size(); ++i) {
        if (!spread.trials[i].best_pdb.empty()) best_pdbs.push_back(spread.trials[i].best_pdb);
    }
    if (best_pdbs.size() >= 2) {
        try {
            const std::size_t fixed =
                    params.fixed_body >= 0 &&
                                    params.fixed_body < static_cast<int>(pdb_paths.size())
                            ? static_cast<std::size_t>(params.fixed_body)
                            : 0;
            spread.uncertainty = estimate_position_uncertainty(
                    best_pdbs, pdb_chain_ids(pdb_paths[fixed]));
            if (spread.uncertainty.n_models >= 2) {
                write_position_uncertainty_pdb(
                        spread.uncertainty, best_pdbs[0],
                        internal::cli::path_join(output_dir, "uncertainty.pdb"));
                write_position_uncertainty_csv(
                        spread.uncertainty,
                        internal::cli::path_join(output_dir, "uncertainty.csv"));
                spread.has_uncertainty = true;
            }
        } catch (const std::exception&) {
            spread.has_uncertainty = false;
        }
    }
    return spread;
}

IMPBFF_END_NAMESPACE

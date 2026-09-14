/**
 *  \file FlexibleFitting.cpp
 *  \brief Flexible fitting to FRET distances -- `imp_bff flexfit`.
 *
 *  The Python program built this from IMP.pmi's `System`, CHARMM, the
 *  kinematics module and two `IMP.OptimizerState` subclasses; everything it
 *  called is C++, so the whole of it is here, step for step. See
 *  include/FlexibleFitting.h.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/FlexibleFitting.h>

//! The module build needs `kinematics` among its required modules; a core+imp
//! build that links IMP.kinematics and IMP.rmf says so with
//! IMPBFF_WITH_IMP_KINEMATICS and IMPBFF_WITH_IMP_RMF.
#if !defined(IMPBFF_STANDALONE) || \
        (defined(IMPBFF_WITH_IMP_KINEMATICS) && defined(IMPBFF_WITH_IMP_RMF))
#  define IMPBFF_FLEXFIT_AVAILABLE 1
#else
#  define IMPBFF_FLEXFIT_AVAILABLE 0
#endif

#if !IMPBFF_FLEXFIT_AVAILABLE
#include <IMP/bff/IMPCompatibility.h>
IMPBFF_BEGIN_NAMESPACE
std::string flexible_fitting(const std::string&, const std::string&,
                             const FlexibleFittingParameters&) {
    IMP_THROW("flexible fitting needs IMP.kinematics and IMP.rmf, which this "
              "build does not link",
              ValueException);
}
IMPBFF_END_NAMESPACE
#else

#include <IMP/bff/IMPHierarchyBridge.h>
#include <IMP/bff/ProbeAccessibleVolumeMeanDistanceRestraint.h>
#include <IMP/bff/internal/CommandLineSubs.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

#include <IMP/Model.h>
#include <IMP/OptimizerState.h>
#include <IMP/Particle.h>
#include <IMP/RestraintSet.h>
#include <IMP/atom/CHARMMParameters.h>
#include <IMP/atom/Residue.h>
#include <IMP/atom/StereochemistryPairFilter.h>
#include <IMP/atom/charmm_segment_topology.h>
#include <IMP/atom/pdb.h>
#include <IMP/container/ClosePairContainer.h>
#include <IMP/container/ListSingletonContainer.h>
#include <IMP/container/generic.h>
#include <IMP/core/MonteCarlo.h>
#include <IMP/core/RestraintsScoringFunction.h>
#include <IMP/core/SerialMover.h>
#include <IMP/core/SphereDistancePairScore.h>
#include <IMP/kinematics/KinematicForestScoreState.h>
#include <IMP/kinematics/ProteinKinematics.h>
#include <IMP/kinematics/RevoluteJointMover.h>
#include <IMP/rmf/atom_io.h>
#include <IMP/rmf/frames.h>
#include <IMP/rmf/restraint_io.h>
#include <RMF/FileConstHandle.h>
#include <RMF/FileHandle.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef _WIN32
#  include <unistd.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace flexfit {

using internal::cli::format;

//! `os.path.splitext(path)[0]`.
std::string stem_of(const std::string& path) {
    return internal::cli::path_with_suffix(path, "");
}

//! `WriteRMFFrame`: a frame of the hierarchy and its restraints per update.
class RmfFrameWriter : public IMP::OptimizerState {
public:
    RmfFrameWriter(const std::string& filename, IMP::atom::Hierarchy root,
                   const IMP::RestraintsTemp& restraints)
        : IMP::OptimizerState(root.get_model(), "WriteRMFFrame") {
        const std::string base = stem_of(filename);
        std::string fn = base + ".0.rmf3";
        if (internal::file_exists(fn)) {
            for (int i = 0; i < 100; ++i) {
                fn = base + format(".%d.rmf3", i);
                if (!internal::file_exists(fn)) break;
            }
        }
        filename_ = fn;
        fh_ = RMF::create_rmf_file(fn);
        IMP::rmf::add_hierarchies(fh_, IMP::atom::Hierarchies(1, root));
        IMP::rmf::add_restraints(fh_, restraints);
        IMP::rmf::save_frame(fh_);
    }
    const std::string& get_filename() const { return filename_; }
    void close() { fh_ = RMF::FileHandle(); }

protected:
    virtual void do_update(unsigned int) override { IMP::rmf::save_frame(fh_); }

private:
    std::string filename_;
    RMF::FileHandle fh_;
};

//! `WritePDBFrame`: one PDB per update and a line of restraint scores.
class PdbFrameWriter : public IMP::OptimizerState {
public:
    PdbFrameWriter(const std::string& filename, IMP::atom::Hierarchy root,
                   const IMP::RestraintsTemp& restraints)
        : IMP::OptimizerState(root.get_model(), "WriteDCDFrame"),
          basename_(filename), restraint_filename_(stem_of(filename) + ".rst.txt"),
          root_(root), restraints_(restraints), frame_(0) {}
    const std::string& get_filename() const { return restraint_filename_; }

protected:
    virtual void do_update(unsigned int) override {
        {
            std::ofstream fp(restraint_filename_.c_str(), std::ios::app);
            fp << frame_ << "\t";
            for (std::size_t i = 0; i < restraints_.size(); ++i) {
                fp << (i ? "\t" : "") << format("%.3f", restraints_[i]->evaluate(false));
            }
            fp << "\t" << "\n";
        }
        const std::string out = stem_of(basename_) + "_state_0_" +
                                format("%04d", frame_) + ".pdb";
        IMP::atom::write_pdb(root_, out);
        ++frame_;
    }

private:
    std::string basename_, restraint_filename_;
    IMP::atom::Hierarchy root_;
    IMP::RestraintsTemp restraints_;
    int frame_;
};

//! A path for a temporary PDB, as `tempfile.mktemp('.pdb')` gave one.
std::string temporary_pdb() {
#ifndef _WIN32
    const char* dir = std::getenv("TMPDIR");
    std::string pattern = std::string(dir && *dir ? dir : "/tmp") + "/imp_bff_flexfitXXXXXX.pdb";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    const int fd = mkstemps(&buf[0], 4);
    if (fd >= 0) close(fd);
    return std::string(&buf[0]);
#else
    return std::string(std::tmpnam(nullptr)) + ".pdb";
#endif
}

}  // namespace flexfit

std::string flexible_fitting(const std::string& input_pdb,
                             const std::string& labeling_json,
                             const FlexibleFittingParameters& params) {
    IMP_NEW(IMP::Model, model, ());
    // `IMP.pmi.topology.System(model).build()` with no states: one node
    IMP::atom::Hierarchy root_hier =
            IMP::atom::Hierarchy::setup_particle(new IMP::Particle(model));
    root_hier->set_name("System");

    // `os.path.splitext(output)[-1].lower() == ".rmf3"`
    const std::string suffix = internal::cli::path_suffix(params.output);
    const bool write_rmf = suffix.size() == 5 && internal::ends_with(suffix, ".rmf3");

    std::string pdb_path = input_pdb;
    bool do_anneal = true;
    if (internal::file_exists(params.output)) {
        // continue a run: the last frame of the trajectory is the start
        IMP_NEW(IMP::Model, previous, ());
        RMF::FileConstHandle f = RMF::open_rmf_file_read_only(params.output);
        const unsigned n_frames = f.get_number_of_frames();
        IMP::rmf::load_frame(f, RMF::FrameID(n_frames > 0 ? n_frames - 1 : 0));
        const IMP::atom::Hierarchies loaded = IMP::rmf::create_hierarchies(f, previous);
        IMP::rmf::load_frame(f, RMF::FrameID(n_frames > 0 ? n_frames - 1 : 0));
        if (loaded.empty() || loaded[0].get_number_of_children() == 0) {
            IMP_THROW(params.output << " holds no structure to continue from",
                      IOException);
        }
        pdb_path = flexfit::temporary_pdb();
        IMP::atom::write_pdb(loaded[0].get_child(0), pdb_path);
        do_anneal = false;
    }

    IMP::atom::Hierarchy mhd = IMP::atom::read_pdb(
            pdb_path, model, new IMP::atom::NonWaterNonHydrogenPDBSelector(), true,
            false);
    root_hier.add_child(mhd);
    const IMP::atom::Hierarchies atom_hierarchies =
            IMP::atom::get_by_type(mhd, IMP::atom::ATOM_TYPE);
    IMP::ParticlesTemp atoms;
    for (std::size_t i = 0; i < atom_hierarchies.size(); ++i) {
        atoms.push_back(atom_hierarchies[i].get_particle());
    }

    IMP_NEW(IMP::atom::CHARMMParameters, ff,
            (IMP::atom::get_data_path("top_heav.lib"), IMP::atom::get_data_path("par.lib")));
    ff->add_radii(mhd, params.radii_scaling);
    IMP::Pointer<IMP::atom::CHARMMTopology> topology = ff->create_topology(mhd);
    topology->apply_default_patches();
    topology->add_atom_types(mhd);
    IMP::Particles bonds = topology->add_bonds(mhd);

    // the flexible residues: one FlexFit block, or the one named
    const nlohmann::json document =
            nlohmann::json::parse(internal::cli::read_text_file(labeling_json));
    if (!document.contains("FlexFit")) {
        IMP_THROW(labeling_json << " has no FlexFit block", ValueException);
    }
    const nlohmann::json& blocks = document["FlexFit"];
    nlohmann::json flex_dict;
    if (blocks.size() > 1) {
        if (!blocks.contains(params.flex_set)) {
            IMP_THROW(labeling_json << " has " << blocks.size()
                                    << " FlexFit blocks; name one with --flex-set ('"
                                    << params.flex_set << "' is not one of them)",
                      ValueException);
        }
        flex_dict = blocks[params.flex_set];
    } else if (blocks.size() == 1) {
        flex_dict = blocks.begin().value();
    } else {
        IMP_THROW(labeling_json << " has an empty FlexFit block", ValueException);
    }
    const FlexFitSelection selection = read_angle_file(mhd, flex_dict.dump());
    const IMP::ParticlesTemp flexible = selection.get_residues();
    const IMP::atom::Bonds extra_bonds = selection.get_bonds();
    for (std::size_t i = 0; i < extra_bonds.size(); ++i) {
        bonds.push_back(extra_bonds[i].get_particle());
    }
    const IMP::Particles angles = ff->create_angles(bonds);
    const IMP::Particles dihedrals = ff->create_dihedrals(bonds);

    IMP::atom::Residues flexible_residues;
    for (std::size_t i = 0; i < flexible.size(); ++i) {
        flexible_residues.push_back(IMP::atom::Residue(flexible[i]));
    }
    IMP_NEW(IMP::kinematics::ProteinKinematics, pk,
            (mhd, flexible_residues, IMP::ParticleIndexQuads()));
    const IMP::kinematics::DihedralAngleRevoluteJoints ordered = pk->get_ordered_joints();
    IMP::kinematics::RevoluteJoints joints;
    for (std::size_t i = 0; i < ordered.size(); ++i) joints.push_back(ordered[i]);
    IMP_NEW(IMP::kinematics::KinematicForestScoreState, kfss,
            (pk->get_kinematic_forest(), pk->get_rigid_bodies(), atoms));
    model->add_score_state(kfss);
    IMP_NEW(IMP::kinematics::RevoluteJointMover, mover, (model, joints));

    // soft spheres over close pairs that are not stereochemical partners
    IMP_NEW(IMP::atom::StereochemistryPairFilter, pair_filter, ());
    pair_filter->set_bonds(bonds);
    pair_filter->set_angles(angles);
    pair_filter->set_dihedrals(dihedrals);
    IMP_NEW(IMP::container::ListSingletonContainer, lsc,
            (model, IMP::get_indexes(atoms)));
    IMP_NEW(IMP::container::ClosePairContainer, cpc, (lsc, 15.0));
    cpc->add_pair_filter(pair_filter);
    IMP_NEW(IMP::core::SoftSpherePairScore, soft, (1));
    IMP::Pointer<IMP::Restraint> pr = IMP::container::create_restraint(soft.get(), cpc.get());
    pr->set_name("stereochemistry");

    // the FRET network
    IMP::Pointer<IMP::RestraintSet> av_rs = probe_network_restraint_set(
            IMP::core::Hierarchy(root_hier), labeling_json, "ProbeNetworkRestraint",
            params.score_set, params.mode == "F");
    IMP::RestraintsTemp all_restraints;
    all_restraints.push_back(av_rs);
    all_restraints.push_back(pr);
    IMP_NEW(IMP::core::RestraintsScoringFunction, sf, (all_restraints));

    IMP_NEW(IMP::core::MonteCarlo, s, (model));
    s->set_scoring_function(sf);
    s->set_return_best(false);
    IMP::core::MonteCarloMovers movers(1, mover);
    IMP_NEW(IMP::core::SerialMover, sm, (movers));
    s->add_mover(sm);

    std::string written;
    IMP::Pointer<flexfit::RmfFrameWriter> rmf_state;
    if (write_rmf) {
        rmf_state = new flexfit::RmfFrameWriter(params.output, root_hier, all_restraints);
        rmf_state->set_period(params.period);
        s->add_optimizer_state(rmf_state);
        written = rmf_state->get_filename();
    } else {
        IMP::Pointer<flexfit::PdbFrameWriter> pdb_state =
                new flexfit::PdbFrameWriter(params.output, root_hier, all_restraints);
        pdb_state->set_period(params.period);
        s->add_optimizer_state(pdb_state);
        written = pdb_state->get_filename();
    }

    if (do_anneal) {
        // numpy.logspace(ln(4T), ln(T), 25): powers of ten of natural logs
        // -- linspace's own arithmetic, i * step + start and the last point
        // set to stop, so the ladder is numpy's to the last bit
        const int n_anneal = 25;
        const double start = std::log(4.0 * params.temperature);
        const double stop = std::log(params.temperature);
        const double step = (stop - start) / (n_anneal - 1);
        for (int i = 0; i < n_anneal; ++i) {
            const double exponent = i == n_anneal - 1 ? stop : i * step + start;
            s->set_kt(std::pow(10.0, exponent));
            s->optimize(10);
        }
    }
    for (int frame = 0; frame < params.num_frames; frame += std::max(1, params.period)) {
        s->optimize(static_cast<unsigned int>(params.period));
    }
    if (rmf_state) rmf_state->close();
    return written;
}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FLEXFIT_AVAILABLE

/**
 *  \file IMP/bff/FlexibleFitting.h
 *  \brief Flexible fitting of a structure to FRET distances by sampling its
 *         backbone dihedrals.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FLEXIBLEFITTING_H
#define IMPBFF_FLEXIBLEFITTING_H

#include <IMP/bff/bff_config.h>

#include <string>

IMPBFF_BEGIN_NAMESPACE

#ifndef SWIG
//! What `imp_bff flexfit` is told.
struct IMPBFFEXPORT FlexibleFittingParameters {
    //! The distance set of the labelling file that is scored.
    std::string score_set;
    //! The `FlexFit` block naming the flexible residues; may be empty when
    //! the file has exactly one.
    std::string flex_set;
    //! Sampling steps after the anneal, taken `period` at a time.
    int num_frames;
    //! The trajectory: `.rmf3` writes `<stem>.<i>.rmf3`, anything else one PDB
    //! per written frame beside a `<stem>.rst.txt` restraint log.
    std::string output;
    //! Monte-Carlo steps between written frames.
    int period;
    //! kT of the sampling; the anneal starts from four times it.
    double temperature;
    //! Scales the CHARMM radii the soft-sphere clash term sees.
    double radii_scaling;
    //! `F`: the volumes are computed once and their mean positions ride on the
    //! rigid bodies; anything else rebuilds them every evaluation.
    std::string mode;

    FlexibleFittingParameters()
        : num_frames(15000), output("out.rmf3"), period(10), temperature(2.0),
          radii_scaling(0.7), mode("F") {}
};

//! Sample a structure's flexible dihedrals against the FRET network of a file.
/*!
    `imp_bff flexfit`. The structure is read without waters and hydrogens and
    given CHARMM (`top_heav.lib`, `par.lib`) radii scaled by
    FlexibleFittingParameters::radii_scaling, bonds, angles and dihedrals; the
    backbone of the residues the file's `FlexFit` block names (and any bonds
    it declares) become `IMP::kinematics` revolute joints moved by one
    `RevoluteJointMover`. The score is the file's FRET network
    (#probe_network_restraint_set, mean positions in mode `F`) plus a
    soft-sphere clash term over close pairs that are not bonded, angle or
    dihedral partners. `IMP::core::MonteCarlo` anneals over 25 temperatures for
    10 steps each and then samples.

    The anneal ladder is `numpy.logspace(ln(4T), ln(T), 25)` -- **base 10** of
    natural logarithms, so for T = 2 it runs from 120 down to 4.9 rather than
    from 8 to 2. That is what the Python program ran and it is kept, so a
    trajectory from this function is a trajectory of that command.

    When FlexibleFittingParameters::output already exists it is read as an RMF
    and sampling continues from its last frame without the anneal.

    \return the trajectory written: the RMF file, or the restraint log of a
            PDB run
    \throw IOException when a file cannot be read
    \throw ValueException when the labelling file has several `FlexFit` blocks
            and none is named
*/
IMPBFFEXPORT std::string flexible_fitting(const std::string& input_pdb,
                                          const std::string& labeling_json,
                                          const FlexibleFittingParameters& params);
#endif

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FLEXIBLEFITTING_H

/**
 *  \file IMP/bff/ProbeModelComparison.h
 *  \brief Two label models of the same site, side by side: an accessible
 *         volume and a screened rotamer ensemble.
 *
 *  What `imp_bff av-vs-rotamer` records into `okf/validation/av_vs_rotamer.md`
 *  and `test/references/cgprobe_av_vs_rotamer_pins.json` (PRD-108 stage 2;
 *  formerly `IMP.bff.representation.compare`, then Python in `bin/imp_bff`).
 *  Per position: the volume's size and extent, the ensemble's, how far their
 *  mean positions are apart, the ensemble's partition function and how much of
 *  its weight sits within 2.5 A of the protein. Per pair: R_mp, <R_DA>,
 *  <R_DA>_E and sigma_R from both models, the ensemble's <kappa^2>, and chi^2
 *  against the experiment where one is recorded.
 *
 *  The numbers are recorded, not gated: the two are different physical models
 *  and their disagreement is the result. The arithmetic follows the Python
 *  program's numpy exactly, so a pin file written here matches one written
 *  there.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_PROBEMODELCOMPARISON_H
#define IMPBFF_PROBEMODELCOMPARISON_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <string>
#include <utility>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A measured distance a comparison is scored against.
struct IMPBFFEXPORT AVRotamerExperiment {
  std::string position1, position2, name, distance_type;
  double distance, error_neg, error_pos;
  AVRotamerExperiment() : distance_type("RDAMean"), distance(0), error_neg(0), error_pos(0) {}
  IMP_SHOWABLE_INLINE(AVRotamerExperiment, out << "AVRotamerExperiment(" << position1 << "-"
                                               << position2 << ")");
};
IMP_VALUES(AVRotamerExperiment, AVRotamerExperiments);

//! One comparison: a structure, its positions, a library per position, the pairs.
struct IMPBFFEXPORT AVRotamerCase {
  //! `hgbp1` or `t4l`.
  std::string system;
  std::string title, pdb;
  //! Position names, in the order the fps.json lists them.
  std::vector<std::string> names;
  //! The fps.json position of each name, as JSON text.
  std::vector<std::string> positions_json;
  //! The rotamer library of each name.
  std::vector<std::string> libraries;
  std::vector<std::pair<std::string, std::string> > pairs;
  std::vector<AVRotamerExperiment> experiments;
  IMP_SHOWABLE_INLINE(AVRotamerCase, out << "AVRotamerCase(" << system << ", "
                                         << names.size() << " positions)");
};
IMP_VALUES(AVRotamerCase, AVRotamerCases);

//! What one position of a comparison came out as.
struct IMPBFFEXPORT AVRotamerPosition {
  std::string name;
  double d_mean_position;
  int av_n_points, n_rotamers;
  //! RMS distance of the weighted cloud from its mean, A; NaN for an empty cloud.
  double av_extent, rot_extent;
  double partition;
  //! Ensemble weight on rotamers with a heavy atom within 2.5 A of the protein.
  double interpenetration_weight;
  //! Over rotamers of weight > 1e-3; infinite when there are none.
  double min_heavy_atom_distance;
  AVRotamerPosition()
      : d_mean_position(0), av_n_points(0), n_rotamers(0), av_extent(0), rot_extent(0),
        partition(0), interpenetration_weight(0), min_heavy_atom_distance(0) {}
  IMP_SHOWABLE_INLINE(AVRotamerPosition, out << "AVRotamerPosition(" << name << ")");
};
IMP_VALUES(AVRotamerPosition, AVRotamerPositions);

//! What one pair of a comparison came out as.
struct IMPBFFEXPORT AVRotamerPair {
  std::string pair, p1, p2;
  double Rmp_av, Rmp_rot, RDAMean_av, RDAMean_rot, RDAMeanE_av, RDAMeanE_rot;
  double sigma_av, sigma_rot, kappa2_rot;
  //! False when either ensemble has Z below #av_rotamer_z_cutoff.
  bool rotamer_valid;
  //! True when the case records an experiment for this pair.
  bool has_experiment;
  double exp_distance;
  std::string exp_type;
  double chi2_av, chi2_rot;
  AVRotamerPair()
      : Rmp_av(0), Rmp_rot(0), RDAMean_av(0), RDAMean_rot(0), RDAMeanE_av(0), RDAMeanE_rot(0),
        sigma_av(0), sigma_rot(0), kappa2_rot(2.0 / 3.0), rotamer_valid(true),
        has_experiment(false), exp_distance(0), chi2_av(0), chi2_rot(0) {}
  IMP_SHOWABLE_INLINE(AVRotamerPair, out << "AVRotamerPair(" << pair << ")");
};
IMP_VALUES(AVRotamerPair, AVRotamerPairs);

//! FRETpredict's partition-function cutoff: below it every rotamer clashes and
//! the ensemble is the uniform fallback, which is excluded from the chi^2 sums.
inline double av_rotamer_z_cutoff() { return 0.05; }

//! R0 of the Alexa488/Alexa594 pairs the reference cases use, A.
inline double av_rotamer_forster_radius() { return 52.0; }

//! A reference case.
/*!
    `hgbp1`: 1DG3 chain A, the `F` positions of hGBP1.fps.json except the
    unresolved 254, donor Alexa488 C1R at 481 against Alexa594 C1R elsewhere.
    `t4l`: 3GUN with every position and distance of fret.fps.json, Alexa488 C1R
    at the `D` positions, Alexa594 C1R at the `A` ones.

    \param[in] system `hgbp1` or `t4l`
    \param[in] cutoff the rotamer-library cutoff, e.g. 30 or 10
    \throw ValueException for another system
*/
IMPBFFEXPORT AVRotamerCase av_rotamer_case(const std::string& system, int cutoff = 30);

//! The accessible volume and the rotamer ensemble of every position of a case.
IMPBFFEXPORT std::vector<AVRotamerPosition> compare_av_and_rotamer_positions(
        const AVRotamerCase& reference_case, int n_samples = 50000,
        double temperature = 298.15);

//! Both models' distance statistics for every pair of a case.
/*! Builds the volumes and ensembles again, deterministically, so it can be
    called on its own. */
IMPBFFEXPORT std::vector<AVRotamerPair> compare_av_and_rotamer_pairs(
        const AVRotamerCase& reference_case, int n_samples = 50000,
        double temperature = 298.15, double forster_radius = 52.0);

#ifndef SWIG
//! Positions and pairs in one pass (the volumes and ensembles built once).
IMPBFFEXPORT void compare_av_and_rotamer(const AVRotamerCase& reference_case, int n_samples,
                                         double temperature, double forster_radius,
                                         std::vector<AVRotamerPosition>& positions,
                                         std::vector<AVRotamerPair>& pairs);
#endif

//! The comparison as the two markdown tables of `av_vs_rotamer.md`.
IMPBFFEXPORT std::string av_rotamer_markdown_table(const std::vector<AVRotamerPosition>& positions,
                                                   const std::vector<AVRotamerPair>& pairs,
                                                   const std::string& title);

//! The pin-able numbers of a comparison, as the pin file's JSON object.
/*! \param[in] indent,depth `json.dumps` layout of the enclosing document */
IMPBFFEXPORT std::string av_rotamer_summary_json(const std::vector<AVRotamerPosition>& positions,
                                                 const std::vector<AVRotamerPair>& pairs,
                                                 int indent = 1, int depth = 0);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PROBEMODELCOMPARISON_H

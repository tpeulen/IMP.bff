/*
 * An accessible volume and a rotamer ensemble of the same site, side by side
 * (ProbeModelComparison.h, connection layer): what `imp_bff av-vs-rotamer`
 * records. The same `std::vector` member trap as IMP_bff.docking.i applies to
 * `AVRotamerCase`: bind the case to a name before reading a member off it.
 */

IMP_SWIG_VALUE(IMP::bff, AVRotamerExperiment, AVRotamerExperiments);
IMP_SWIG_VALUE(IMP::bff, AVRotamerCase, AVRotamerCases);
IMP_SWIG_VALUE(IMP::bff, AVRotamerPosition, AVRotamerPositions);
IMP_SWIG_VALUE(IMP::bff, AVRotamerPair, AVRotamerPairs);

%feature("kwargs") IMP::bff::av_rotamer_case;
%feature("kwargs") IMP::bff::compare_av_and_rotamer_positions;
%feature("kwargs") IMP::bff::compare_av_and_rotamer_pairs;
%feature("kwargs") IMP::bff::av_rotamer_markdown_table;
%feature("kwargs") IMP::bff::av_rotamer_summary_json;

%include "IMP/bff/ProbeModelComparison.h"

%template(AVRotamerPositionList) std::vector<IMP::bff::AVRotamerPosition>;
%template(AVRotamerPairList) std::vector<IMP::bff::AVRotamerPair>;
%template(AVRotamerExperimentList) std::vector<IMP::bff::AVRotamerExperiment>;

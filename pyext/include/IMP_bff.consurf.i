/*
 * ConSurf end to end, native: search, homologues, alignment, rates, grades.
 */

IMP_SWIG_VALUE(IMP::bff, ConsurfOptions, ConsurfOptionsList);
IMP_SWIG_VALUE(IMP::bff, ConsurfResult, ConsurfResults);

%feature("compactdefaultargs") IMP::bff::compute_consurf;
%feature("compactdefaultargs") IMP::bff::compute_consurf_from_msa;
%feature("compactdefaultargs") IMP::bff::get_consurf_grades_text;
%feature("compactdefaultargs") IMP::bff::write_consurf_grades;

%include "IMP/bff/Consurf.h"

/* Fast-NPS direct dye and dye-model metadata, plus the thin Bayesian
   direct-FRET likelihood restraint on live existing IMP label sites. */
IMP_SWIG_VALUE(IMP::bff, NPSDirectDye, NPSDirectDyes);
IMP_SWIG_VALUE(IMP::bff, NPSDyeModel, NPSDyeModels);
IMP_SWIG_OBJECT(IMP::bff, NPSIsotropicFRETEfficiencyRestraint,
                NPSIsotropicFRETEfficiencyRestraints);
%feature("kwargs") IMP::bff::NPSIsotropicFRETEfficiencyRestraint::NPSIsotropicFRETEfficiencyRestraint;
%include "IMP/bff/NPS.h"
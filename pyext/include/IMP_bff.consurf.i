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

/* ConsurfOptions' nested options (search, clusters, homologs, conservation)
   are value members: SWIG hands out a copy, so `o.clusters.min_homologues = 600`
   used to set a field of that copy and change nothing, silently. The getter
   now returns the copy as an instance of a subclass whose field assignment
   writes the struct back into its parent. It is still an instance of the
   real class, so it passes to C++ functions as before. */
%pythoncode %{
def _write_back_members(cls, names):
    for name in names:
        member = getattr(cls, name)

        def make(member):
            bound = {}

            def fget(parent):
                value = member.fget(parent)
                base = type(value)
                if base not in bound:
                    base_setattr = base.__setattr__

                    def __setattr__(self, key, v, _base_setattr=base_setattr):
                        _base_setattr(self, key, v)
                        owner = self.__dict__.get("_write_back_parent")
                        if owner is not None and key not in ("this", "thisown"):
                            member.fset(owner, self)

                    bound[base] = type(base.__name__, (base,),
                                       {"__setattr__": __setattr__, "__doc__": base.__doc__})
                value.__class__ = bound[base]
                object.__setattr__(value, "_write_back_parent", parent)
                return value

            return property(fget, member.fset, None, member.__doc__)

        setattr(cls, name, make(member))


_write_back_members(ConsurfOptions, ("search", "clusters", "homologs", "conservation"))
%}

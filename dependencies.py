# `rmf` stays in required_modules for now, and it is worth saying why, because
# src/RmfIO.cpp no longer uses IMP.rmf at all -- it is written against RMF's
# own API.
#
# Dropping `rmf` here would take `isd` and `saxs` with it (rmf -> isd -> saxs),
# neither of which this module ever includes: 13 modules down to 10. The
# blocker is IMP's configure ordering, not this module. Declaring
# `required_dependencies = 'IMP.em:RMF'` instead works only where RMF's
# descriptor already exists when bff is configured. An *installed* IMP ships
# `share/IMP/build_info/RMF`, so the out-of-tree build the conda recipe uses
# resolves it; an in-tree build writes that descriptor while configuring the
# `rmf` module, which without the module edge now sorts *after* bff, and the
# configure dies with FileNotFoundError: './build_info/RMF' and silently
# disables IMP.bff.
#
# So this change lands with the standalone build, which resolves its own
# dependencies rather than asking IMP's generator to. Verified out-of-tree
# before it does.
#
# `isd` is a direct requirement of the NPS Bayesian structural tracer
# (`NPSIsotropicFRETEfficiencyRestraint` reads live `IMP.isd.Nuisance`
# particles). It is already in the transitive closure through `rmf`, so
# declaring it direct adds no new module to the build; it only makes the
# real edge visible and keeps the include legal if `rmf` is ever dropped.
required_modules = 'container:core:em:atom:rmf:rotamer:kinematics:isd'
required_dependencies = 'IMP.em'
optional_dependencies = 'tttrlib'

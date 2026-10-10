# The themes under src/, in dependency order (a theme may use those before it).
# Each src/<theme>/Headers.cmake names its public headers; src/imp/ is the
# connection layer to IMP and is listed in src/imp/Headers.cmake.
set(imp_bff_themes "util;graph;fit;bayesian;spectroscopy;fret;search;learn;probe;structure;sequence;smlm")

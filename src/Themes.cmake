# The themes, in dependency order (a theme may use those before it). Each has
# its sources in src/<theme>/ and its public headers in include/<theme>/ (the
# directory is the assignment); src/imp/ and include/imp/ are the connection
# layer to IMP. cmake/ThemeHeaders.cmake links the themed headers into the
# build tree, tools/dev/theme_layout.py writes ThemeSources.cpp and the
# include/<theme>.h umbrellas.
set(imp_bff_themes "util;graph;fit;bayesian;spectroscopy;fret;search;learn;probe;structure;sequence;smlm")

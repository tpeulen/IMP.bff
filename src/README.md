Place source `.cpp` files here to be compiled into the `IMP.bff` library.

Sources are laid out by theme (`src/<theme>/`, one level deep: the depth IMP's
build tooling globs), and the public headers by the same themes in
`include/<theme>/`; the directory is the assignment. `src/Themes.cmake` lists
the themes, `src/ThemeSources.cpp` is how IMP's unity build reaches the themed
files, and `tools/dev/theme_layout.py` regenerates it together with the
`include/<theme>.h` umbrellas. See `okf/themes/index.md`.

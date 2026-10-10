Place source `.cpp` files here to be compiled into the `IMP.bff` library.

Sources are laid out by theme (`src/<theme>/`, one level deep: the depth IMP's
build tooling globs). `src/Themes.cmake` lists the themes, each
`src/<theme>/Headers.cmake` assigns the flat `include/*.h` headers, and
`src/ThemeSources.cpp` is how IMP's unity build reaches the themed files
(`tools/dev/move_sources.py` regenerates it). See `okf/themes/index.md`.

# Included very early (before IMP.bff-lib exists) by the generated
# per-module ModuleBuild.cmake -- see IMP's tools/build/setup_cmake.py,
# setup_module(): any modules/bff/Setup.cmake present is included, and any
# variable this sets is visible later when modules/bff/src/CMakeLists.txt
# (ModuleLib.cmake) builds the library, because add_subdirectory() inherits
# the parent scope.

# MSVC only exports symbols an object explicitly marks dllexport; a Unix
# .so/.dylib exports everything by default. IMP is built with
# IMP_USE_SYSTEM_IHM=off, so the mmCIF/BinaryCIF parser bff calls directly
# (include/internal/Cif.h includes ihm_format.h unqualified; see the ihm
# comment in this repo's own CMakeLists.txt) is compiled as an internal,
# unexported implementation detail of imp_atom -- reachable from another
# .dylib/.so on macOS/Linux, invisible from another DLL on Windows. Linking
# imp_bff.dll then fails with "unresolved external symbol ihm_error_free"
# (and friends) even though imp_atom.dll happens to contain that code.
#
# Give bff its own private copy on Windows only: the vendored sources
# already shipped for the standalone build (standalone/thirdparty/ihm,
# byte-identical to IMP's). Neither copy exports anything, so this cannot
# collide with imp_atom's -- each DLL just gets its own internal definition.
# Scoped to MSVC so the platforms that already link cleanly are untouched.
if(MSVC)
  set(IMP_bff_LIBRARY_EXTRA_SOURCES
      ${CMAKE_SOURCE_DIR}/modules/bff/standalone/thirdparty/ihm/ihm_format.c
      ${CMAKE_SOURCE_DIR}/modules/bff/standalone/thirdparty/ihm/cmp.c)
endif()

# `%pythoncode "registry_access.py"` in pyext/include/IMP_bff.core.i (PRD-147
# A2) needs its file on swig's own search path -- ModuleSwig.cmake (IMP's
# generated per-module swig-wrapper step, checked in here as
# pyext/CMakeLists.txt) only stages *.i files into the build tree's swig/
# directory, so a same-directory .py companion is otherwise invisible to
# swig even though the .i file including it sits right next to it in the
# source tree. IMP_SWIG_PATH is that template's extension point
# (imp/tools/build/cmake_templates/ModuleSwig.cmake, line "list(APPEND
# swig_path ${IMP_SWIG_PATH})"). In the full ../imp source-tree build (as
# opposed to this repo's own standalone build), the top-level CMakeLists.txt
# add_subdirectory()s modules/bff and modules/bff/pyext as SIBLING calls, not
# nested -- so a plain `set()` here (in modules/bff's own scope, via
# ModuleBuild.cmake's include()) never reaches pyext/CMakeLists.txt's scope.
# A CACHE variable is visible everywhere regardless of that call graph.
list(APPEND IMP_SWIG_PATH ${CMAKE_SOURCE_DIR}/modules/bff/pyext/include)
list(REMOVE_DUPLICATES IMP_SWIG_PATH)
set(IMP_SWIG_PATH "${IMP_SWIG_PATH}" CACHE INTERNAL "Extra swig -I search paths" FORCE)

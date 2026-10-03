include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SwigThreads.cmake")
imp_bff_register_swig_threads()

# Are we running cmake from this directory (out of tree build) ?
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set (CMAKE_CXX_STANDARD 14)

  # ptolib requires CMake 3.16; this also enables linking the generated module
  # target from this maintained parent directory (CMP0079).
  cmake_minimum_required(VERSION 3.16)
  project(imp_module)

  # Default Python3_FIND_STRATEGY to LOCATION
  if(POLICY CMP0094)
    cmake_policy(SET CMP0094 NEW)
  endif(POLICY CMP0094)
  cmake_policy(SET CMP0058 NEW)

  # Use new-style variables/escapes
  if(POLICY CMP0053)
    cmake_policy(SET CMP0053 NEW)
  endif(POLICY CMP0053)

  # Quiet warning from cmake 3.1 or later about quoted variables in if()
  if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
  endif(POLICY CMP0054)

  # Honor link flags in try_compile() tests
  if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
  endif(POLICY CMP0056)

  SET(SWIG_EXECUTABLE swig CACHE STRING "Swig program")

  IF(APPLE)

    FIND_PACKAGE(Threads)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

    # compatability with older systems
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.13")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.13")
    SET(CMAKE_LFLAGS "${CMAKE_LFLAGS} -mmacosx-version-min=10.13")

    # explicit link to libc++ for recent osx versions
    MESSAGE(STATUS "Explicitly link to libc++ on modern osx versions")
    ADD_DEFINITIONS(-stdlib=libc++)

    # Don't set MACOSX_RPATH by default
    # https://github.com/conda/conda/issues/3624
    if(DEFINED ENV{CONDA_PREFIX})
      SET(MACOSX_RPATH OFF)
      SET(CMAKE_SKIP_RPATH TRUE)
      SET(CMAKE_SKIP_INSTALL_RPATH TRUE)
    endif()

    # Don't set MACOSX_RPATH by default
    SET(CMAKE_MACOSX_RPATH 0)
  ENDIF(APPLE)

  # IMP
  ###########################
  FIND_PACKAGE(IMP REQUIRED)
  MESSAGE ("-- IMP_USE_FILE \"${IMP_USE_FILE}\"")
  MESSAGE ("-- IMP_INCLUDE_DIR \"${IMP_INCLUDE_DIR}\"")
  INCLUDE(${IMP_USE_FILE})
  INCLUDE_DIRECTORIES(SYSTEM ${IMP_INCLUDE_DIR})
  INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/include")
  # `include/internal/Cif.h` includes "ihm_format.h" unqualified. IMP puts
  # its copy on the include path only inside its own build tree
  # (python-ihm.cmake), never in what an installed IMP exports -- even
  # conda-forge's imp, which builds with IMP_USE_SYSTEM_IHM=on against the
  # ihm package and keeps that header out of the module include path. So the
  # module build needs a copy of its own: the one vendored for the
  # standalone build, byte-identical to IMP's (standalone/thirdparty/ihm),
  # and the ihm_* symbols come from the libimp_atom we already link.
  INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/standalone/thirdparty/ihm")

  enable_testing()

  SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

  # `%pythoncode "registry_access.py"` in pyext/include/IMP_bff.core.i (PRD-147
  # A2) needs its file on swig's own search path -- only *.i files get staged
  # into the build tree's swig/ directory, so a same-directory .py companion
  # is otherwise invisible to swig even though the .i file that includes it
  # is right next to it in the source tree. IMP_SWIG_PATH is ModuleSwig.cmake's
  # extension point for exactly this (imp/tools/build/cmake_templates/ModuleSwig.cmake).
  SET(IMP_SWIG_PATH "${CMAKE_SOURCE_DIR}/pyext/include")

  imp_build_module(${CMAKE_SOURCE_DIR} bff)

else()
  INCLUDE(ModuleBuild.cmake)
endif()

# Generated IMP module files are regenerated on configure. Keep dependency
# wiring here, after they have created the module library.
if(TARGET IMP.bff-lib)
  if(NOT TARGET ptolib::ptolib)
    add_subdirectory(thirdparty/ptolib EXCLUDE_FROM_ALL)
  endif()
  # The whole archive: tttrlib, earlier on the link line, exports its own
  # ptolib (maybe another version), and a symbol found there first was bound
  # to it -- the codec table of an older ptolib, and a crash. Every ptolib
  # symbol now resolves here; inline members stay here too.
  # PRIVATE, or the WHOLE_ARCHIVE feature propagates onto every consumer's
  # link line -- on the swig pyd it lands after the imp_* import libraries,
  # whose thunks for std::vector<double> members (PtoTag::floats) are already
  # defined when whole-archive ptolib.cpp.obj is forced in: LNK2005/LNK1169
  # on Windows (CI run 37124027581). On this lib's own link the feature sits
  # before the imp_* libraries and is harmless; consumers reference no ptolib
  # symbol (public headers only forward-declare pto::StoreReader).
  # set_property, not the keyword signature: src/CMakeLists.txt's generated
  # template calls plain target_link_libraries(IMP.bff-lib ...) (and plain
  # IS the transitive form -- the propagation we are removing here); keyword
  # and plain cannot mix on one target. APPEND to LINK_LIBRARIES is the
  # keyword-PRIVATE storage.
  set_property(TARGET IMP.bff-lib APPEND PROPERTY
    LINK_LIBRARIES "$<LINK_LIBRARY:WHOLE_ARCHIVE,ptolib::ptolib>")
  set_target_properties(IMP.bff-lib PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
  # Optional: zlib reads gzip-compressed FASTA into a sequence database.
  find_package(ZLIB QUIET)
  if(ZLIB_FOUND)
    target_link_libraries(IMP.bff-lib ZLIB::ZLIB)
    target_compile_definitions(IMP.bff-lib PRIVATE IMP_BFF_HAS_ZLIB=1)
  endif()
  # Optional: libcurl asks an MSA server when no sequence database is set up.
  find_package(CURL QUIET)
  if(CURL_FOUND)
    target_link_libraries(IMP.bff-lib CURL::libcurl)
    target_compile_definitions(IMP.bff-lib PRIVATE IMP_BFF_HAS_CURL=1)
  endif()
  # Optional: Accelerate's BLAS for the protein language model's matrix
  # products on macOS (the AMX units); Eigen's own products otherwise.
  if(APPLE)
    find_library(IMPBFF_ACCELERATE Accelerate)
    if(IMPBFF_ACCELERATE)
      target_link_libraries(IMP.bff-lib ${IMPBFF_ACCELERATE})
      target_compile_definitions(IMP.bff-lib PRIVATE IMP_BFF_HAS_ACCELERATE=1)
    endif()
  endif()
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ptolib/include/ptolib"
          DESTINATION include)
  # tttrlib's headers are C++17 (std::any, <filesystem>) and
  # src/HMMSurrogate.cpp includes them, so a bff that links tttrlib compiles
  # as C++17, as the standalone build always does.
  if(TTTRLIB_LIBRARIES AND NOT "${TTTRLIB_LIBRARIES}" MATCHES "NOTFOUND")
    target_compile_features(IMP.bff-lib PRIVATE cxx_std_17)
  endif()
endif()

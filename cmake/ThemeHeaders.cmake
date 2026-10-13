# The public headers sit in include/<theme>/ (src/Themes.cmake, plus imp/ for
# the connection layer). IMP's module tooling links only include/*.h and
# include/internal/*.h into the build tree (tools/build/setup.py), so this puts
# the themed ones beside them: include/<theme>/X.h becomes
# <build>/include/IMP/bff/<theme>/X.h, reached as <IMP/bff/<theme>/X.h>. IMP's
# install step dereferences the whole <build>/include/IMP tree, so the themed
# headers install with the rest. The umbrella headers include/<theme>.h are
# flat and linked by IMP itself; IMP/bff.h reaches every header through them.
#
# Symbolic links, as IMP's own tooling makes them (copies where the platform
# refuses, as on Windows without developer mode); a link whose header is gone
# is removed. CONFIGURE_DEPENDS re-runs this when a header is added or removed.

function(imp_bff_link_theme_headers source_root dest_root)
  file(READ "${source_root}/src/Themes.cmake" _themes_cmake)
  string(REGEX MATCH "set\\(imp_bff_themes[ \t\r\n]+\"([^\"]*)\"\\)" _m "${_themes_cmake}")
  set(_themes "${CMAKE_MATCH_1}")
  list(APPEND _themes imp)
  foreach(_theme ${_themes})
    # The umbrella is IMP's to link, but on a case-insensitive file system a
    # build tree from the flat layout had FRET.h and SMLM.h there: IMP's
    # stale-link cleanup removes the old name and takes fret.h/smlm.h with it.
    set(_umbrella "${dest_root}/${_theme}.h")
    if(NOT EXISTS "${_umbrella}")
      file(CREATE_LINK "${source_root}/include/${_theme}.h" "${_umbrella}" COPY_ON_ERROR SYMBOLIC)
    endif()
    set(_dest "${dest_root}/${_theme}")
    file(MAKE_DIRECTORY "${_dest}")
    file(GLOB _headers CONFIGURE_DEPENDS "${source_root}/include/${_theme}/*.h")
    set(_wanted "")
    foreach(_h ${_headers})
      get_filename_component(_n "${_h}" NAME)
      list(APPEND _wanted "${_n}")
      set(_target "${_dest}/${_n}")
      if(IS_SYMLINK "${_target}")
        file(READ_SYMLINK "${_target}" _points_to)
        if(_points_to STREQUAL _h)
          continue()
        endif()
        file(REMOVE "${_target}")
      endif()
      file(CREATE_LINK "${_h}" "${_target}" COPY_ON_ERROR SYMBOLIC)
    endforeach()
    file(GLOB _present "${_dest}/*.h")
    foreach(_p ${_present})
      get_filename_component(_n "${_p}" NAME)
      list(FIND _wanted "${_n}" _i)
      if(_i EQUAL -1)
        file(REMOVE "${_p}")
      endif()
    endforeach()
  endforeach()
endfunction()

# ModuleSwig.cmake interpolates SWIG_EXECUTABLE and consumes the module's
# PYTHON_EXTRA_DEPENDENCIES. Set normal directory variables before that
# generated file is included; never replace IMP's global/cache executable.
get_filename_component(_IMPBFF_SWIG_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" REALPATH)

# IMP configures modules first, then adds each pyext as a sibling directory.
# There is no ModuleSwig pre-include extension point. A variable watcher runs
# in the reading directory, letting us supply its normal local variable
# without changing the parent or editing IMP's generated module template.
function(_imp_bff_swig_read variable access value current_list_file stack)
  if(NOT access STREQUAL "READ_ACCESS")
    return()
  endif()
  get_property(_entry GLOBAL PROPERTY IMPBFF_SWIG_GENERATED_ENTRY)
  get_filename_component(_current "${current_list_file}" REALPATH)
  if(NOT _current STREQUAL _entry)
    return()
  endif()
  get_property(_IMPBFF_SWIG_CMAKE_DIR GLOBAL PROPERTY IMPBFF_SWIG_CMAKE_DIR)
  set(SWIG_EXECUTABLE "${value}")
  imp_bff_enable_swig_threads()
  set(SWIG_EXECUTABLE "${SWIG_EXECUTABLE}" PARENT_SCOPE)
  set(_IMPBFF_REAL_SWIG_EXECUTABLE "${_IMPBFF_REAL_SWIG_EXECUTABLE}" PARENT_SCOPE)
  set(IMP_bff_PYTHON_EXTRA_DEPENDENCIES
    "${IMP_bff_PYTHON_EXTRA_DEPENDENCIES}" PARENT_SCOPE)
endfunction()

function(imp_bff_register_swig_threads)
  get_filename_component(_entry
    "${CMAKE_CURRENT_SOURCE_DIR}/pyext/CMakeLists.txt" REALPATH)
  set_property(GLOBAL PROPERTY IMPBFF_SWIG_GENERATED_ENTRY "${_entry}")
  set_property(GLOBAL PROPERTY IMPBFF_SWIG_CMAKE_DIR "${_IMPBFF_SWIG_CMAKE_DIR}")
  variable_watch(SWIG_EXECUTABLE _imp_bff_swig_read)
endfunction()

function(imp_bff_enable_swig_threads)
  if(NOT SWIG_EXECUTABLE OR NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "BFF SWIG thread support requires configured SWIG and Python executables")
  endif()
  if(_IMPBFF_REAL_SWIG_EXECUTABLE)
    set(_real "${_IMPBFF_REAL_SWIG_EXECUTABLE}")
  elseif(IS_ABSOLUTE "${SWIG_EXECUTABLE}")
    set(_real "${SWIG_EXECUTABLE}")
  else()
    # Resolve the configured command once, without substituting another SWIG
    # name/version. Absolute configured paths are passed through unchanged.
    find_program(_IMPBFF_CONFIGURED_SWIG NAMES "${SWIG_EXECUTABLE}")
    set(_real "${_IMPBFF_CONFIGURED_SWIG}")
    if(NOT _real)
      message(FATAL_ERROR "Cannot resolve configured SWIG: ${SWIG_EXECUTABLE}")
    endif()
    unset(_IMPBFF_CONFIGURED_SWIG CACHE)
  endif()
  set(_helper "${_IMPBFF_SWIG_CMAKE_DIR}/../tools/swig_threads.py")
  get_filename_component(_helper "${_helper}" REALPATH)
  set(_launcher_dir "${CMAKE_CURRENT_BINARY_DIR}/swig-launcher")
  file(MAKE_DIRECTORY "${_launcher_dir}")
  if(WIN32)
    set(_launcher "${_launcher_dir}/swig-threads.cmd")
    set(_template "${_IMPBFF_SWIG_CMAKE_DIR}/swig_threads.cmd.in")
    # Batch launchers use double quotes and disable delayed expansion.
    set(_python "${PYTHON_EXECUTABLE}")
    set(_swig "${_real}")
    string(REPLACE "%" "%%" _python "${_python}")
    string(REPLACE "%" "%%" _swig "${_swig}")
    string(REPLACE "%" "%%" _helper_argument "${_helper}")
  else()
    set(_launcher "${_launcher_dir}/swig-threads")
    set(_template "${_IMPBFF_SWIG_CMAKE_DIR}/swig_threads.sh.in")
    # Quote literal paths for sh, including spaces and single quotes.
    string(REPLACE "'" "'\"'\"'" _python "${PYTHON_EXECUTABLE}")
    string(REPLACE "'" "'\"'\"'" _swig "${_real}")
    string(REPLACE "'" "'\"'\"'" _helper_argument "${_helper}")
  endif()
  # configure_file preserves the executable permissions of the POSIX source
  # template (also on the minimum supported CMake 3.16).
  configure_file("${_template}" "${_launcher}" @ONLY)
  set(_IMPBFF_REAL_SWIG_EXECUTABLE "${_real}" PARENT_SCOPE)
  set(SWIG_EXECUTABLE "${_launcher}" PARENT_SCOPE)
  list(APPEND IMP_bff_PYTHON_EXTRA_DEPENDENCIES
    "${_launcher}" "${_helper}" "${_template}"
    "${_IMPBFF_SWIG_CMAKE_DIR}/SwigThreads.cmake")
  list(REMOVE_DUPLICATES IMP_bff_PYTHON_EXTRA_DEPENDENCIES)
  set(IMP_bff_PYTHON_EXTRA_DEPENDENCIES
    "${IMP_bff_PYTHON_EXTRA_DEPENDENCIES}" PARENT_SCOPE)
endfunction()

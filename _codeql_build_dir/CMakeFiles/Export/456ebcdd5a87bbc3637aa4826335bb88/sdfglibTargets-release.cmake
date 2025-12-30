#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sdfglib::daisy_rtl" for configuration "Release"
set_property(TARGET sdfglib::daisy_rtl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sdfglib::daisy_rtl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdaisy_rtl.a"
  )

list(APPEND _cmake_import_check_targets sdfglib::daisy_rtl )
list(APPEND _cmake_import_check_files_for_sdfglib::daisy_rtl "${_IMPORT_PREFIX}/lib/libdaisy_rtl.a" )

# Import target "sdfglib::arg_capture_io" for configuration "Release"
set_property(TARGET sdfglib::arg_capture_io APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sdfglib::arg_capture_io PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libarg_capture_io.a"
  )

list(APPEND _cmake_import_check_targets sdfglib::arg_capture_io )
list(APPEND _cmake_import_check_files_for_sdfglib::arg_capture_io "${_IMPORT_PREFIX}/lib/libarg_capture_io.a" )

# Import target "sdfglib::sdfglib" for configuration "Release"
set_property(TARGET sdfglib::sdfglib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sdfglib::sdfglib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsdfglib.a"
  )

list(APPEND _cmake_import_check_targets sdfglib::sdfglib )
list(APPEND _cmake_import_check_files_for_sdfglib::sdfglib "${_IMPORT_PREFIX}/lib/libsdfglib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

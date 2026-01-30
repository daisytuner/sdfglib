# Find SDFG
if(NOT TARGET sdfg::sdfg)
    find_package(sdfg CONFIG REQUIRED)
endif()

# Find SDGOPT
if(NOT TARGET sdfg::opt)
    find_package(sdfgopt CONFIG REQUIRED)
endif()

# Source files
set(SOURCE_FILES
    src/printf_target.cpp
    src/printf_data_offloading_node.cpp
    src/printf_map_dispatcher.cpp
    src/printf_transform.cpp
)

add_library(sdfgtutorial_printf_target
    ${SOURCE_FILES}
)
add_library(sdfg::tutorial_printf_target ALIAS sdfgtutorial_printf_target)
add_library(sdfgtutorial_printf_target::sdfgtutorial_printf_target ALIAS sdfgtutorial_printf_target)
target_include_directories(sdfgtutorial_printf_target
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfgtutorial_printf_target PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfgtutorial_printf_target PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgtutorial_printf_target PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfgtutorial_printf_target PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(sdfgtutorial_printf_target PUBLIC sdfg::sdfg sdfg::opt)
set_target_properties(sdfgtutorial_printf_target PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfgtutorial_printf_target
    EXPORT sdfgtutorial_printf_targetTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfgtutorial_printf_targetTargets
    NAMESPACE sdfg::
    FILE sdfgtutorial_printf_targetTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtutorial_printf_targetConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfgtutorial_printf_targetConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtutorial_printf_targetConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtutorial_printf_targetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtutorial_printf_targetConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

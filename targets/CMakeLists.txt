# Find SDFG
if(NOT TARGET sdfglib::sdfglib)
    find_package(sdfglib CONFIG REQUIRED)
endif()

# Find ISL
find_package(PkgConfig REQUIRED)

set(SOURCE_FILES
    src/cuda/codegen/cuda_map_dispatcher.cpp
    src/cuda/nodes/cuda_offloading_nodes.cpp
    src/cuda/passes/sync_condition_propagation.cpp
    src/cuda/transformations/cublas_offloading_extension.cpp
    src/cuda/transformations/cuda_parallelize_nested_map.cpp
    src/cuda/transformations/cuda_parallelize_nested_map.cpp
    src/cuda/transformations/cuda_transform.cpp
    src/cuda/transformations/gpu_condition_propagation.cpp
    src/cuda/transformations/gpu_loop_reordering.cpp
    src/cuda/transformations/gpu_tiling.cpp
    src/cuda/transformations/kernel_local_storage.cpp
    src/cuda/cuda.cpp
    src/passes/code_motion/extended_block_hoisting.cpp
    src/passes/code_motion/extended_block_sorting.cpp
    src/passes/data_transfer_minimization_pass.cpp
    src/passes/readonly_transfer_hoisting_pass.cpp
    src/passes/remove_redundant_transfers_pass.cpp
    src/transformations/offload_transform.cpp
    src/data_offloading_node.cpp
    src/utils.cpp
)
add_library(sdfgtargets
    ${SOURCE_FILES}
)
add_library(sdfgtargets::sdfgtargets ALIAS sdfgtargets)
target_include_directories(sdfglib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfgtargets PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfgtargets PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgtargets PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfgtargets PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(sdfgtargets PUBLIC sdfglib PkgConfig::ISL)
set_target_properties(sdfgtargets PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfgtargets
    EXPORT sdfgtargetsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfgtargetsTargets
    NAMESPACE sdfgtargets::
    FILE sdfgtargetsTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgtargets
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtargetsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfgtargetsConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtargetsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgtargets
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtargetsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgtargetsConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgtargets
)

option(SDFG_BUILD_TESTS "Build tests" ON)
if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

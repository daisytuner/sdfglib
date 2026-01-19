# Find SDFG
if(NOT TARGET sdfglib::sdfglib)
    find_package(sdfglib CONFIG REQUIRED)
endif()

# Find ISL
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)

set(SOURCE_FILES
    src/analysis/scop_analysis.cpp
    src/cuda/cuda.cpp
    src/cuda/codegen/cuda_map_dispatcher.cpp
    src/cuda/nodes/cuda_data_offloading_node.cpp
    src/cuda/passes/sync_condition_propagation.cpp
    src/cuda/transformations/cublas_offloading_expansion.cpp
    src/cuda/transformations/cuda_parallelize_nested_map.cpp
    src/cuda/transformations/cuda_transform.cpp
    src/cuda/transformations/gpu_condition_propagation.cpp
    src/cuda/transformations/gpu_loop_reordering.cpp
    src/cuda/transformations/gpu_tiling.cpp
    src/cuda/transformations/kernel_local_storage.cpp
    src/offloading/data_offloading_node.cpp
    src/offloading/utils.cpp
    src/passes/normalization/perfect_loop_distribution.cpp
    src/passes/normalization/stride_minimization.cpp
    src/passes/scheduler/highway_scheduler.cpp
    src/passes/scheduler/loop_scheduler.cpp
    src/passes/scheduler/omp_scheduler.cpp
    src/passes/scheduler/polly_scheduler.cpp
    src/targets/highway/codegen/highway_map_dispatcher.cpp
    src/targets/omp/codegen/omp_map_dispatcher.cpp
    src/transformations/highway_transform.cpp
    src/transformations/omp_transform.cpp
    src/transformations/polly_transform.cpp
)
add_library(sdfgopt
    ${SOURCE_FILES}
)
add_library(sdfgopt::sdfgopt ALIAS sdfgopt)
target_include_directories(sdfglib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfgopt PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfgopt PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgopt PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfgopt PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(sdfgopt PUBLIC sdfglib PkgConfig::ISL)
set_target_properties(sdfgopt PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfgopt
    EXPORT sdfgoptTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfgoptTargets
    NAMESPACE sdfgopt::
    FILE sdfgoptTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgopt
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfgoptConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgopt
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfgopt
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

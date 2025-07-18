cmake_minimum_required(VERSION 3.16)

project(sdfglib VERSION 0.0.1 DESCRIPTION "A library for generating structured dataflow graphs (SDFG)")
include(GNUInstallDirs)

option(SDFGLIB_BUILD_TESTS
       "Build tests" ON)

add_subdirectory(3rdParty/symengine)
add_subdirectory(rtl)
add_subdirectory(arg-capture-io)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SDFGLIB_ENABLE_SANITIZER
       "Enable Address/Memory/Undefined/Leak/Thread sanitizers" OFF)

set(SDFGLIB_SANITIZER
    "address"
    CACHE STRING
    "List of sanitizers to enable; supported: \
     address, memory, undefined, leak, thread")

if(SDFGLIB_ENABLE_SANITIZER)
  message(STATUS "Enabling sanitizer: ${SDFGLIB_SANITIZER}")
  set(_SAN_COMPILE_OPTS
    -fno-omit-frame-pointer
    -fsanitize=${SDFGLIB_SANITIZER}
  )
  set(_SAN_LINK_OPTS
    -fsanitize=${SDFGLIB_SANITIZER}
  )
endif()

# NLohmann json
find_package(nlohmann_json REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS graph)

# Find the ISL library
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)

# Coverage option
option(SDFGLIB_ENABLE_COVERAGE "Enable coverage flags" OFF)

set(SOURCE_FILES
    src/analysis/analysis.cpp
    src/analysis/assumptions_analysis.cpp
    src/analysis/data_parallelism_analysis.cpp
    src/analysis/data_dependency_analysis.cpp
    src/analysis/loop_analysis.cpp
    src/analysis/mem_access_range_analysis.cpp
    src/analysis/scope_analysis.cpp
    src/analysis/users.cpp
    src/builder/function_builder.cpp
    src/builder/sdfg_builder.cpp
    src/builder/structured_sdfg_builder.cpp
    src/codegen/utils.cpp
    src/codegen/code_generator.cpp
    src/codegen/code_snippet_factory.cpp
    src/codegen/instrumentation/capture_var_plan.cpp
    src/codegen/language_extensions/c_language_extension.cpp
    src/codegen/language_extensions/cpp_language_extension.cpp
    src/codegen/language_extensions/cuda_language_extension.cpp
    src/codegen/code_generators/c_code_generator.cpp
    src/codegen/code_generators/c_style_base_code_generator.cpp
    src/codegen/code_generators/cpp_code_generator.cpp
    src/codegen/code_generators/cuda_code_generator.cpp
    src/codegen/dispatchers/node_dispatcher.cpp
    src/codegen/dispatchers/node_dispatcher_registry.cpp
    src/codegen/dispatchers/block_dispatcher.cpp
    src/codegen/dispatchers/if_else_dispatcher.cpp
    src/codegen/dispatchers/sequence_dispatcher.cpp
    src/codegen/dispatchers/for_dispatcher.cpp
    src/codegen/dispatchers/map_dispatcher.cpp
    src/codegen/dispatchers/while_dispatcher.cpp
    src/codegen/instrumentation/outermost_loops_instrumentation.cpp
    src/control_flow/interstate_edge.cpp
    src/control_flow/state.cpp
    src/data_flow/access_node.cpp
    src/data_flow/code_node.cpp
    src/data_flow/data_flow_graph.cpp
    src/data_flow/data_flow_node.cpp
    src/data_flow/library_node.cpp
    src/data_flow/memlet.cpp
    src/data_flow/tasklet.cpp
    src/data_flow/library_nodes/barrier_local_node.cpp
    src/data_flow/library_nodes/metadata_node.cpp
    src/data_flow/library_nodes/math/math_node.cpp
    src/data_flow/library_nodes/math/ml/relu.cpp
    src/deepcopy/structured_sdfg_deep_copy.cpp
    src/graph/graph.cpp
    src/visitor/structured_sdfg_visitor.cpp
    src/passes/pass.cpp
    src/passes/pipeline.cpp
    src/passes/canonicalize_cfg.cpp
    src/passes/dataflow/dead_data_elimination.cpp
    src/passes/dataflow/dead_reference_elimination.cpp
    src/passes/dataflow/constant_elimination.cpp
    src/passes/dataflow/memlet_propagation.cpp
    src/passes/dataflow/redundant_array_elimination.cpp
    src/passes/dataflow/trivial_array_elimination.cpp
    src/passes/dataflow/view_propagation.cpp
    src/passes/schedules/parallelization_pass.cpp
    src/passes/symbolic/symbol_evolution.cpp
    src/passes/symbolic/symbol_propagation.cpp
    src/passes/symbolic/symbol_promotion.cpp
    src/passes/symbolic/condition_propagation.cpp
    src/passes/structured_control_flow/block_fusion.cpp
    src/passes/structured_control_flow/common_assignment_elimination.cpp
    src/passes/structured_control_flow/dead_cfg_elimination.cpp
    src/passes/structured_control_flow/for2map.cpp
    src/passes/structured_control_flow/loop_normalization.cpp
    src/passes/structured_control_flow/sequence_fusion.cpp
    src/passes/structured_control_flow/while_to_for_conversion.cpp
    src/serializer/json_serializer.cpp
    src/structured_control_flow/block.cpp
    src/structured_control_flow/control_flow_node.cpp
    src/structured_control_flow/for.cpp
    src/structured_control_flow/if_else.cpp
    src/structured_control_flow/map.cpp
    src/structured_control_flow/return.cpp
    src/structured_control_flow/sequence.cpp
    src/structured_control_flow/structured_loop.cpp
    src/structured_control_flow/while.cpp
    src/symbolic/assumptions.cpp
    src/symbolic/conjunctive_normal_form.cpp
    src/symbolic/extreme_values.cpp
    src/symbolic/maps.cpp
    src/symbolic/polynomials.cpp
    src/symbolic/series.cpp
    src/symbolic/sets.cpp
    src/symbolic/symbolic.cpp
    src/symbolic/utils.cpp
    src/transformations/loop_distribute.cpp
    src/transformations/loop_interchange.cpp
    src/transformations/loop_slicing.cpp
    src/transformations/loop_tiling.cpp
    src/transformations/out_local_storage.cpp
    src/transformations/parallelization.cpp
    src/transformations/recorder.cpp
    src/transformations/utils.cpp
    src/types/array.cpp
    src/types/function.cpp
    src/types/pointer.cpp
    src/types/scalar.cpp
    src/types/structure.cpp
    src/types/utils.cpp
    src/visualizer/dot_visualizer.cpp
    src/visualizer/visualizer.cpp
    src/element.cpp
    src/function.cpp
    src/sdfg.cpp
    src/structured_sdfg.cpp
)

# Original Library
add_library(sdfglib
    ${SOURCE_FILES}
)
add_library(sdfglib::sdfglib ALIAS sdfglib)
target_include_directories(sdfglib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine/symengine/utilities/cereal/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine/symengine/utilities/cereal/include>
)

target_compile_options(sdfglib PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-private-field -Wno-switch -Wno-deprecated-declarations)

if(SDFGLIB_ENABLE_SANITIZER)
  target_compile_options(sdfglib PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfglib PUBLIC ${_SAN_LINK_OPTS})
endif()

# Add coverage flags if enabled
if(SDFGLIB_ENABLE_COVERAGE)
    target_compile_options(sdfglib PUBLIC --coverage)
    target_link_options(sdfglib PUBLIC --coverage)
endif()

target_link_libraries(sdfglib PUBLIC nlohmann_json::nlohmann_json ${Boost_LIBRARIES} PkgConfig::ISL symengine)
set_target_properties(sdfglib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfglib
    EXPORT sdfglibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES
    include/export/daisyrtl.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfglibTargets
    NAMESPACE sdfglib::
    FILE sdfglibTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfglibConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

if(SDFGLIB_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

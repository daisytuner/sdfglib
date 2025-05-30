cmake_minimum_required(VERSION 3.16)

project(sdfglib VERSION 0.0.1 DESCRIPTION "A library for generating structured dataflow graphs (SDFG)")
include(GNUInstallDirs)

add_subdirectory(3rdParty/symengine)
add_subdirectory(rtl)

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
    src/analysis/happens_before_analysis.cpp
    src/analysis/loop_tree_analysis.cpp
    src/analysis/scope_tree_analysis.cpp
    src/analysis/users.cpp
    src/builder/function_builder.cpp
    src/builder/sdfg_builder.cpp
    src/builder/structured_sdfg_builder.cpp
    src/codegen/utils.cpp
    src/codegen/language_extensions/c_language_extension.cpp
    src/codegen/language_extensions/cpp_language_extension.cpp
    src/codegen/language_extensions/cuda_language_extension.cpp
    src/codegen/code_generators/c_code_generator.cpp
    src/codegen/code_generators/cpp_code_generator.cpp
    src/codegen/code_generators/cuda_code_generator.cpp
    src/codegen/dispatchers/node_dispatcher.cpp
    src/codegen/dispatchers/block_dispatcher.cpp
    src/codegen/dispatchers/if_else_dispatcher.cpp
    src/codegen/dispatchers/sequence_dispatcher.cpp
    src/codegen/dispatchers/for_dispatcher.cpp
    src/codegen/dispatchers/map_dispatcher.cpp
    src/codegen/dispatchers/while_dispatcher.cpp
    src/codegen/dispatchers/kernel_dispatcher.cpp
    src/codegen/dispatchers/schedules/highway_dispatcher.cpp
    src/codegen/dispatchers/schedules/openmp_dispatcher.cpp
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
    src/deepcopy/structured_sdfg_deep_copy.cpp
    src/graph/graph.cpp
    src/visitor/structured_sdfg_visitor.cpp
    src/passes/pass.cpp
    src/passes/pipeline.cpp
    src/passes/canonicalize_cfg.cpp
    src/passes/dataflow/dead_data_elimination.cpp
    src/passes/dataflow/dead_reference_elimination.cpp
    src/passes/dataflow/memlet_propagation.cpp
    src/passes/dataflow/redundant_array_elimination.cpp
    src/passes/dataflow/trivial_array_elimination.cpp
    src/passes/dataflow/view_propagation.cpp
    src/passes/schedule/allocation_inference.cpp
    src/passes/symbolic/symbol_propagation.cpp
    src/passes/symbolic/symbol_promotion.cpp
    src/passes/symbolic/condition_propagation.cpp
    src/passes/structured_control_flow/block_fusion.cpp
    src/passes/structured_control_flow/common_assignment_elimination.cpp
    src/passes/structured_control_flow/dead_cfg_elimination.cpp
    src/passes/structured_control_flow/for2map.cpp
    src/passes/structured_control_flow/loop_bound_normalization.cpp
    src/passes/structured_control_flow/loop_dependent_symbol_elimination.cpp
    src/passes/structured_control_flow/sequence_fusion.cpp
    src/passes/structured_control_flow/while_to_for_conversion.cpp
    src/serializer/json_serializer.cpp
    src/structured_control_flow/block.cpp
    src/structured_control_flow/control_flow_node.cpp
    src/structured_control_flow/for.cpp
    src/structured_control_flow/if_else.cpp
    src/structured_control_flow/kernel.cpp
    src/structured_control_flow/map.cpp
    src/structured_control_flow/return.cpp
    src/structured_control_flow/sequence.cpp
    src/structured_control_flow/while.cpp
    src/symbolic/analysis.cpp
    src/symbolic/assumptions.cpp
    src/symbolic/symbolic.cpp
    src/transformations/kernel_local_storage.cpp
    src/transformations/loop_distribute.cpp
    src/transformations/loop_interchange.cpp
    src/transformations/loop_slicing.cpp
    src/transformations/loop_tiling.cpp
    src/transformations/loop_to_kernel_dim.cpp
    src/transformations/loop_unroll.cpp
    src/transformations/out_local_storage.cpp
    src/transformations/transformation.cpp
    src/transformations/utils.cpp
    src/transformations/vectorization.cpp
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
    src/schedule.cpp
    src/conditional_schedule.cpp
)

# Original Library
add_library(sdfglib
    ${SOURCE_FILES}
)
target_include_directories(sdfglib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(sdfglib
    SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine
        ${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/symengine/symengine/utilities/cereal/include
        ${CMAKE_CURRENT_BINARY_DIR}/3rdParty/symengine/symengine/utilities/cereal/include
)
target_compile_options(sdfglib PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-private-field -Wno-switch -Wno-deprecated-declarations)

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

install(TARGETS sdfglib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES
    include/export/daisyrtl.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_subdirectory(tests)

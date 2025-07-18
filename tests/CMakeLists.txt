include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    analysis/assumptions_analysis_test.cpp
    analysis/data_parallelism_analysis_test.cpp
    analysis/data_dependency_analysis_test.cpp
    analysis/loop_analysis_test.cpp
    analysis/mem_access_range_analysis_test.cpp
    analysis/users_test.cpp
    builder/function_builder_test.cpp
    builder/sdfg_builder_test.cpp
    builder/structured_sdfg_builder_test.cpp
    codegen/code_generators/c_code_generator_test.cpp
    codegen/code_generators/cpp_code_generator_test.cpp
    codegen/code_generators/cuda_code_generator_test.cpp
    codegen/dispatchers/block_dispatcher_test.cpp
    codegen/dispatchers/for_dispatcher_test.cpp
    codegen/dispatchers/if_else_dispatcher_test.cpp
    codegen/dispatchers/map_dispatcher_test.cpp
    codegen/dispatchers/node_dispatcher_factory_test.cpp
    codegen/dispatchers/node_dispatcher_test.cpp
    codegen/dispatchers/sequence_dispatcher_test.cpp
    codegen/dispatchers/while_dispatcher_test.cpp
    codegen/language_extensions/c_language_extension_test.cpp
    codegen/language_extensions/cpp_language_extension_test.cpp
    codegen/language_extensions/cuda_language_extension_test.cpp
    codegen/utils_test.cpp
    data_flow/data_flow_graph_test.cpp
    data_flow/library_nodes/math_test.cpp
    deepcopy/structured_sdfg_deep_copy_test.cpp
    element_test.cpp
    graph/graph_algorithms_test.cpp
    helpers/helpers_test.cpp
    passes/dataflow/dead_data_elimination_test.cpp
    passes/dataflow/redundant_array_elimination_test.cpp
    passes/dataflow/trivial_array_elimination_test.cpp
    passes/dataflow/view_propagation_test.cpp
    passes/pipeline_test.cpp
    passes/schedules/parallelization_pass_test.cpp
    passes/structured_control_flow/block_fusion_test.cpp
    passes/structured_control_flow/for2map_test.cpp
    passes/structured_control_flow/loop_normalization_test.cpp
    passes/symbolic/symbol_promotion_test.cpp
    passes/symbolic/symbol_propagation_test.cpp
    replace/symbol_replace_test.cpp
    sdfg_test.cpp
    serializer/json_serializer_test.cpp
    structured_sdfg_test.cpp
    symbolic/assumptions_test.cpp
    symbolic/conjunctive_normal_form_test.cpp
    symbolic/extreme_values_test.cpp
    symbolic/polynomials_test.cpp
    symbolic/symbolic_test.cpp
    symbolic/sets_test.cpp
    transformations/loop_interchange_test.cpp
    transformations/loop_slicing_test.cpp
    transformations/loop_tiling_test.cpp
    transformations/out_local_storage_test.cpp
    transformations/parallelization_test.cpp
    transformations/recorder_test.cpp
    types/array_test.cpp
    types/function_test.cpp
    types/pointer_test.cpp
    types/primitive_type_test.cpp
    types/reference_test.cpp
    types/scalar_test.cpp
    types/sizeof_test.cpp
    types/structure_test.cpp
    types/utils_test.cpp
    visitor/structured_sdfg_visitor_test.cpp
    visualizer/dot_visualizer_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfglib_test ${TEST_FILES})
target_include_directories(sdfglib_test PRIVATE ./)
target_link_libraries(sdfglib_test gtest_main sdfglib)

add_test(NAME sdfglib_test_test COMMAND sdfglib_test)

include(GoogleTest)
gtest_discover_tests(sdfglib_test)

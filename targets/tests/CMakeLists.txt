include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    cuda/cuda_kernel_test.cpp
    cuda/cuda_library_nodes_test.cpp
    passes/code_motion/extended_block_hoisting_test.cpp
    passes/data_transfer_minimization_pass_test.cpp
    passes/readonly_transfer_hoisting_pass_test.cpp
    passes/remove_redundant_data_transfers_pass_test.cpp
    transformations/cuda_parallelize_nested_map_test.cpp
    transformations/gpu_tiling_test.cpp
    transformations/kernel_local_storage_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfgopt_test ${TEST_FILES})
target_include_directories(sdfgopt_test PRIVATE ./)
target_link_libraries(sdfgopt_test gtest_main sdfgopt)
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgopt_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(sdfgopt_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME sdfgopt_test_test COMMAND sdfgopt_test)

include(GoogleTest)
gtest_discover_tests(sdfgopt_test)

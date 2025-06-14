include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    analysis/loop_dependency_analysis_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfglib_test ${TEST_FILES})
target_include_directories(sdfglib_test PRIVATE ./)
target_link_libraries(sdfglib_test gtest_main sdfglib)

add_test(NAME sdfglib_test_test COMMAND sdfglib_test)

include(GoogleTest)
gtest_discover_tests(sdfglib_test)

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(arg_capture_io STATIC
    src/arg_capture_io.cpp
    src/base64.cpp
)

add_library(sdfglib::arg_capture_io ALIAS arg_capture_io)

target_include_directories(arg_capture_io PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${PARENT_PROJECT_INSTALL_PREFIX}include>
)

# Coverage option
option(ARG_CAPTURE_IO_ENABLE_COVERAGE "Enable coverage flags" OFF)

# Add coverage flags if enabled
if(ARG_CAPTURE_IO_ENABLE_COVERAGE)
    target_compile_options(arg_capture_io PUBLIC --coverage)
    target_link_options(arg_capture_io PUBLIC --coverage)
endif()

find_package(nlohmann_json REQUIRED)

target_link_libraries(arg_capture_io PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(arg_capture_io PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-private-field -Wno-switch -Wno-deprecated-declarations
)

set_target_properties(arg_capture_io PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS arg_capture_io
    EXPORT sdfglibTargets
    LIBRARY DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT runtime
    ARCHIVE DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}lib COMPONENT runtime
    PUBLIC_HEADER DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT runtime
    INCLUDES DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include
)

install(DIRECTORY include/daisy_rtl DESTINATION ${PARENT_PROJECT_INSTALL_PREFIX}include COMPONENT runtime)

if(SDFGLIB_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()

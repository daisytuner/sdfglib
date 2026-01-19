cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)
set(SDFGLIB_VERSION "${PROJECT_VERSION}")

project(sdfglib VERSION ${SDFGLIB_VERSION} DESCRIPTION "Core SDFG library for generating and manipulating Stateful DataFlow Graphs" LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tests Options
option(SDFG_BUILD_TESTS "Build tests" ON)

# Sanitizer Options
option(SDFG_ENABLE_SANITIZER "Enable Address/Memory/Undefined/Leak/Thread sanitizers" OFF)
set(SDFG_SANITIZER "address" CACHE STRING
    "List of sanitizers to enable; supported: \
     address, memory, undefined, leak, thread")
if(SDFG_ENABLE_SANITIZER)
  message(STATUS "Enabling sanitizer: ${SDFG_SANITIZER}")
  set(_SAN_COMPILE_OPTS
    -fno-omit-frame-pointer
    -fsanitize=${SDFG_SANITIZER}
  )
  set(_SAN_LINK_OPTS
    -fsanitize=${SDFG_SANITIZER}
  )
endif()

# Coverage option
option(SDFG_ENABLE_COVERAGE "Enable coverage flags" OFF)

add_subdirectory(sdfg)
add_subdirectory(examples/applications)
add_subdirectory(opt)
add_subdirectory(rtl)
add_subdirectory(arg-capture-io)

option(PYTHON_BUILD_FRONTEND "Build Python frontend" OFF)
if(PYTHON_BUILD_FRONTEND)
    add_subdirectory(python)
endif()

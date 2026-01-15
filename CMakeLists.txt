cmake_minimum_required(VERSION 3.16)

project(sdfglib VERSION 0.0.1 DESCRIPTION "Core SDFG library for generating and manipulating Stateful DataFlow Graphs" LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(sdfg)
add_subdirectory(rtl)
add_subdirectory(arg-capture-io)

# To use SymEngine from another CMake project include the following in your
# `CMakeLists.txt` file

#    `find_package(SymEngine CONFIG)`

# You can give the path to the SymEngine installation directory if it was
# installed to a non standard location by,

#    `find_package(SymEngine CONFIG Paths /path/to/install/dir)`

# Alternatively, you can give the path to the build directory.

# Variable exported are
# SYMENGINE_BUILD_TYPE         - Cofiguration Type Debug or Release
# SYMENGINE_INCLUDE_DIRS       - Header file directories
# SYMENGINE_LIBRARIES          - SymEngine libraries and dependency libraries to link against
# SYMENGINE_FOUND              - Set to yes
# SYMENGINE_CXX_FLAGS_RELEASE  - C++ flags for Release configuration
# SYMENGINE_CXX_FLAGS_DEBUG    - C++ flags for Debug configuration
# SYMENGINE_C_FLAGS_RELEASE    - C flags for Release configuration
# SYMENGINE_C_FLAGS_DEBUG      - C flags for Debug configuration

# An example project would be,
#
# cmake_minimum_required(VERSION 2.8)
# find_package(symengine CONFIG)
# set(CMAKE_CXX_FLAGS_RELEASE ${SYMENGINE_CXX_FLAGS_RELEASE})
#
# include_directories(${SYMENGINE_INCLUDE_DIRS})
# add_executable(example main.cpp)
# target_link_libraries(example ${SYMENGINE_LIBRARIES})
#

cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW) # needed for llvm >= 16
endif ()

include(CMakeFindDependencyMacro)

set(SYMENGINE_CXX_FLAGS "-std=c++11  -fPIC")
set(SYMENGINE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-unknown-pragmas")
set(SYMENGINE_CXX_FLAGS_DEBUG "-g -Wno-unknown-pragmas")
set(SYMENGINE_C_FLAGS "")
set(SYMENGINE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(SYMENGINE_C_FLAGS_DEBUG "-g")

# ... for the build tree
get_filename_component(SYMENGINE_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(SYMENGINE_BUILD_TREE yes)

if(NOT TARGET symengine)
    include("${SYMENGINE_CMAKE_DIR}/SymEngineTargets.cmake")
endif()
set_target_properties(symengine PROPERTIES INTERFACE_LINK_LIBRARIES "")

if(SYMENGINE_BUILD_TREE)
    set(SYMENGINE_INSTALL_CMAKE_DIR "${SYMENGINE_CMAKE_DIR}")
    set(SYMENGINE_INCLUDE_DIRS /home/runner/work/sdfglib/sdfglib/3rdParty/symengine;/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine;/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include ${SYMENGINE_CMAKE_DIR})
    if (TARGET teuchos)
        set(SYMENGINE_INCLUDE_DIRS ${SYMENGINE_INCLUDE_DIRS} ${SYMENGINE_CMAKE_DIR}/symengine/utilities/teuchos)
    endif()
else()
    set(SYMENGINE_INSTALL_CMAKE_DIR "/usr/local/lib/cmake/symengine")
    set(SYMENGINE_INCLUDE_DIRS "${SYMENGINE_CMAKE_DIR}/../../../include")
    if (NOT no)
        set(SYMENGINE_INCLUDE_DIRS ${SYMENGINE_INCLUDE_DIRS}
            "${SYMENGINE_CMAKE_DIR}/../../../include/symengine/utilities/cereal/include")
    endif()
endif()



set(SYMENGINE_GMP_LIBRARIES /usr/lib/x86_64-linux-gnu/libgmp.so)
set(SYMENGINE_GMP_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(HAVE_SYMENGINE_GMP True)

set(SYMENGINE_LLVM_COMPONENTS )

if (NOT "${SYMENGINE_LLVM_COMPONENTS}" STREQUAL "")
    find_package(LLVM REQUIRED ${SYMENGINE_LLVM_COMPONENTS} HINTS )
    llvm_map_components_to_libnames(llvm_libs_direct ${SYMENGINE_LLVM_COMPONENTS})
    llvm_expand_dependencies(llvm_libs ${llvm_libs_direct})
    set(SYMENGINE_LIBRARIES ${SYMENGINE_LIBRARIES} ${llvm_libs})
else()
    set(SYMENGINE_LLVM_INCLUDE_DIRS)
endif()

if (TARGET gmp)
    # Avoid defining targets again
    set(SYMENGINE_SKIP_DEPENDENCIES yes CACHE BOOL "Skip finding dependencies")
else()
    set(SYMENGINE_SKIP_DEPENDENCIES no CACHE BOOL "Skip finding dependencies")
endif()

foreach(PKG GMP)
    set(SYMENGINE_INCLUDE_DIRS ${SYMENGINE_INCLUDE_DIRS} ${SYMENGINE_${PKG}_INCLUDE_DIRS})
    set(SYMENGINE_LIBRARIES ${SYMENGINE_LIBRARIES} ${SYMENGINE_${PKG}_LIBRARIES})
endforeach()

#Use CMake provided find_package(BOOST) module
if (NOT "" STREQUAL "")
    find_dependency(Boost REQUIRED COMPONENTS )
    set(SYMENGINE_INCLUDE_DIRS ${SYMENGINE_INCLUDE_DIRS} )
    set(SYMENGINE_LIBRARIES ${SYMENGINE_LIBRARIES} )
endif()

list(REMOVE_DUPLICATES SYMENGINE_INCLUDE_DIRS)

foreach(LIB "symengine")
    # Remove linking of dependencies to later add them as targets
    set_target_properties(${LIB} PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "")
    set_target_properties(${LIB} PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "")
endforeach()

set(SYMENGINE_LIBRARIES symengine ${SYMENGINE_LIBRARIES})
set(SYMENGINE_BUILD_TYPE "Release")
set(SYMENGINE_FOUND yes)

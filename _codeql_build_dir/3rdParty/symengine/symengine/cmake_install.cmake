# Install script for directory: /home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/libsymengine.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES
    "/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/symengine_config.h"
    "/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/symengine_config_cling.h"
    "/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/symengine_export.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/add.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/basic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/basic-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/basic-methods.inc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/complex_double.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/complex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/complex_mpc.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/constants.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/cwrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/derivative.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/dict.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/diophantine.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/eval_arb.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/eval_double.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/eval.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/eval_mpc.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/eval_mpfr.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/expression.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/fields.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/finitediff.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/flint_wrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/functions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/infinity.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/integer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/lambda_double.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/llvm_double.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/logic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/monomials.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/mp_class.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/mp_wrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/mul.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/nan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/ntheory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/ntheory_funcs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/number.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/parser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/parser" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/parser/parser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/parser" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/parser/tokenizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/parser/sbml" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/parser/sbml/sbml_parser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/parser/sbml" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/parser/sbml/sbml_tokenizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/basic_conversions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/cancel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/uexprpoly.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/uintpoly_flint.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/uintpoly.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/uintpoly_piranha.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/upolybase.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/uratpoly.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/usymenginepoly.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/polys" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/polys/msymenginepoly.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/pow.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/prime_sieve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/codegen.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/mathml.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/sbml.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/strprinter.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/latex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/unicode.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/printers" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers/stringbox.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/printers.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/rational.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/real_double.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/real_mpfr.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/rings.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/serialize-cereal.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/series_flint.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/series_generic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/series.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/series_piranha.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/series_visitor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/sets.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/solve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/subs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/symbol.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/symengine_assert.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/symengine_casts.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/symengine_exception.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/symengine_rcp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/tribool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/type_codes.inc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/visitor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/test_visitors.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/assumptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/refine.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/simplify.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/stream_fmt.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/tuple.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrix_expressions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/matrix_expr.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/identity_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/matrix_symbol.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/zero_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/diagonal_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/immutable_dense_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/matrix_add.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/hadamard_product.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/matrix_mul.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/conjugate_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/size.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/transpose.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/matrices" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/matrices/trace.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/access.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/archives" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/archives/adapters.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/archives" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/archives/binary.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/archives" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/archives/json.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/archives" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/archives/portable_binary.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/archives" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/archives/xml.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/cereal.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/helpers.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/polymorphic_impl.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/polymorphic_impl_fwd.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/static_object.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/traits.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/details" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/details/util.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/macros.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/specialize.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/array.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/atomic.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/base_class.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/bitset.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/boost_variant.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/chrono.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/common.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/complex.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types/concepts" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/concepts/pair_associative_container.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/deque.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/forward_list.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/functional.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/list.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/map.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/memory.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/optional.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/polymorphic.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/queue.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/set.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/stack.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/string.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/tuple.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/unordered_map.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/unordered_set.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/utility.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/valarray.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/variant.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal/types" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/types/vector.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/symengine/utilities/cereal/include/cereal" TYPE FILE FILES "/home/runner/work/sdfglib/sdfglib/3rdParty/symengine/symengine/utilities/cereal/include/cereal/version.hpp")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/utilities/matchpycpp/cmake_install.cmake")
  include("/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/utilities/catch/cmake_install.cmake")
  include("/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/tests/cmake_install.cmake")
  include("/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/utilities/matchpycpp/tests/cmake_install.cmake")
  include("/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/utilities/matchpycpp/autogen_tests/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/runner/work/sdfglib/sdfglib/_codeql_build_dir/3rdParty/symengine/symengine/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()

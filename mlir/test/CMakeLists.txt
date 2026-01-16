configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set(MLIR_SDFG_TEST_DEPENDS
    FileCheck
    count
    not
    mlir-opt
    sdfg-opt
)

add_lit_testsuite(check-sdfg-opt "Running the SDFG MLIR frontend regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_SDFG_TEST_DEPENDS}
)
add_lit_testsuites(SDFG-OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_SDFG_TEST_DEPENDS})
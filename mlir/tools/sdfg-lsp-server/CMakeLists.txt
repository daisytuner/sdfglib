get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
    ${dialect_libs}
    MLIROptLib
    MLIRPass
    MLIRSDFGDialect
    MLIRLspServerLib
)

add_llvm_executable(sdfg-lsp-server
    sdfg-lsp-server.cpp
    DEPENDS ${LIBS}
)

llvm_update_compile_flags(sdfg-lsp-server)
target_link_libraries(sdfg-lsp-server PRIVATE ${LIBS})

mlir_check_all_link_libraries(sdfg-lsp-server)
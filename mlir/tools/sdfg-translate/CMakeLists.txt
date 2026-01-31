set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
    ${dialect_libs}
    ${translation_libs}
    MLIROptLib
    MLIRSDFGDialect
)

add_llvm_executable(sdfg-translate
    sdfg-translate.cpp
    DEPENDS ${LIBS}
)

llvm_update_compile_flags(sdfg-translate)
target_link_libraries(sdfg-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(sdfg-translate)
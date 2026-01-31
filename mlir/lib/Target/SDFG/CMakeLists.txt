set_source_files_properties(TranslateToSDFG.cpp PROPERTIES COMPILE_FLAGS -fexceptions)
add_mlir_translation_library(MLIRTargetSDFG
    TranslateToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Target/SDFG

    LINK_LIBS PUBLIC
    MLIRSDFGDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
    sdfglib
)
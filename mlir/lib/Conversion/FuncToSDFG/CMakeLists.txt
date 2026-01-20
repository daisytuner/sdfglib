add_mlir_conversion_library(MLIRFuncToSDFG
    FuncToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Conversion/FuncToSDFG

    DEPENDS
    SDFGConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRSDFGDialect
)
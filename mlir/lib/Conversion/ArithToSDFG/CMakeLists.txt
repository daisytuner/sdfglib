add_mlir_conversion_library(MLIRArithToSDFG
    ArithToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Conversion/ArithToSDFG

    DEPENDS
    SDFGConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRSDFGDialect
)
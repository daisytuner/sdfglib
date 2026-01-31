add_mlir_conversion_library(MLIRLinalgToSDFG
    LinalgToSDFG.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Conversion/LinalgToSDFG

    DEPENDS
    SDFGConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRSDFGDialect
)
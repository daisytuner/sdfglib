add_mlir_dialect_library(MLIRSDFGDialect
    SDFGDialect.cpp
    SDFGOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/SDFG

    DEPENDS
    MLIRSDFGOpsIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
)
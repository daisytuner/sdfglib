set(LLVM_TARGET_DEFINITIONS SDFGOps.td)
mlir_tablegen(SDFGOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SDFGOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SDFGOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=sdfg)
mlir_tablegen(SDFGOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=sdfg)

add_mlir_dialect(SDFGOps sdfg)
add_mlir_doc(SDFGOps SDFGDialect Dialects/ -gen-dialect-doc)
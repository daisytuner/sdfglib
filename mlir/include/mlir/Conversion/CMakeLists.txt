set(LLVM_TARGET_DEFINITIONS SDFGPasses.td)
mlir_tablegen(SDFGPasses.h.inc -gen-pass-decls -name SDFGConversion)
mlir_tablegen(SDFGPasses.capi.h.inc -gen-pass-capi-header --prefix SDFGConversion)
mlir_tablegen(SDFGPasses.capi.cpp.inc -gen-pass-capi-impl --prefix SDFGConversion)
add_public_tablegen_target(SDFGConversionPassIncGen)

add_mlir_doc(SDFGPasses SDFGConversionPasses ./ -gen-pass-doc)
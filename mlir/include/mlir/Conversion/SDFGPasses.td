#ifndef SDFG_CONVERSION_PASSES
#define SDFG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithToSDFG : Pass<"convert-arith-to-sdfg"> {
    let summary = "Convert Arith dialect to SDFG dialect";
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_CONVERSION_PASSES
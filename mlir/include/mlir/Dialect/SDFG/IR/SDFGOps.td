#ifndef SDFG_OPS
#define SDFG_OPS

include "mlir/Dialect/SDFG/IR/SDFGBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class SDFG_Op<string mnemonic, list<Trait> traits = []> :
    Op<SDFG_Dialect, mnemonic, traits>;

def SDFG_ConstantOp : SDFG_Op<"constant", [ConstantLike, AllTypesMatch<["value", "result"]>]> {
    let summary = "Represents a constant literal value";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs SDFG_PrimitiveType:$result);

    let assemblyFormat = "attr-dict $value";
}

def SDFG_TaskletOp : SDFG_Op<"tasklet"> {
    let summary = "Simple computational operations";
    let description = [{
        Tasklets represent primitive computational operations like arithmetic, logic, and type conversions. They are the
        fundamental building blocks for computation in the SDFG.
    }];

    let arguments = (ins SDFG_TaskletCodeAttr:$code, SDFG_PrimitiveType:$inputs);
    let results = (outs SDFG_PrimitiveType:$output);
    let assemblyFormat = "$code `,` $inputs attr-dict `:` `(` type($inputs) `)` `->` type($output)";
}

#endif // SDFG_OPS

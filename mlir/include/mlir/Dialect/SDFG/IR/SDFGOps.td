#ifndef SDFG_OPS
#define SDFG_OPS

include "mlir/Dialect/SDFG/IR/SDFGBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Generell SDFG operation
class SDFG_Op<string mnemonic, list<Trait> traits = []> : Op<SDFG_Dialect, mnemonic, traits>;

// SDFG operation that represents structured control flow
class SDFG_ControlFlowNode<string mnemonic, list<Trait> traits = []> : SDFG_Op<mnemonic, traits>;

// SDFG operation that represents data flow
class SDFG_DataFlowNode<string mnemonic, list<Trait> traits = []> : SDFG_Op<mnemonic, traits>;

// SDFG operation that represents a structured SDFG
def SDFG_SDFGOp : SDFG_Op<"sdfg", [HasParent<"ModuleOp">, NoTerminator, IsolatedFromAbove]> {
    let summary = "Strutured Stateful Data Flow Multigraph";
    let description = [{
        A dataflow-based graph representation for programs.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<StrAttr>:$sym_visibility);
    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;
}

// SDFG operations that represents a return
def SDFG_ReturnOp : SDFG_ControlFlowNode<"return", [Pure, HasParent<"SDFGOp">, ReturnLike, Terminator]> {
    let summary = "Represents a return statement that exits the SDFG";
    let description = [{
        Can return an operand or nothing depending on the result type of the SDFG.
    }];

    let arguments = (ins Optional<AnyType>:$operand);

    let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";

    let hasVerifier = 1;
}

// SDFG data flow node that represents a constant literal value
def SDFG_ConstantOp : SDFG_DataFlowNode<"constant", [ConstantLike, AllTypesMatch<["value", "result"]>]> {
    let summary = "Represents a constant literal value";

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs SDFG_PrimitiveType:$result);

    let assemblyFormat = "attr-dict $value";
}

// SDFG data flow node that represents a simple computation operation
def SDFG_TaskletOp : SDFG_DataFlowNode<"tasklet"> {
    let summary = "Simple computational operations";
    let description = [{
        Tasklets represent primitive computational operations like arithmetic, logic, and type conversions. They are the
        fundamental building blocks for computation in the SDFG.
    }];

    let arguments = (ins SDFG_TaskletCodeAttr:$code, Variadic<SDFG_PrimitiveType>:$inputs);
    let results = (outs SDFG_PrimitiveType:$output);

    let assemblyFormat = "$code `,` $inputs attr-dict `:` `(` type($inputs) `)` `->` type($output)";

    let hasVerifier = 1;
}

#endif // SDFG_OPS

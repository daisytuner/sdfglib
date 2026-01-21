#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

// SDFG Dialect
#include "mlir/Dialect/SDFG/IR/SDFGOpsDialect.h.inc"

// SDFG Dialect Enum Attributes
#include "mlir/Dialect/SDFG/IR/SDFGOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOpsAttributes.h.inc"

// SDFG Dialect Operations
#define GET_OP_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOps.h.inc"

namespace mlir {
namespace sdfg {

bool is_primitive(const Type& type);

/**
 * @brief Get the number of inputs for a tasklet operation
 * @param c TaskletCode operation
 * @return Number of inputs required (arity)
 * @throws InvalidSDFGException if code is invalid
 */
constexpr size_t arity(TaskletCode c) {
    switch (c) {
        case TaskletCode::assign:
        case TaskletCode::int_abs:
            return 1;
        // Integer Relational Ops
        case TaskletCode::int_add:
        case TaskletCode::int_sub:
        case TaskletCode::int_mul:
        case TaskletCode::int_sdiv:
        case TaskletCode::int_srem:
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_and:
        case TaskletCode::int_or:
        case TaskletCode::int_xor:
        case TaskletCode::int_shl:
        case TaskletCode::int_ashr:
        case TaskletCode::int_lshr:
        case TaskletCode::int_smin:
        case TaskletCode::int_smax:
        case TaskletCode::int_umin:
        case TaskletCode::int_scmp:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
            return 2;
        // Comparisions
        case TaskletCode::int_eq:
        case TaskletCode::int_ne:
        case TaskletCode::int_sge:
        case TaskletCode::int_sgt:
        case TaskletCode::int_sle:
        case TaskletCode::int_slt:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return 2;
        // Floating Point
        case TaskletCode::fp_neg:
            return 1;
        case TaskletCode::fp_add:
        case TaskletCode::fp_sub:
        case TaskletCode::fp_mul:
        case TaskletCode::fp_div:
        case TaskletCode::fp_rem:
            return 2;
        // Comparisions
        case TaskletCode::fp_oeq:
        case TaskletCode::fp_one:
        case TaskletCode::fp_oge:
        case TaskletCode::fp_ogt:
        case TaskletCode::fp_ole:
        case TaskletCode::fp_olt:
        case TaskletCode::fp_ord:
        case TaskletCode::fp_ueq:
        case TaskletCode::fp_une:
        case TaskletCode::fp_ugt:
        case TaskletCode::fp_uge:
        case TaskletCode::fp_ult:
        case TaskletCode::fp_ule:
        case TaskletCode::fp_uno:
            return 2;
        case TaskletCode::fp_fma:
            return 3;
    };
    return 0xFFFFFFFF;
};

} // namespace sdfg
} // namespace mlir

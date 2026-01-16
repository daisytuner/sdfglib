#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"

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

bool is_primitive(Type& type);

}
}

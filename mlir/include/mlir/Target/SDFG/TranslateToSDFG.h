#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sdfg {

/// Translate an MLIR operation in the SDFG dialect to C code
LogicalResult translateToSDFG(Operation* op, llvm::raw_ostream& os);

/// Register the SDFG translation with MLIR
void registerToSDFGTranslation();

} // namespace sdfg
} // namespace mlir

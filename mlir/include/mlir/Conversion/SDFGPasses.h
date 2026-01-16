#pragma once

#include "mlir/Conversion/ArithToSDFG/ArithToSDFG.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/SDFGPasses.h.inc"

} // namespace mlir

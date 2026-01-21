#pragma once

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

} // namespace mlir

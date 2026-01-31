#pragma once

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTFUNCTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace func {

void populateFuncToSDFGPatterns(RewritePatternSet& patterns);

} // namespace func
} // namespace mlir

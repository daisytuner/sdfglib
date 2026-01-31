#pragma once

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace arith {

void populateArithToSDFGPatterns(RewritePatternSet& patterns);

} // namespace arith
} // namespace mlir

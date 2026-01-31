#pragma once

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTLINALGTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace linalg {

void populateLinalgToSDFGPatterns(RewritePatternSet& patterns);

} // namespace linalg
} // namespace mlir

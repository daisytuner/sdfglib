#include "mlir/Conversion/ArithToSDFG/ArithToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTARITHTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace arith2sdfg {

struct ConstantOpConversion : public OpRewritePattern<arith::ConstantOp> {
    using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter& rewriter) const override {
        Type type = op.getType();
        if (!sdfg::is_primitive(type)) {
            return rewriter.notifyMatchFailure(op, "unsupported type");
        }
        rewriter.replaceOpWithNewOp<sdfg::ConstantOp>(op, type, op.getValue());
        return success();
    }
};

} // namespace arith2sdfg

namespace {

struct ConvertArithToSDFG : public impl::ConvertArithToSDFGBase<ConvertArithToSDFG> {
    using ConvertArithToSDFGBase::ConvertArithToSDFGBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        arith::populateArithToSDFGPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

} // namespace

namespace arith {

void populateArithToSDFGPatterns(RewritePatternSet& patterns) {
    patterns.add<arith2sdfg::ConstantOpConversion>(patterns.getContext());
}

} // namespace arith
} // namespace mlir

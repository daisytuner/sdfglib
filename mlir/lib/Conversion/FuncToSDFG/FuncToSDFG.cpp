#include "mlir/Conversion/FuncToSDFG/FuncToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTFUNCTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace func2sdfg {

struct FuncOpConversion : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::FuncOp op, PatternRewriter& rewriter) const override {
        if (op.isDeclaration()) {
            // Ignore declarations for now
            return failure();
        }

        // Disallow any attribute other than the function name and type
        for (const auto& named_attr : op->getAttrs()) {
            if (named_attr.getName() != op.getFunctionTypeAttrName() &&
                named_attr.getName() != op.getSymNameAttrName()) {
                return failure();
            }
        }

        // Check that at most one type is returned
        if (op.getFunctionType().getNumResults() > 1) {
            return failure();
        }

        // Replace the FuncOp with an SDFGOp
        sdfg::SDFGOp sdfg_op = rewriter.create<sdfg::SDFGOp>(op.getLoc(), op.getName(), op.getFunctionType());
        rewriter.inlineRegionBefore(op.getBody(), sdfg_op.getBody(), sdfg_op.getBody().end());
        rewriter.eraseOp(op);

        return success();
    }
};

struct ReturnOpConversion : public OpRewritePattern<func::ReturnOp> {
    using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::ReturnOp op, PatternRewriter& rewriter) const override {
        // Check that at most one operand is returned
        if (op.getNumOperands() > 1) {
            return failure();
        }

        Value operand;
        if (op.getNumOperands() == 1) {
            operand = op.getOperand(0);
        }
        rewriter.create<sdfg::ReturnOp>(op.getLoc(), operand);
        rewriter.eraseOp(op);

        return failure();
    }
};

} // namespace func2sdfg

namespace {

struct ConvertFuncToSDFG : public impl::ConvertFuncToSDFGBase<ConvertFuncToSDFG> {
    using ConvertFuncToSDFGBase::ConvertFuncToSDFGBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        func::populateFuncToSDFGPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

} // namespace

namespace func {

void populateFuncToSDFGPatterns(RewritePatternSet& patterns) {
    patterns.add<func2sdfg::FuncOpConversion, func2sdfg::ReturnOpConversion>(patterns.getContext());
}

} // namespace func
} // namespace mlir

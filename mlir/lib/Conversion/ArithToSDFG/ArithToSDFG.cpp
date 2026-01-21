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

struct AddIOpConversion : public OpRewritePattern<arith::AddIOp> {
    using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddIOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getLhs().getType()) || !sdfg::is_primitive(op.getRhs().getType()) ||
            !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp lhs_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), lhs.getType(), lhs);
        sdfg::MemletOp rhs_memlet_op = rewriter.create<sdfg::MemletOp>(lhs_memlet_op.getLoc(), rhs.getType(), rhs);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            rhs_memlet_op.getLoc(),
            result_type,
            sdfg::TaskletCode::int_add,
            SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op =
            rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);
        
        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct ConstantOpConversion : public OpRewritePattern<arith::ConstantOp> {
    using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter& rewriter) const override {
        Type type = op.getType();
        if (!sdfg::is_primitive(type)) {
            return failure();
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
    patterns.add<arith2sdfg::AddIOpConversion, arith2sdfg::ConstantOpConversion>(patterns.getContext());
}

} // namespace arith
} // namespace mlir

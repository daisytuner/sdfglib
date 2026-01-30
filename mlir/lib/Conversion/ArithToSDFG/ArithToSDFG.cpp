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

template<typename OrigOp, sdfg::TaskletCode code>
struct BinaryOpConversion : public OpRewritePattern<OrigOp> {
    using OpRewritePattern<OrigOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(OrigOp op, PatternRewriter& rewriter) const override {
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
            rhs_memlet_op.getLoc(), result_type, code, SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

typedef BinaryOpConversion<arith::AddFOp, sdfg::TaskletCode::fp_add> AddFOpConversion;
typedef BinaryOpConversion<arith::AddIOp, sdfg::TaskletCode::int_add> AddIOpConversion;
typedef BinaryOpConversion<arith::AndIOp, sdfg::TaskletCode::int_and> AndIOpConversion;
typedef BinaryOpConversion<arith::DivFOp, sdfg::TaskletCode::fp_div> DivFOpConversion;
typedef BinaryOpConversion<arith::DivSIOp, sdfg::TaskletCode::int_sdiv> DivSIOpConversion;
typedef BinaryOpConversion<arith::DivUIOp, sdfg::TaskletCode::int_udiv> DivUIOpConversion;
typedef BinaryOpConversion<arith::MaxSIOp, sdfg::TaskletCode::int_smax> MaxSIOpConversion;
typedef BinaryOpConversion<arith::MaxUIOp, sdfg::TaskletCode::int_umax> MaxUIOpConversion;
typedef BinaryOpConversion<arith::MinSIOp, sdfg::TaskletCode::int_smin> MinSIOpConversion;
typedef BinaryOpConversion<arith::MinUIOp, sdfg::TaskletCode::int_umin> MinUIOpConversion;
typedef BinaryOpConversion<arith::MulFOp, sdfg::TaskletCode::fp_mul> MulFOpConversion;
typedef BinaryOpConversion<arith::MulIOp, sdfg::TaskletCode::int_mul> MulIOpConversion;
typedef BinaryOpConversion<arith::OrIOp, sdfg::TaskletCode::int_or> OrIOpConversion;
typedef BinaryOpConversion<arith::RemFOp, sdfg::TaskletCode::fp_rem> RemFOpConversion;
typedef BinaryOpConversion<arith::RemSIOp, sdfg::TaskletCode::int_srem> RemSIOpConversion;
typedef BinaryOpConversion<arith::RemUIOp, sdfg::TaskletCode::int_urem> RemUIOpConversion;
typedef BinaryOpConversion<arith::ShLIOp, sdfg::TaskletCode::int_shl> ShLIOpConversion;
typedef BinaryOpConversion<arith::ShRSIOp, sdfg::TaskletCode::int_ashr> ShRSIOpConversion;
typedef BinaryOpConversion<arith::ShRUIOp, sdfg::TaskletCode::int_lshr> ShRUIOpConversion;
typedef BinaryOpConversion<arith::SubFOp, sdfg::TaskletCode::fp_sub> SubFOpConversion;
typedef BinaryOpConversion<arith::SubIOp, sdfg::TaskletCode::int_sub> SubIOpConversion;
typedef BinaryOpConversion<arith::XOrIOp, sdfg::TaskletCode::int_xor> XOrIOpConversion;

struct CmpFOpConversion : public OpRewritePattern<arith::CmpFOp> {
    using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getLhs().getType()) || !sdfg::is_primitive(op.getRhs().getType()) ||
            !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        // TODO handle AlwaysFalse and AlwaysTrue separately

        // Determine supported tasklet code from predicate
        sdfg::TaskletCode code;
        switch (op.getPredicate()) {
            case arith::CmpFPredicate::OEQ:
                code = sdfg::TaskletCode::fp_oeq;
                break;
            case arith::CmpFPredicate::OGT:
                code = sdfg::TaskletCode::fp_ogt;
                break;
            case arith::CmpFPredicate::OGE:
                code = sdfg::TaskletCode::fp_oge;
                break;
            case arith::CmpFPredicate::OLT:
                code = sdfg::TaskletCode::fp_olt;
                break;
            case arith::CmpFPredicate::OLE:
                code = sdfg::TaskletCode::fp_ole;
                break;
            case arith::CmpFPredicate::ONE:
                code = sdfg::TaskletCode::fp_one;
                break;
            case arith::CmpFPredicate::ORD:
                code = sdfg::TaskletCode::fp_ord;
                break;
            case arith::CmpFPredicate::UEQ:
                code = sdfg::TaskletCode::fp_ueq;
                break;
            case arith::CmpFPredicate::UGT:
                code = sdfg::TaskletCode::fp_ugt;
                break;
            case arith::CmpFPredicate::UGE:
                code = sdfg::TaskletCode::fp_uge;
                break;
            case arith::CmpFPredicate::ULT:
                code = sdfg::TaskletCode::fp_ult;
                break;
            case arith::CmpFPredicate::ULE:
                code = sdfg::TaskletCode::fp_ule;
                break;
            case arith::CmpFPredicate::UNE:
                code = sdfg::TaskletCode::fp_une;
                break;
            case arith::CmpFPredicate::UNO:
                code = sdfg::TaskletCode::fp_uno;
                break;
            default:
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
            rhs_memlet_op.getLoc(), result_type, code, SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct ConstantOpConversion : public OpRewritePattern<arith::ConstantOp> {
    using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter& rewriter) const override {
        Type type = op.getType();
        if (!sdfg::is_tensor_of_primitive(type)) {
            return failure();
        }
        if (auto elements = dyn_cast<ElementsAttr>(op.getValue())) {
            rewriter.replaceOpWithNewOp<sdfg::TensorConstantOp>(op, type, elements);
            return success();
        }
        return failure();
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
    patterns.add<
        arith2sdfg::AddFOpConversion,
        arith2sdfg::AddIOpConversion,
        arith2sdfg::AndIOpConversion,
        arith2sdfg::DivFOpConversion,
        arith2sdfg::DivSIOpConversion,
        arith2sdfg::DivUIOpConversion,
        arith2sdfg::MaxSIOpConversion,
        arith2sdfg::MaxUIOpConversion,
        arith2sdfg::MinSIOpConversion,
        arith2sdfg::MinUIOpConversion,
        arith2sdfg::MulFOpConversion,
        arith2sdfg::MulIOpConversion,
        arith2sdfg::OrIOpConversion,
        arith2sdfg::RemFOpConversion,
        arith2sdfg::RemSIOpConversion,
        arith2sdfg::RemUIOpConversion,
        arith2sdfg::ShLIOpConversion,
        arith2sdfg::ShRSIOpConversion,
        arith2sdfg::ShRUIOpConversion,
        arith2sdfg::SubFOpConversion,
        arith2sdfg::SubIOpConversion,
        arith2sdfg::XOrIOpConversion,
        arith2sdfg::CmpFOpConversion,
        arith2sdfg::ConstantOpConversion>(patterns.getContext());
}

} // namespace arith
} // namespace mlir

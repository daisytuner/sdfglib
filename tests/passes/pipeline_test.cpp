#include "sdfg/passes/pipeline.h"

#include <gtest/gtest.h>

#include "sdfg/passes/symbolic/symbol_promotion.h"

using namespace sdfg;

TEST(PipelineTest, RegisterPass) {
    passes::Pipeline pipeline("pipeline");
    pipeline.register_pass<passes::SymbolPromotion>();

    EXPECT_EQ(pipeline.size(), 1);
}

TEST(PipelineTest, ExpressionCombine) {
    passes::Pipeline expression_combine = passes::Pipeline::expression_combine();
    EXPECT_EQ(expression_combine.name(), "ExpressionCombine");
    EXPECT_EQ(expression_combine.size(), 3);
}

TEST(PipelineTest, MemletCombine) {
    passes::Pipeline memlet_combine = passes::Pipeline::memlet_combine();
    EXPECT_EQ(memlet_combine.name(), "MemletCombine");
    EXPECT_EQ(memlet_combine.size(), 4);
}

TEST(PipelineTest, ControlFlowSimplification) {
    passes::Pipeline control_flow_simplification = passes::Pipeline::controlflow_simplification();
    EXPECT_EQ(control_flow_simplification.name(), "ControlFlowSimplification");
    EXPECT_EQ(control_flow_simplification.size(), 3);
}

TEST(PipelineTest, DataParallelism) {
    passes::Pipeline data_parallelism = passes::Pipeline::data_parallelism();
    EXPECT_EQ(data_parallelism.name(), "DataParallelism");
    EXPECT_EQ(data_parallelism.size(), 3);
}

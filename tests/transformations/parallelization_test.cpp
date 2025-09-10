#include "sdfg/transformations/parallelization.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(ParallelizationTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::Parallelization transformation(loop);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(loop.schedule_type().value(), structured_control_flow::ScheduleType_CPU_Parallel::value());
}

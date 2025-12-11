#include "sdfg/passes/structured_control_flow/pointer_evolution.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(PointerEvolutionTest, ZeroOffsets) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("i", sym_desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root, sym, symbolic::Lt(sym, symbolic::integer(10)), symbolic::one(), symbolic::add(sym, symbolic::one())
    );

    // A[0] = A[0] + 1
    auto& block1 = builder.add_block(loop.root());
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer float_ptr(float_desc);
    auto& a_input = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0f", float_desc);
    auto& a_output = builder.add_access(block1, "A");

    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& edge1 = builder.add_computational_memlet(block1, a_input, tasklet, "_in1", {symbolic::zero()}, float_ptr);
    auto& edge2 = builder.add_computational_memlet(block1, one_node, tasklet, "_in2", {});
    auto& edge3 = builder.add_computational_memlet(block1, tasklet, "_out", a_output, {symbolic::zero()}, float_ptr);

    // A = ((int8*) A)[4]
    auto& block2 = builder.add_block(loop.root());
    auto& a_iter_in = builder.add_access(block2, "A");
    auto& a_iter_out = builder.add_access(block2, "A");

    types::Scalar byte_desc(types::PrimitiveType::UInt8);
    types::Pointer byte_ptr(byte_desc);
    builder.add_reference_memlet(block2, a_iter_in, a_iter_out, {symbolic::integer(4)}, byte_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::PointerEvolution pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(block2.dataflow().nodes().empty());
    EXPECT_TRUE(block2.dataflow().edges().empty());
}

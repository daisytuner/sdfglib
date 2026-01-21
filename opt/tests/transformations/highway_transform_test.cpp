#include "sdfg/transformations/highway_transform.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(HighwayTransformTest, ContiguousReadWrite) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // A[i] = A[i]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(loop1.schedule_type().value(), highway::ScheduleType_Highway::value());
}

TEST(HighwayTransformTest, ConstantRead) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // A[i] = A[i]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("j")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(loop1.schedule_type().value(), highway::ScheduleType_Highway::value());
}

TEST(HighwayTransformTest, ContiguousInLinearizedForm) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // A[i] = A[i]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("j")}, desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", A_out, {symbolic::add(indvar1, symbolic::mul(symbolic::symbol("j"), bound1))}, desc
    );

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(loop1.schedule_type().value(), highway::ScheduleType_Highway::value());
}

TEST(HighwayTransformTest, NonContiguousLoop) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(2));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // A[i] = A[i]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(HighwayTransformTest, NonContiguousMemory) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", opaque_desc);
    builder.add_container("tmp", opaque_desc);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add gep
    // a = A + i
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "a");
    builder.add_reference_memlet(block, A_in, A_out, {indvar1}, types::Pointer(static_cast<types::IType&>(opaque_desc)));

    // Load a
    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "a");
    auto& tmp_out = builder.add_access(block1, "tmp");
    builder.add_dereference_memlet(block1, a_in, tmp_out, true, types::Pointer(static_cast<types::IType&>(opaque_desc)));

    // tmp[i] = tmp[i]
    auto& block2 = builder.add_block(body1);
    auto& A_in2 = builder.add_access(block2, "tmp");
    auto& A_out2 = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, A_in2, tasklet2, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", A_out2, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(HighwayTransformTest, ScalarOutputArgument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("a", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // a = A[i]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {}, base_desc);

    // A[i] = a
    auto& block2 = builder.add_block(body1);
    auto& A_in2 = builder.add_access(block2, "a");
    auto& A_out2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, A_in2, tasklet2, "_in", {}, base_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", A_out2, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::HighwayTransform transformation(loop1);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

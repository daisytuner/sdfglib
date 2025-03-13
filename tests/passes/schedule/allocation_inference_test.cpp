#include "sdfg/passes/schedule/allocation_inference.h"

#include <gtest/gtest.h>

#include "fixtures/polybench.h"

using namespace sdfg;

TEST(AllocationInferenceTest, Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    // Write to i
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", sym_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(structured_sdfg);

    passes::AllocationInference AI_pass;
    AI_pass.run_pass(*schedule);

    EXPECT_EQ(schedule->allocation_type("i"), AllocationType::ALLOCATE);
}

TEST(AllocationInferenceTest, Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    types::Array desc_array(base_desc, {symbolic::integer(10)});
    builder.add_container("a", desc_array);

    // Write to i
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(structured_sdfg);

    passes::AllocationInference AI_pass;
    AI_pass.run_pass(*schedule);

    EXPECT_EQ(schedule->allocation_type("a"), AllocationType::ALLOCATE);
}

TEST(AllocationInferenceTest, Structure) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);

    auto& S_def = builder.add_structure("S");
    S_def.add_member(base_desc);

    types::Structure desc_struct("S");
    builder.add_container("s", desc_struct);

    // Write to i
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "s");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(structured_sdfg);

    passes::AllocationInference AI_pass;
    AI_pass.run_pass(*schedule);

    EXPECT_EQ(schedule->allocation_type("s"), AllocationType::ALLOCATE);
}

TEST(AllocationInferenceTest, Pointer) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("a", desc);

    // Write to i
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"0", base_desc}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(structured_sdfg);

    passes::AllocationInference AI_pass;
    AI_pass.run_pass(*schedule);

    EXPECT_EQ(schedule->allocation_type("a"), AllocationType::ALLOCATE);
}

TEST(AllocationInferenceTest, Pointer_Refs) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("a", desc);

    // Write to i
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "a");
    builder.add_memlet(block, input_node, "refs", output_node, "void", {symbolic::integer(0)});

    auto structured_sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(structured_sdfg);

    passes::AllocationInference AI_pass;
    AI_pass.run_pass(*schedule);

    EXPECT_EQ(schedule->allocation_type("a"), AllocationType::DECLARE);
}

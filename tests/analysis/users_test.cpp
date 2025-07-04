#include "sdfg/analysis/users.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(UsersTest, Transition_WAR) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    auto sym1 = symbolic::symbol("A");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::add(sym1, symbolic::one())}});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto& transition1 = builder_opt.subject().root().at(0).second;

    // Check result
    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 1);

    auto& read = reads.at(0);
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &transition1);
    EXPECT_EQ(read->subsets().size(), 1);
    EXPECT_TRUE(read->subsets().at(0).empty());

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 1);

    auto& write = writes.at(0);
    EXPECT_EQ(write->use(), analysis::Use::WRITE);
    EXPECT_EQ(write->container(), "A");
    EXPECT_EQ(write->element(), &transition1);
    EXPECT_EQ(write->subsets().size(), 1);
    EXPECT_TRUE(write->subsets().at(0).empty());

    EXPECT_TRUE(users.dominates(*read, *write));
    EXPECT_FALSE(users.dominates(*write, *read));
    EXPECT_TRUE(users.post_dominates(*write, *read));
    EXPECT_FALSE(users.post_dominates(*read, *write));
}

TEST(UsersTest, Transition_WAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    auto sym1 = symbolic::symbol("A");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::zero()}});
    auto& block2 = builder.add_block(root, {{sym1, symbolic::zero()}});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto& transition1 = builder_opt.subject().root().at(0).second;
    auto& transition2 = builder_opt.subject().root().at(1).second;

    // Check result
    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 0);

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 2);

    auto& write1 = writes.at(0);
    auto& write2 = writes.at(1);
    if (write1->element() == &transition2) {
        std::swap(write1, write2);
    }

    EXPECT_EQ(write1->use(), analysis::Use::WRITE);
    EXPECT_EQ(write1->container(), "A");
    EXPECT_EQ(write1->element(), &transition1);
    EXPECT_EQ(write1->subsets().size(), 1);
    EXPECT_TRUE(write1->subsets().at(0).empty());

    EXPECT_EQ(write2->use(), analysis::Use::WRITE);
    EXPECT_EQ(write2->container(), "A");
    EXPECT_EQ(write2->element(), &transition2);
    EXPECT_EQ(write2->subsets().size(), 1);
    EXPECT_TRUE(write2->subsets().at(0).empty());

    EXPECT_TRUE(users.dominates(*write1, *write2));
    EXPECT_FALSE(users.dominates(*write2, *write1));
    EXPECT_TRUE(users.post_dominates(*write2, *write1));
    EXPECT_FALSE(users.post_dominates(*write1, *write2));
}

TEST(UsersTest, Transition_RAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    auto sym1 = symbolic::symbol("A");
    auto sym2 = symbolic::symbol("B");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::zero()}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto& transition1 = builder_opt.subject().root().at(0).second;
    auto& transition2 = builder_opt.subject().root().at(1).second;

    // Check result
    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 1);

    auto& read = reads.at(0);
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &transition2);
    EXPECT_EQ(read->subsets().size(), 1);
    EXPECT_TRUE(read->subsets().at(0).empty());

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 1);

    auto& write = writes.at(0);
    EXPECT_EQ(write->use(), analysis::Use::WRITE);
    EXPECT_EQ(write->container(), "A");
    EXPECT_EQ(write->element(), &transition1);
    EXPECT_EQ(write->subsets().size(), 1);
    EXPECT_TRUE(write->subsets().at(0).empty());

    EXPECT_TRUE(users.dominates(*write, *read));
    EXPECT_FALSE(users.dominates(*read, *write));
    EXPECT_TRUE(users.post_dominates(*read, *write));
    EXPECT_FALSE(users.post_dominates(*write, *read));
}

TEST(UsersTest, AccessNode_Scalar_WAR) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::Scalar(types::PrimitiveType::Int32)}}
    );
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 1);

    auto& read = reads.at(0);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->element(), &input_node);

    auto& read_subsets = read->subsets();
    EXPECT_EQ(read_subsets.size(), 1);
    EXPECT_TRUE(read_subsets.at(0).empty());

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 1);

    auto& write = writes.at(0);
    EXPECT_EQ(write->container(), "A");
    EXPECT_EQ(write->use(), analysis::Use::WRITE);
    EXPECT_EQ(write->element(), &output_node);

    auto& write_subsets = write->subsets();
    EXPECT_EQ(write_subsets.size(), 1);
    EXPECT_TRUE(write_subsets.at(0).empty());

    EXPECT_TRUE(users.dominates(*read, *write));
    EXPECT_FALSE(users.dominates(*write, *read));
    EXPECT_TRUE(users.post_dominates(*write, *read));
    EXPECT_FALSE(users.post_dominates(*read, *write));
}

TEST(UsersTest, AccessNode_Scalar_WAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(
        block2,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 0);

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 2);

    auto& write1 = writes.at(0);
    auto& write2 = writes.at(1);
    if (write1->element() == &output_node2) {
        std::swap(write1, write2);
    }

    EXPECT_EQ(write1->use(), analysis::Use::WRITE);
    EXPECT_EQ(write1->container(), "A");
    EXPECT_EQ(write1->element(), &output_node);
    EXPECT_EQ(write1->subsets().size(), 1);
    EXPECT_TRUE(write1->subsets().at(0).empty());

    EXPECT_EQ(write2->use(), analysis::Use::WRITE);
    EXPECT_EQ(write2->container(), "A");
    EXPECT_EQ(write2->element(), &output_node2);
    EXPECT_EQ(write2->subsets().size(), 1);
    EXPECT_TRUE(write2->subsets().at(0).empty());

    EXPECT_TRUE(users.dominates(*write1, *write2));
    EXPECT_FALSE(users.dominates(*write2, *write1));
    EXPECT_TRUE(users.post_dominates(*write2, *write1));
    EXPECT_FALSE(users.post_dominates(*write1, *write2));
}

TEST(UsersTest, AccessNode_Scalar_RAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"0", types::PrimitiveType::Int32}}
    );
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});

    auto& output_node2 = builder.add_access(block, "B");
    auto& tasklet2 = builder.add_tasklet(
        block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::Scalar(types::PrimitiveType::Int32)}}
    );
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    auto reads = users.reads("A");
    EXPECT_EQ(reads.size(), 1);

    auto& read = reads.at(0);
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &output_node);
    EXPECT_EQ(read->subsets().size(), 1);
    EXPECT_TRUE(read->subsets().at(0).empty());

    auto writes = users.writes("A");
    EXPECT_EQ(writes.size(), 1);

    auto& write = writes.at(0);
    EXPECT_EQ(write->use(), analysis::Use::WRITE);
    EXPECT_EQ(write->container(), "A");
    EXPECT_EQ(write->element(), &output_node);
    EXPECT_EQ(write->subsets().size(), 1);
    EXPECT_TRUE(write->subsets().at(0).empty());

    EXPECT_TRUE(users.dominates(*write, *read));
    EXPECT_FALSE(users.dominates(*read, *write));
    EXPECT_TRUE(users.post_dominates(*read, *write));
    EXPECT_FALSE(users.post_dominates(*write, *read));
}

TEST(UsersTest, For_Definition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    auto sym1 = symbolic::symbol("A");
    auto sym2 = symbolic::symbol("B");

    auto& root = builder.subject().root();
    auto& loop =
        builder
            .add_for(root, sym1, symbolic::Lt(sym1, sym2), symbolic::integer(0), symbolic::add(sym1, symbolic::one()));

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    auto reads = users.reads("A");
    auto writes = users.writes("A");

    EXPECT_EQ(reads.size(), 2);
    EXPECT_EQ(writes.size(), 2);

    auto read1 = dynamic_cast<analysis::ForUser*>(reads.at(0));
    auto read2 = dynamic_cast<analysis::ForUser*>(reads.at(1));
    auto write1 = dynamic_cast<analysis::ForUser*>(writes.at(0));
    auto write2 = dynamic_cast<analysis::ForUser*>(writes.at(1));

    if (read1->is_update()) {
        std::swap(read1, read2);
    }
    if (write1->is_update()) {
        std::swap(write1, write2);
    }

    EXPECT_TRUE(write1->is_init());
    EXPECT_EQ(write1->use(), analysis::Use::WRITE);
    EXPECT_EQ(write1->container(), "A");

    EXPECT_TRUE(write2->is_update());
    EXPECT_EQ(write2->use(), analysis::Use::WRITE);
    EXPECT_EQ(write2->container(), "A");

    EXPECT_TRUE(read1->is_condition());
    EXPECT_EQ(read1->use(), analysis::Use::READ);
    EXPECT_EQ(read1->container(), "A");

    EXPECT_TRUE(read2->is_update());
    EXPECT_EQ(read2->use(), analysis::Use::READ);
    EXPECT_EQ(read2->container(), "A");

    EXPECT_TRUE(users.dominates(*write1, *read1));
    EXPECT_TRUE(users.dominates(*write1, *read2));
    EXPECT_TRUE(users.dominates(*write1, *write2));

    EXPECT_TRUE(users.dominates(*read2, *write2));
}

TEST(UsersTest, Locals_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32), true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("a"), symbolic::integer(0)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto locals = users.locals(root);
    EXPECT_EQ(locals.size(), 0);
}

TEST(UsersTest, Locals_External) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32), false, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("a"), symbolic::integer(0)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto locals = users.locals(root);
    EXPECT_EQ(locals.size(), 0);
}

TEST(UsersTest, Locals_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& sequence_1 = builder.add_sequence(root);
    auto& sequence_2 = builder.add_sequence(root);
    auto& block_1 = builder.add_block(sequence_1);
    auto& block_2 = builder.add_block(sequence_2, {{symbolic::symbol("a"), symbolic::integer(0)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto sequence_1_locals = users.locals(sequence_1);
    EXPECT_EQ(sequence_1_locals.size(), 0);

    auto sequence_2_locals = users.locals(sequence_2);
    EXPECT_EQ(sequence_2_locals.size(), 1);
    EXPECT_TRUE(sequence_2_locals.find("a") != sequence_2_locals.end());

    auto root_locals = users.locals(root);
    EXPECT_EQ(root_locals.size(), 1);
    EXPECT_TRUE(root_locals.find("a") != root_locals.end());
}

TEST(UsersTest, Locals_Transient2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("a", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& sequence_1 = builder.add_sequence(root);
    auto& sequence_2 = builder.add_sequence(root);
    auto& block_1 = builder.add_block(sequence_1, {{symbolic::symbol("a"), symbolic::integer(0)}});
    auto& block_2 = builder.add_block(sequence_2, {{symbolic::symbol("a"), symbolic::integer(0)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    auto sequence_1_locals = users.locals(sequence_1);
    EXPECT_EQ(sequence_1_locals.size(), 0);

    auto sequence_2_locals = users.locals(sequence_2);
    EXPECT_EQ(sequence_2_locals.size(), 0);

    auto root_locals = users.locals(root);
    EXPECT_EQ(root_locals.size(), 1);
    EXPECT_TRUE(root_locals.find("a") != root_locals.end());
}

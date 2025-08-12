#include "sdfg/passes/symbolic/symbol_promotion.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

using namespace sdfg;

TEST(SymbolPromotionTest, as_symbol_int) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto res = passes::SymbolPromotion::as_symbol(block.dataflow(), tasklet, "0");
    codegen::CPPLanguageExtension language_extension;
    EXPECT_EQ(language_extension.expression(res), "0");
}

TEST(SymbolPromotionTest, as_symbol_long) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"4294967296"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto res = passes::SymbolPromotion::as_symbol(block.dataflow(), tasklet, "4294967296");
    codegen::CPPLanguageExtension language_extension;
    EXPECT_EQ(language_extension.expression(res), "4294967296");
}

TEST(SymbolPromotionTest, as_symbol_input) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto res = passes::SymbolPromotion::as_symbol(block.dataflow(), tasklet, "_in");
    codegen::CPPLanguageExtension language_extension;
    EXPECT_EQ(language_extension.expression(res), "j");
}

TEST(SymbolPromotionTest, Assignment1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(0)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Assignment2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *sym1));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Add1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"0", "1"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(1)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Add2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::add(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

TEST(SymbolPromotionTest, Sub1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::sub, "_out", {"0", "1"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(-1)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Sub2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::sub(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

TEST(SymbolPromotionTest, Mul1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::mul, "_out", {"0", "1"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(0)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Mul2) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::mul(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

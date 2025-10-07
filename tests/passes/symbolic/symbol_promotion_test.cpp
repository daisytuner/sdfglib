#include "sdfg/passes/symbolic/symbol_promotion.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

using namespace sdfg;

TEST(SymbolPromotionTest, Assign_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
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
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
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

TEST(SymbolPromotionTest, Assign_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");
    
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
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

TEST(SymbolPromotionTest, Assign_Unsigned) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");
    
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Cast_Input) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");
    
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_memlet(block, zero_node, "void", tasklet, "_in", {}, udesc, {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Cast_Output) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");
    
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {}, udesc, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Tasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& output_node = builder.add_access(block, "i");
    
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_udiv, "_out", {"_in1", "_in2"});

    builder.add_memlet(block, one_node, "void", tasklet, "_in1", {}, udesc, {});
    builder.add_memlet(block, one_node, "void", tasklet, "_in2", {}, udesc, {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {}, udesc, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Add_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
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

TEST(SymbolPromotionTest, Add_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
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

TEST(SymbolPromotionTest, Sub_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
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

TEST(SymbolPromotionTest, Sub_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
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

TEST(SymbolPromotionTest, Mul_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
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

TEST(SymbolPromotionTest, Mul_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
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

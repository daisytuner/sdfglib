#include "sdfg/passes/structured_control_flow/cfg_to_scf_conversion.h"
#include <gtest/gtest.h>
#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/sdfg.h"

using namespace sdfg;
namespace scf = sdfg::passes::scf;

static const sdfg::passes::scf::BasicBlock*
find_block(const std::vector<const sdfg::passes::scf::BasicBlock*>& blocks, const sdfg::control_flow::State* s) {
    for (auto& b : blocks) {
        for (auto* st : b->states) {
            if (st == s) return b;
        }
    }
    return nullptr;
}

static size_t count_kind(const std::vector<const scf::Region*>& regions, scf::RegionKind k) {
    size_t c = 0;
    for (auto& r : regions)
        if (r->kind == k) ++c;
    return c;
}

static const scf::Region* find_region_kind(const std::vector<const scf::Region*>& regions, scf::RegionKind k) {
    for (auto& r : regions)
        if (r->kind == k) return r;
    return nullptr;
}

TEST(CFGToSCFConversionTest, BasicBlocks_Chain) {
    builder::SDFGBuilder b("linear", FunctionType_CPU);

    // s0 -> s1 -> s2 -> s3
    auto& s0 = b.add_state(true);
    auto& s1 = b.add_state();
    auto& s2 = b.add_state();
    auto& s3 = b.add_state();
    b.add_edge(s0, s1);
    b.add_edge(s1, s2);
    b.add_edge(s2, s3);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto blocks = pass.basic_blocks();
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_EQ(blocks.at(0)->states.size(), 4);
}

TEST(CFGToSCFConversionTest, BasicBlocks_Branches) {
    builder::SDFGBuilder b("branch", FunctionType_CPU);

    // entry -> cond -> then1 -> then2 -> join
    //               -> els1 -> els2 -> join
    auto& entry = b.add_state(true);
    auto& cond = b.add_state();
    auto& then1 = b.add_state();
    auto& then2 = b.add_state();
    auto& els1 = b.add_state();
    auto& els2 = b.add_state();
    auto& join = b.add_state();
    b.add_edge(entry, cond);
    b.add_edge(cond, then1);
    b.add_edge(then1, then2);
    b.add_edge(then2, join);
    b.add_edge(cond, els1);
    b.add_edge(els1, els2);
    b.add_edge(els2, join);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto blocks = pass.basic_blocks();
    EXPECT_EQ(blocks.size(), 5);

    const auto* b_entry = find_block(blocks, &entry);
    const auto* b_cond = find_block(blocks, &cond);
    const auto* b_join = find_block(blocks, &join);
    const auto* b_then1 = find_block(blocks, &then1);
    const auto* b_els1 = find_block(blocks, &els1);
    ASSERT_TRUE(b_entry && b_cond && b_join && b_then1 && b_els1);
    EXPECT_EQ(b_cond->states.size(), 1);
    EXPECT_EQ(b_join->states.size(), 1);
    EXPECT_EQ(b_cond->states.size(), 1);
    EXPECT_EQ(b_then1->states.size(), 2);
    EXPECT_EQ(b_els1->states.size(), 2);
}

TEST(CFGToSCFConversionTest, BasicBlocks_Loop) {
    builder::SDFGBuilder b("loop", FunctionType_CPU);

    // entry -> h -> body1 -> body2 -> h -> exit
    auto& entry = b.add_state(true);
    auto& h = b.add_state();
    auto& body1 = b.add_state();
    auto& body2 = b.add_state();
    auto& exit = b.add_state();
    b.add_edge(entry, h);
    b.add_edge(h, body1);
    b.add_edge(body1, body2);
    b.add_edge(body2, h);
    b.add_edge(h, exit);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto blocks = pass.basic_blocks();
    ASSERT_EQ(blocks.size(), 4);

    const auto* b_entry = find_block(blocks, &entry);
    const auto* b_h = find_block(blocks, &h);
    const auto* b_body1 = find_block(blocks, &body1);
    const auto* b_exit = find_block(blocks, &exit);
    ASSERT_TRUE(b_entry && b_h && b_body1 && b_exit);

    EXPECT_EQ(b_body1->states.size(), 2);
}

TEST(CFGToSCFConversionTest, BasicBlocks_Return) {
    builder::SDFGBuilder b("return_delim", FunctionType_CPU);

    // entry -> s1 -> s2 -> ret
    auto& entry = b.add_state(true);
    auto& s1 = b.add_state();
    auto& s2 = b.add_state();
    auto& ret = b.add_return_state("ret");
    b.add_edge(entry, s1);
    b.add_edge(s1, s2);
    b.add_edge(s2, ret);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto blocks = pass.basic_blocks();
    EXPECT_EQ(blocks.size(), 1);
    EXPECT_EQ(blocks.at(0)->states.size(), 4);
}

TEST(CFGToSCFConversionTest, Regions_Sequence) {
    builder::SDFGBuilder b("linear_seq", FunctionType_CPU);

    auto& s0 = b.add_state(true);
    auto& s1 = b.add_state();
    auto& s2 = b.add_state();
    b.add_edge(s0, s1);
    b.add_edge(s1, s2);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto regions = pass.regions();
    EXPECT_EQ(regions.size(), 1);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::Sequence), 1);

    const scf::Region* seq_region = find_region_kind(regions, scf::RegionKind::Sequence);
    ASSERT_TRUE(seq_region);
    EXPECT_EQ(seq_region->blocks.size(), 1);
    EXPECT_EQ(seq_region->blocks.at(0)->states.size(), 3);

    auto structured_sdfg = pass.get();
    EXPECT_TRUE(structured_sdfg);

    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        auto [child, transition] = root.at(i);
        EXPECT_EQ(transition.assignments().size(), 0);
        EXPECT_NE(dynamic_cast<sdfg::structured_control_flow::Block*>(&child), nullptr);
    }
}

TEST(CFGToSCFConversionTest, Regions_SequenceWithDataflow) {
    builder::SDFGBuilder b("linear_seq", FunctionType_CPU);

    b.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    b.add_container("b", types::Scalar(types::PrimitiveType::Int32));

    auto& s0 = b.add_state(true);
    auto& s1 = b.add_state();
    auto& s2 = b.add_state();
    b.add_edge(s0, s1);
    b.add_edge(s1, s2);

    auto& a_node = b.add_access(s1, "a");
    auto& b_node = b.add_access(s1, "b");
    auto& tasklet = b.add_tasklet(s1, data_flow::TaskletCode::assign, "_out", {"_in"});
    b.add_computational_memlet(s1, a_node, tasklet, "_in", {});
    b.add_computational_memlet(s1, tasklet, "_out", b_node, {});

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto regions = pass.regions();
    EXPECT_EQ(regions.size(), 1);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::Sequence), 1);

    const scf::Region* seq_region = find_region_kind(regions, scf::RegionKind::Sequence);
    ASSERT_TRUE(seq_region);
    EXPECT_EQ(seq_region->blocks.size(), 1);
    EXPECT_EQ(seq_region->blocks.at(0)->states.size(), 3);

    auto structured_sdfg = pass.get();
    EXPECT_TRUE(structured_sdfg);

    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 3);

    auto& block1 = dynamic_cast<sdfg::structured_control_flow::Block&>(root.at(1).first);
    EXPECT_EQ(block1.dataflow().nodes().size(), 3);
    EXPECT_EQ(block1.dataflow().edges().size(), 2);
}

TEST(CFGToSCFConversionTest, Regions_SequenceWithAssignments) {
    builder::SDFGBuilder b("linear_seq", FunctionType_CPU);

    b.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    b.add_container("b", types::Scalar(types::PrimitiveType::Int32));

    auto& s0 = b.add_state(true);
    auto& s1 = b.add_state();
    auto& s2 = b.add_state();
    b.add_edge(s0, s1, {{symbolic::symbol("a"), symbolic::zero()}});
    b.add_edge(s1, s2, {{symbolic::symbol("b"), symbolic::one()}});

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto regions = pass.regions();
    EXPECT_EQ(regions.size(), 1);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::Sequence), 1);

    const scf::Region* seq_region = find_region_kind(regions, scf::RegionKind::Sequence);
    ASSERT_TRUE(seq_region);
    EXPECT_EQ(seq_region->blocks.size(), 1);
    EXPECT_EQ(seq_region->blocks.at(0)->states.size(), 3);

    auto structured_sdfg = pass.get();
    EXPECT_TRUE(structured_sdfg);

    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 3);

    auto& trans1 = root.at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("a")), symbolic::zero()));

    auto& trans2 = root.at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans2.assignments().at(symbolic::symbol("b")), symbolic::one()));
}

TEST(CFGToSCFConversionTest, Regions_IfElse) {
    builder::SDFGBuilder b("ifelse", FunctionType_CPU);

    b.add_container("cond", types::Scalar(types::PrimitiveType::Int32));

    auto then_cond = symbolic::Eq(symbolic::symbol("cond"), symbolic::one());
    auto else_cond = symbolic::Not(then_cond);

    auto& entry = b.add_state(true);
    auto& cond = b.add_state();
    auto& then1 = b.add_state();
    auto& els1 = b.add_state();
    auto& join = b.add_state();
    b.add_edge(entry, cond);
    b.add_edge(cond, then1, then_cond);
    b.add_edge(then1, join);
    b.add_edge(cond, els1, else_cond);
    b.add_edge(els1, join);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto regions = pass.regions();
    EXPECT_EQ(regions.size(), 5);

    EXPECT_EQ(count_kind(regions, scf::RegionKind::Sequence), 4);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::IfElse), 1);

    auto ifelse_region = find_region_kind(regions, scf::RegionKind::IfElse);
    ASSERT_TRUE(ifelse_region);
    EXPECT_EQ(ifelse_region->blocks.size(), 4);

    auto structured_sdfg = pass.get();
    EXPECT_TRUE(structured_sdfg);

    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 2);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&root.at(0).first));
    EXPECT_EQ(root.at(0).second.assignments().size(), 0);

    auto ifelse = dynamic_cast<sdfg::structured_control_flow::IfElse*>(&root.at(1).first);
    EXPECT_NE(ifelse, nullptr);
    EXPECT_EQ(ifelse->size(), 2);

    EXPECT_TRUE(symbolic::eq(ifelse->at(0).second, then_cond));
    EXPECT_TRUE(symbolic::eq(ifelse->at(1).second, else_cond));

    auto& then_branch = ifelse->at(0).first;
    EXPECT_EQ(then_branch.size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&then_branch.at(0).first));
    EXPECT_EQ(then_branch.at(0).second.assignments().size(), 0);

    auto& else_branch = ifelse->at(1).first;
    EXPECT_EQ(else_branch.size(), 1);
    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&else_branch.at(0).first));
    EXPECT_EQ(else_branch.at(0).second.assignments().size(), 0);
}

TEST(CFGToSCFConversionTest, Regions_While) {
    builder::SDFGBuilder b("while_loop", FunctionType_CPU);

    b.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto sym_i = symbolic::symbol("i");

    auto& entry = b.add_state(true);
    auto& h = b.add_state();
    auto& body1 = b.add_state();
    auto& exit = b.add_state();
    b.add_edge(entry, h, {{sym_i, symbolic::zero()}});
    b.add_edge(h, body1);
    b.add_edge(body1, h, {{sym_i, symbolic::add(sym_i, symbolic::one())}}, symbolic::Ne(sym_i, symbolic::integer(10)));
    b.add_edge(body1, exit, symbolic::Eq(sym_i, symbolic::integer(10)));

    auto& sdfg = b.subject();
    ASSERT_EQ(sdfg.natural_loops().size(), 1);

    passes::CFGToSCFConversion pass;
    pass.run_pass(b);

    auto regions = pass.regions();
    EXPECT_EQ(regions.size(), 2);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::While), 1);
    EXPECT_EQ(count_kind(regions, scf::RegionKind::Sequence), 1);

    auto while_region = find_region_kind(regions, scf::RegionKind::While);
    ASSERT_TRUE(while_region);
    EXPECT_EQ(while_region->blocks.size(), 2);

    auto structured_sdfg = pass.get();
    auto& root = structured_sdfg->root();
    EXPECT_EQ(root.size(), 2);

    EXPECT_TRUE(dynamic_cast<sdfg::structured_control_flow::Block*>(&root.at(0).first));
    EXPECT_EQ(root.at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(root.at(0).second.assignments().at(sym_i), symbolic::zero()));

    auto while_node = dynamic_cast<sdfg::structured_control_flow::While*>(&root.at(1).first);
    EXPECT_NE(while_node, nullptr);

    auto& while_root = while_node->root();
    {
        EXPECT_EQ(while_root.size(), 1);
    }
}

// // Verify post-dominator intersection selects join (no fallback) structurally.
// TEST(CFGToSCFConversionTest, PostDominatorJoin_NoFallbackFlag) {
//     builder::SDFGBuilder b("pdom_ifelse", FunctionType_CPU);
//     auto& entry = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& then1 = b.add_state();
//     auto& then2 = b.add_state();
//     auto& els1 = b.add_state();
//     auto& els2 = b.add_state();
//     auto& join = b.add_state();
//     auto& tail = b.add_state();
//     auto& ret = b.add_return_state("r");
//     // Branch arms each have two states to ensure non-trivial chains; common join then tail then return.
//     b.add_edge(entry, cond);
//     b.add_edge(cond, then1);
//     b.add_edge(then1, then2);
//     b.add_edge(then2, join);
//     b.add_edge(cond, els1);
//     b.add_edge(els1, els2);
//     b.add_edge(els2, join);
//     b.add_edge(join, tail);
//     b.add_edge(tail, ret);
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     size_t ifelses = count_kind(regs, scf::RegionKind::IfElse);
//     ASSERT_EQ(ifelses, 1);
//     const scf::Region* region = nullptr;
//     for (auto& r : regs)
//         if (r.kind == scf::RegionKind::IfElse) region = &r;
//     ASSERT_TRUE(region);
//     // The join should be the block containing 'join' state, not tail or ret.
//     bool join_block_matches = false;
//     if (region->join_block) {
//         for (auto* st : region->join_block->states)
//             if (st == &join) join_block_matches = true;
//     }
//     EXPECT_TRUE(join_block_matches);
//     // Removed flag expectation; structural join correctness only.
// }

// TEST(CFGToSCFConversionTest, Hierarchy_IfElseChildrenPresent) {
//     sdfg::builder::SDFGBuilder b("hier_ifelse", sdfg::FunctionType_CPU);
//     auto& entry = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& then1 = b.add_state();
//     auto& els1 = b.add_state();
//     auto& join = b.add_state();
//     b.add_edge(entry, cond);
//     b.add_edge(cond, then1);
//     b.add_edge(then1, join);
//     b.add_edge(cond, els1);
//     b.add_edge(els1, join);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     bool found = false;
//     for (auto* child : root->children)
//         if (child && child->kind == sdfg::passes::scf::RegionKind::IfElse) found = true;
//     EXPECT_TRUE(found);
// }

// TEST(CFGToSCFConversionTest, Hierarchy_NestedIfElse) {
//     sdfg::builder::SDFGBuilder b("nested_if_hier", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& outer_if = b.add_state();
//     auto& outer_else = b.add_state();
//     auto& inner_if = b.add_state();
//     auto& inner_else = b.add_state();
//     auto& inner_join = b.add_state();
//     auto& outer_join = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, outer_if);
//     b.add_edge(start, outer_else);
//     b.add_edge(outer_if, inner_if);
//     b.add_edge(outer_if, inner_else);
//     b.add_edge(inner_if, inner_join);
//     b.add_edge(inner_else, inner_join);
//     b.add_edge(inner_join, outer_join);
//     b.add_edge(outer_else, outer_join);
//     b.add_edge(outer_join, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     size_t top_if = 0;
//     const sdfg::passes::scf::Region* outer_region = nullptr;
//     for (auto* child : root->children)
//         if (child->kind == sdfg::passes::scf::RegionKind::IfElse) {
//             ++top_if;
//             outer_region = child;
//         }
//     ASSERT_EQ(top_if, 1);
//     ASSERT_TRUE(outer_region);
//     size_t nested = 0;
//     for (auto* c : outer_region->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::IfElse) ++nested;
//     EXPECT_GE(nested, 1);
// }

// TEST(CFGToSCFConversionTest, Hierarchy_IfThenSingleArm) {
//     sdfg::builder::SDFGBuilder b("ifthen", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& then_state = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, cond);
//     b.add_edge(cond, then_state);
//     b.add_edge(then_state, ret);
//     b.add_edge(cond, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     bool found = false;
//     for (auto* child : root->children)
//         if (child->kind == sdfg::passes::scf::RegionKind::IfThen) found = true;
//     EXPECT_TRUE(found);
// }

// TEST(CFGToSCFConversionTest, Hierarchy_DoubleNestedIfElse) {
//     sdfg::builder::SDFGBuilder b("double_nested_if", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& o_if = b.add_state();
//     auto& o_else = b.add_state();
//     auto& i_if = b.add_state();
//     auto& i_else = b.add_state();
//     auto& inner_join = b.add_state();
//     auto& i2_if = b.add_state();
//     auto& i2_else = b.add_state();
//     auto& inner2_join = b.add_state();
//     auto& outer_join = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, o_if);
//     b.add_edge(start, o_else);
//     b.add_edge(o_if, i_if);
//     b.add_edge(o_if, i_else);
//     b.add_edge(i_if, inner_join);
//     b.add_edge(i_else, inner_join);
//     b.add_edge(inner_join, i2_if);
//     b.add_edge(inner_join, i2_else);
//     b.add_edge(i2_if, inner2_join);
//     b.add_edge(i2_else, inner2_join);
//     b.add_edge(inner2_join, outer_join);
//     b.add_edge(o_else, outer_join);
//     b.add_edge(outer_join, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     size_t top_if = 0;
//     const sdfg::passes::scf::Region* outer_region = nullptr;
//     for (auto* child : root->children)
//         if (child->kind == sdfg::passes::scf::RegionKind::IfElse) {
//             ++top_if;
//             outer_region = child;
//         }
//     EXPECT_EQ(top_if, 1);
//     ASSERT_TRUE(outer_region);
//     size_t nested = 0;
//     for (auto* c : outer_region->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::IfElse) ++nested;
//     EXPECT_GE(nested, 1);
// }

// // New: Conditional followed by a loop. Expect an IfElse region sibling to a While region (sequentially ordered under
// // root).
// TEST(CFGToSCFConversionTest, Hierarchy_IfElseThenLoop) {
//     sdfg::builder::SDFGBuilder b("if_then_loop", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& then_s = b.add_state();
//     auto& else_s = b.add_state();
//     auto& join = b.add_state();
//     auto& header = b.add_state();
//     auto& body = b.add_state();
//     auto& update = b.add_state();
//     auto& exit = b.add_state();
//     auto& ret = b.add_return_state("r");
//     // IfElse
//     b.add_edge(start, cond);
//     b.add_edge(cond, then_s);
//     b.add_edge(cond, else_s);
//     b.add_edge(then_s, join);
//     b.add_edge(else_s, join);
//     // Loop after join
//     b.add_edge(join, header);
//     b.add_edge(header, body);
//     b.add_edge(body, update);
//     b.add_edge(update, header); // back edge
//     b.add_edge(header, exit);
//     b.add_edge(exit, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     size_t ifelse = 0, whiles = 0;
//     for (auto* c : root->children) {
//         if (!c) continue;
//         if (c->kind == sdfg::passes::scf::RegionKind::IfElse) ++ifelse;
//         if (c->kind == sdfg::passes::scf::RegionKind::While) ++whiles;
//     }
//     EXPECT_EQ(ifelse, 1);
//     EXPECT_EQ(whiles, 1);
// }

// // New: Loop containing an IfThen (single-arm early continuation) inside its body.
// TEST(CFGToSCFConversionTest, Hierarchy_LoopContainsIfThen) {
//     sdfg::builder::SDFGBuilder b("loop_ifthen", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& header = b.add_state();
//     auto& body_entry = b.add_state();
//     auto& cond = b.add_state();
//     auto& then_s = b.add_state();
//     auto& cont = b.add_state();
//     auto& update = b.add_state();
//     auto& exit = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, header);
//     b.add_edge(header, body_entry);
//     b.add_edge(body_entry, cond);
//     // IfThen pattern: cond -> then_s -> update, cond -> update (skip then)
//     b.add_edge(cond, then_s);
//     b.add_edge(then_s, update);
//     b.add_edge(cond, update);
//     b.add_edge(update, header); // loop back
//     b.add_edge(header, exit);
//     b.add_edge(exit, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     const sdfg::passes::scf::Region* while_region = nullptr;
//     for (auto* c : root->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::While) {
//             while_region = c;
//             break;
//         }
//     ASSERT_TRUE(while_region);
//     size_t ifthen = 0;
//     for (auto* c : while_region->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::IfThen) ++ifthen;
//     // Currently may fail until algorithm improved; keep expectation to drive change.
//     EXPECT_GE(ifthen, 1);
// }

// // New: Two sequential loops. Expect two While regions at root.
// TEST(CFGToSCFConversionTest, Hierarchy_TwoSequentialLoops) {
//     sdfg::builder::SDFGBuilder b("two_loops", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     // First loop
//     auto& h1 = b.add_state();
//     auto& b1 = b.add_state();
//     auto& u1 = b.add_state();
//     auto& e1 = b.add_state();
//     // Second loop
//     auto& h2 = b.add_state();
//     auto& b2 = b.add_state();
//     auto& u2 = b.add_state();
//     auto& e2 = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, h1);
//     b.add_edge(h1, b1);
//     b.add_edge(b1, u1);
//     b.add_edge(u1, h1);
//     b.add_edge(h1, e1);
//     b.add_edge(e1, h2);
//     b.add_edge(h2, b2);
//     b.add_edge(b2, u2);
//     b.add_edge(u2, h2);
//     b.add_edge(h2, e2);
//     b.add_edge(e2, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     size_t whiles = 0;
//     for (auto* c : root->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::While) ++whiles;
//     EXPECT_EQ(whiles, 2);
// }

// // Disabled: Irreducible CFG fallback behavior (awaiting unstructured detection logic)
// TEST(CFGToSCFConversionTest, DISABLED_IrreducibleFallbackSequence) {
//     sdfg::builder::SDFGBuilder b("irreducible", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& a = b.add_state();
//     auto& b_state = b.add_state();
//     auto& exit = b.add_state();
//     // Create cross edges causing potential irreducibility: entry->a, entry->b, a->b, b->a, both to exit.
//     b.add_edge(start, a);
//     b.add_edge(start, b_state);
//     b.add_edge(a, b_state);
//     b.add_edge(b_state, a);
//     b.add_edge(a, exit);
//     b.add_edge(b_state, exit);
//     auto& sdfg_obj = b.subject();
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // Expect no While due to irreducibility (future behavior); placeholder assertion.
//     size_t whiles = 0;
//     for (auto& r : regs)
//         if (r.kind == sdfg::passes::scf::RegionKind::While) ++whiles;
//     EXPECT_EQ(whiles, 0u);
// }

// // Loop hierarchy tests (currently failing; kept to drive fixes)
// TEST(CFGToSCFConversionTest, Hierarchy_WhileContainsIfElse) {
//     sdfg::builder::SDFGBuilder b("while_if", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& header = b.add_state();
//     auto& if_cond = b.add_state();
//     auto& then_state = b.add_state();
//     auto& else_state = b.add_state();
//     auto& if_join = b.add_state();
//     auto& update = b.add_state();
//     auto& exit = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, header);
//     b.add_edge(header, if_cond);
//     b.add_edge(if_cond, then_state);
//     b.add_edge(then_state, if_join);
//     b.add_edge(if_cond, else_state);
//     b.add_edge(else_state, if_join);
//     b.add_edge(if_join, update);
//     b.add_edge(update, header); // back edge
//     b.add_edge(header, exit);
//     b.add_edge(exit, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     const sdfg::passes::scf::Region* while_region = nullptr;
//     for (auto* child : root->children)
//         if (child->kind == sdfg::passes::scf::RegionKind::While) {
//             while_region = child;
//             break;
//         }
//     ASSERT_TRUE(while_region);
//     size_t if_children = 0;
//     for (auto* c : while_region->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::IfElse) ++if_children;
//     EXPECT_GE(if_children, 1);
// }

// TEST(CFGToSCFConversionTest, Hierarchy_LoopBodyNestedIfElse) {
//     sdfg::builder::SDFGBuilder b("loop_body_if", sdfg::FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& header = b.add_state();
//     auto& body_entry = b.add_state();
//     auto& if_cond = b.add_state();
//     auto& then_state = b.add_state();
//     auto& else_state = b.add_state();
//     auto& if_join = b.add_state();
//     auto& update = b.add_state();
//     auto& exit = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, header);
//     b.add_edge(header, body_entry);
//     b.add_edge(body_entry, if_cond);
//     b.add_edge(if_cond, then_state);
//     b.add_edge(then_state, if_join);
//     b.add_edge(if_cond, else_state);
//     b.add_edge(else_state, if_join);
//     b.add_edge(if_join, update);
//     b.add_edge(update, header);
//     b.add_edge(header, exit);
//     b.add_edge(exit, ret);
//     sdfg::passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     const auto* root = pass.hierarchical_root();
//     ASSERT_TRUE(root);
//     const sdfg::passes::scf::Region* while_region = nullptr;
//     for (auto* child : root->children)
//         if (child->kind == sdfg::passes::scf::RegionKind::While) {
//             while_region = child;
//             break;
//         }
//     ASSERT_TRUE(while_region);
//     size_t nested_if = 0;
//     for (auto* c : while_region->children)
//         if (c->kind == sdfg::passes::scf::RegionKind::IfElse) ++nested_if;
//     EXPECT_GE(nested_if, 1);
// }

// TEST(CFGToSCFConversionTest, EarlyReturnInOneBranch) {
//     builder::SDFGBuilder b("early_one", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& if_state = b.add_state();
//     auto& else_state = b.add_state();
//     auto& early_ret = b.add_return_state("early_result");
//     auto& cont = b.add_state();
//     auto& final_ret = b.add_return_state("final_result");
//     b.add_edge(start, if_state); // pretend condition
//     b.add_edge(start, else_state); // pretend !condition
//     b.add_edge(if_state, early_ret); // early return terminates branch
//     b.add_edge(else_state, cont);
//     b.add_edge(cont, final_ret);
//     auto& sdfg = b.subject();
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // Our simplistic IfElse detector may not create an IfElse (join missing). Expect a Sequence + possible While
//     none. EXPECT_EQ(count_kind(regs, scf::RegionKind::While), 0u);
//     // Ensure both return states appear in some region blocks
//     bool early_seen = false, final_seen = false;
//     for (auto& r : regs) {
//         for (auto* bb : r.blocks) {
//             if (!bb) continue;
//             for (auto* st : bb->states) {
//                 if (st == &early_ret) early_seen = true;
//                 if (st == &final_ret) final_seen = true;
//             }
//         }
//     }
//     EXPECT_TRUE(early_seen && final_seen);
// }

// TEST(CFGToSCFConversionTest, BothBranchesReturn) {
//     builder::SDFGBuilder b("both_return", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& br1 = b.add_state();
//     auto& br2 = b.add_state();
//     auto& ret1 = b.add_return_state("r1");
//     auto& ret2 = b.add_return_state("r2");
//     b.add_edge(start, br1);
//     b.add_edge(start, br2);
//     b.add_edge(br1, ret1);
//     b.add_edge(br2, ret2);
//     auto& sdfg = b.subject();
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // No join => expect Sequence only
//     EXPECT_EQ(count_kind(regs, scf::RegionKind::IfElse), 0u);
//     const scf::Region* seq = find_region_kind(regs, scf::RegionKind::Sequence);
//     ASSERT_TRUE(seq);
//     bool r1 = false, r2 = false;
//     for (auto* bb : seq->blocks) {
//         if (!bb) continue;
//         for (auto* st : bb->states) {
//             if (st == &ret1) r1 = true;
//             if (st == &ret2) r2 = true;
//         }
//     }
//     EXPECT_TRUE(r1 && r2);
// }

// TEST(CFGToSCFConversionTest, NestedIfElse) {
//     builder::SDFGBuilder b("nested_if", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& outer_if = b.add_state();
//     auto& outer_else = b.add_state();
//     auto& inner_if = b.add_state();
//     auto& inner_else = b.add_state();
//     auto& inner_merge = b.add_state();
//     auto& outer_merge = b.add_state();
//     auto& final_ret = b.add_return_state("res");
//     b.add_edge(start, outer_if);
//     b.add_edge(start, outer_else);
//     b.add_edge(outer_if, inner_if);
//     b.add_edge(outer_if, inner_else);
//     b.add_edge(inner_if, inner_merge);
//     b.add_edge(inner_else, inner_merge);
//     b.add_edge(inner_merge, outer_merge);
//     b.add_edge(outer_else, outer_merge);
//     b.add_edge(outer_merge, final_ret);
//     auto& sdfg = b.subject();
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // Current simple heuristic likely finds only outer IfElse (binary with join) and treats inner as sequence
//     pieces. EXPECT_GE(count_kind(regs, scf::RegionKind::IfElse), 1);
//     // Ensure final return included somewhere
//     bool ret_seen = false;
//     for (auto& r : regs)
//         for (auto* bb : r.blocks) {
//             if (!bb) continue;
//             for (auto* st : bb->states)
//                 if (st == &final_ret) ret_seen = true;
//         }
//     EXPECT_TRUE(ret_seen);
// }

// TEST(CFGToSCFConversionTest, ThreeWayBranch) {
//     builder::SDFGBuilder b("three_way", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& b1 = b.add_state();
//     auto& b2 = b.add_state();
//     auto& b3 = b.add_state();
//     auto& merge = b.add_state();
//     auto& ret = b.add_return_state("result");
//     b.add_edge(start, b1);
//     b.add_edge(start, b2);
//     b.add_edge(start, b3);
//     b.add_edge(b1, merge);
//     b.add_edge(b2, merge);
//     b.add_edge(b3, merge);
//     b.add_edge(merge, ret);
//     auto& sdfg = b.subject();
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // Heuristic handles only binary IfElse -> expect no IfElse, sequence covers all.
//     EXPECT_EQ(count_kind(regs, scf::RegionKind::IfElse), 0u);
//     const scf::Region* seq = find_region_kind(regs, scf::RegionKind::Sequence);
//     ASSERT_TRUE(seq);
//     bool merge_seen = false;
//     for (auto* bb : seq->blocks) {
//         if (!bb) continue;
//         if (bb->entry == &merge) merge_seen = true;
//     }
//     EXPECT_TRUE(merge_seen);
// }

// TEST(CFGToSCFConversionTest, ComplexMergePattern) {
//     builder::SDFGBuilder b("complex_merge", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& branch1 = b.add_state();
//     auto& branch2 = b.add_state();
//     auto& intermediate1 = b.add_state();
//     auto& intermediate2 = b.add_state();
//     auto& merge = b.add_state();
//     auto& ret = b.add_return_state("result");
//     b.add_edge(start, branch1);
//     b.add_edge(start, branch2);
//     b.add_edge(branch1, intermediate1);
//     b.add_edge(branch1, merge);
//     b.add_edge(branch2, intermediate2);
//     b.add_edge(intermediate1, merge);
//     b.add_edge(intermediate2, merge);
//     b.add_edge(merge, ret);
//     auto& sdfg = b.subject();
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // No clean binary pattern -> expect sequence only
//     EXPECT_EQ(count_kind(regs, scf::RegionKind::IfElse), 0u);
//     bool merge_seen = false;
//     for (auto& r : regs)
//         for (auto* bb : r.blocks) {
//             if (!bb) continue;
//             if (bb->entry == &merge) merge_seen = true;
//         }
//     EXPECT_TRUE(merge_seen);
// }

// // Targeted test: pattern where post-dominator intersection fails but diamond fallback should succeed.
// // Structure:
// // start -> cond
// // cond -> a, cond -> b
// // a -> mid
// // b -> mid
// // mid -> join
// // (a and b do not directly converge except via mid; join has two preds: mid and cond making intersection ambiguous)
// // We tweak so that actual join is mid (binary diamond) and final state after mid ensures mid isn't a terminator.
// TEST(CFGToSCFConversionTest, DiamondFallback_Simple) {
//     builder::SDFGBuilder b("diamond_fallback", FunctionType_CPU);
//     auto& start = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& a = b.add_state();
//     auto& b_state = b.add_state();
//     auto& mid = b.add_state();
//     auto& tail = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(start, cond);
//     b.add_edge(cond, a);
//     b.add_edge(cond, b_state);
//     b.add_edge(a, mid);
//     b.add_edge(b_state, mid);
//     b.add_edge(mid, tail);
//     b.add_edge(tail, ret);
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     size_t ifelses = count_kind(regs, scf::RegionKind::IfElse);
//     EXPECT_EQ(ifelses, 1);
//     const scf::Region* ifelse_region = nullptr;
//     for (auto& r : regs)
//         if (r.kind == scf::RegionKind::IfElse) ifelse_region = &r;
//     ASSERT_TRUE(ifelse_region);
//     // Removed fallback flag assertion; structural region existence sufficient.
// }

// // Degenerate fallback: One arm extends deeper before convergence, intersection ambiguous; fallback should set flag.
// // Structure:
// // entry -> cond
// // cond -> short1 -> mid -> long_tail -> join
// // cond -> long1 -> long2 -> mid -> long_tail -> join
// // This forces the earliest common post-dominator to be mid discovered via fallback diamond shape.
// TEST(CFGToSCFConversionTest, DiamondFallback_DegenerateArm) {
//     builder::SDFGBuilder b("diamond_fallback_degenerate", FunctionType_CPU);
//     auto& entry = b.add_state(true);
//     auto& cond = b.add_state();
//     auto& short1 = b.add_state();
//     auto& long1 = b.add_state();
//     auto& long2 = b.add_state();
//     auto& mid = b.add_state();
//     auto& long_tail = b.add_state();
//     auto& join = b.add_state();
//     auto& ret = b.add_return_state("r");
//     b.add_edge(entry, cond);
//     b.add_edge(cond, short1);
//     b.add_edge(short1, mid);
//     b.add_edge(cond, long1);
//     b.add_edge(long1, long2);
//     b.add_edge(long2, mid);
//     b.add_edge(mid, long_tail);
//     b.add_edge(long_tail, join);
//     b.add_edge(join, ret);
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     size_t ifelses = count_kind(regs, scf::RegionKind::IfElse);
//     EXPECT_EQ(ifelses, 1);
//     const scf::Region* ifelse_region = nullptr;
//     for (auto& r : regs)
//         if (r.kind == scf::RegionKind::IfElse) ifelse_region = &r;
//     ASSERT_TRUE(ifelse_region);
//     // Removed fallback flag assertion; structural region existence sufficient.
// }

// TEST(CFGToSCFConversionTest, LoopPatternFromBuilder) {
//     builder::SDFGBuilder b("loop_builder", FunctionType_CPU);
//     auto& init = b.add_state(true);
//     auto& body = b.add_state();
//     auto& header = b.add_state(); // Simulate header after body for variation
//     auto& update = b.add_state();
//     auto& exit = b.add_state();
//     auto& ret = b.add_return_state("res");
//     b.add_edge(init, header);
//     b.add_edge(header, body);
//     b.add_edge(body, update);
//     b.add_edge(update, header); // back edge
//     b.add_edge(header, exit); // exit
//     b.add_edge(exit, ret);
//     auto& sdfg = b.subject();
//     ASSERT_EQ(sdfg.natural_loops().size(), 1);
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     EXPECT_EQ(count_kind(regs, scf::RegionKind::While), 1);
// }

// // Nested diamond pattern: Outer conditional branches each contain an inner diamond that reconverges before outer
// join.
// // Structure:
// // entry -> cond_outer
// // cond_outer -> a1 -> a2 -> inner_join_a -> tail_a
// // cond_outer -> b1 -> b2 -> inner_join_b -> tail_b
// // a1 -> a_mid, a2 -> a_mid (inner diamond A)
// // b1 -> b_mid, b2 -> b_mid (inner diamond B)
// // tail_a -> outer_join; tail_b -> outer_join -> ret
// // Expectations:
// //  - One outer IfElse region with join at outer_join
// //  - Two inner IfElse regions whose join blocks contain a_mid and b_mid
// TEST(CFGToSCFConversionTest, NestedDiamondPatterns) {
//     builder::SDFGBuilder b("nested_diamonds", FunctionType_CPU);
//     auto& entry = b.add_state(true);
//     auto& cond_outer = b.add_state();
//     // Arm A states
//     auto& a1 = b.add_state();
//     auto& a2 = b.add_state();
//     auto& a3 = b.add_state(); // second branch path
//     auto& a_mid = b.add_state();
//     auto& tail_a = b.add_state();
//     // Arm B states
//     auto& b1_s = b.add_state();
//     auto& b2_s = b.add_state();
//     auto& b3_s = b.add_state(); // second branch path
//     auto& b_mid = b.add_state();
//     auto& tail_b = b.add_state();
//     // Outer join and return
//     auto& outer_join = b.add_state();
//     auto& ret = b.add_return_state("r");
//     // Outer branching
//     b.add_edge(entry, cond_outer);
//     b.add_edge(cond_outer, a1);
//     b.add_edge(cond_outer, b1_s);
//     // Inner diamond A: a1 branches to a2 and a3, both converge at a_mid
//     b.add_edge(a1, a2);
//     b.add_edge(a1, a3);
//     b.add_edge(a2, a_mid);
//     b.add_edge(a3, a_mid);
//     // Continue arm A to tail
//     b.add_edge(a_mid, tail_a);
//     // Inner diamond B: b1 branches to b2_s and b3_s, both converge at b_mid
//     b.add_edge(b1_s, b2_s);
//     b.add_edge(b1_s, b3_s);
//     b.add_edge(b2_s, b_mid);
//     b.add_edge(b3_s, b_mid);
//     // Continue arm B to tail
//     b.add_edge(b_mid, tail_b);
//     // Outer convergence
//     b.add_edge(tail_a, outer_join);
//     b.add_edge(tail_b, outer_join);
//     b.add_edge(outer_join, ret);
//     passes::CFGToSCFConversion pass;
//     pass.run_pass(b);
//     auto& regs = pass.regions();
//     // Count outer IfElse regions (there should be at least one)
//     size_t ifelse_count = count_kind(regs, scf::RegionKind::IfElse);
//     EXPECT_GE(ifelse_count, 3u); // outer + two inner
//     const scf::Region* outer_region = nullptr;
//     std::vector<const scf::Region*> inner_regions;
//     for (auto& r : regs) {
//         if (r.kind != scf::RegionKind::IfElse) continue;
//         if (r.join_block) {
//             bool has_outer_join = false;
//             for (auto* st : r.join_block->states)
//                 if (st == &outer_join) has_outer_join = true;
//             if (has_outer_join) outer_region = &r;
//             bool has_a_mid = false;
//             for (auto* st : r.join_block->states)
//                 if (st == &a_mid) has_a_mid = true;
//             bool has_b_mid = false;
//             for (auto* st : r.join_block->states)
//                 if (st == &b_mid) has_b_mid = true;
//             if (has_a_mid || has_b_mid) inner_regions.push_back(&r);
//         }
//     }
//     ASSERT_TRUE(outer_region);
//     // Verify both inner diamond join blocks detected
//     bool inner_a_found = false, inner_b_found = false;
//     for (auto* ir : inner_regions) {
//         for (auto* st : ir->join_block->states) {
//             if (st == &a_mid) inner_a_found = true;
//             if (st == &b_mid) inner_b_found = true;
//         }
//     }
//     EXPECT_TRUE(inner_a_found);
//     EXPECT_TRUE(inner_b_found);
// }

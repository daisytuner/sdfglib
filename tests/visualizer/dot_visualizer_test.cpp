#include "sdfg/visualizer/dot_visualizer.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include <regex>
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

static std::regex dotIdBadChars("[^a-zA-Z0-9_]+");

static std::string escapeDotId(const std::string& id, const std::string& prefix = "") {
    return prefix + std::regex_replace(id, dotIdBadChars, "_");
}

static std::string escapeDotId(size_t id, const std::string& prefix) { return prefix + std::to_string(id); }

TEST(DotVisualizerTest, transpose) {
    builder::StructuredSDFGBuilder builder("transpose", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    // Define loops
    auto bound1 = symbolic::symbol("M");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A, tasklet, "_in", {indvar1, indvar2}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", B, {indvar2, indvar1}, desc_2);

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(loop1.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0; i < M; i = 1 + i\";" << std::endl
        << escapeDotId(loop1.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(loop2.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j = 0; j < N; j = 1 + j\";" << std::endl
        << escapeDotId(loop2.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(block.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(A.element_id(), "n_") << " [penwidth=3.0,label=\"A\"];" << std::endl
        << escapeDotId(tasklet.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(A.element_id(), "n_") << " -> " << escapeDotId(tasklet.element_id(), "n_")
        << " [label=\"   _in = A[i][j]   \"];" << std::endl
        << escapeDotId(B.element_id(), "n_") << " [penwidth=3.0,label=\"B\"];" << std::endl
        << escapeDotId(tasklet.element_id(), "n_") << " -> " << escapeDotId(B.element_id(), "n_")
        << " [label=\"   B[j][i] = _out   \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}


TEST(DotVisualizerTest, syrk) {
    builder::StructuredSDFGBuilder sdfg("sdfg_1", FunctionType_CPU);

    types::Scalar desc_symbols(types::PrimitiveType::UInt64);
    sdfg.add_container("M", desc_symbols, true);
    sdfg.add_container("N", desc_symbols, true);
    sdfg.add_container("i", desc_symbols);
    sdfg.add_container("j_1", desc_symbols);
    sdfg.add_container("j_2", desc_symbols);
    sdfg.add_container("k", desc_symbols);

    types::Scalar desc_element(types::PrimitiveType::Double);
    sdfg.add_container("beta", desc_element, true);
    sdfg.add_container("tmp", desc_element);

    types::Array desc_1d(desc_element, symbolic::symbol("M"));
    types::Pointer desc_2d(desc_1d);

    types::Pointer opaque_desc;
    sdfg.add_container("A", opaque_desc, true);
    sdfg.add_container("C", opaque_desc, true);

    auto& root = sdfg.subject().root();

    auto& loop_i = sdfg.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body_i = loop_i.root();

    auto& loop_j_1 = sdfg.add_for(
        body_i,
        symbolic::symbol("j_1"),
        symbolic::Le(symbolic::symbol("j_1"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1))
    );

    auto& block1 = sdfg.add_block(loop_j_1.root());
    auto& C_in_node_1 = sdfg.add_access(block1, "C");
    auto& C_out_node_1 = sdfg.add_access(block1, "C");
    auto& beta_node = sdfg.add_access(block1, "beta");
    auto& tasklet1 = sdfg.add_tasklet(block1, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    sdfg.add_computational_memlet(
        block1, C_in_node_1, tasklet1, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_1")}, desc_2d
    );
    sdfg.add_computational_memlet(block1, beta_node, tasklet1, "_in2", {});
    sdfg.add_computational_memlet(
        block1, tasklet1, "_out", C_out_node_1, {symbolic::symbol("i"), symbolic::symbol("j_1")}, desc_2d
    );

    auto& loop_k = sdfg.add_for(
        body_i,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& loop_j_2 = sdfg.add_for(
        loop_k.root(),
        symbolic::symbol("j_2"),
        symbolic::Le(symbolic::symbol("j_2"), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1))
    );

    auto& block2 = sdfg.add_block(loop_j_2.root());
    auto& A_node = sdfg.add_access(block2, "A");
    auto& tmp_node = sdfg.add_access(block2, "tmp");
    auto& tasklet2 = sdfg.add_tasklet(block2, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    sdfg.add_computational_memlet(
        block2, A_node, tasklet2, "_in1", {symbolic::symbol("j_2"), symbolic::symbol("k")}, desc_2d
    );
    sdfg.add_computational_memlet(
        block2, A_node, tasklet2, "_in2", {symbolic::symbol("i"), symbolic::symbol("k")}, desc_2d
    );
    sdfg.add_computational_memlet(block2, tasklet2, "_out", tmp_node, {});

    auto& block3 = sdfg.add_block(loop_j_2.root());
    auto& C_in_node_2 = sdfg.add_access(block3, "C");
    auto& C_out_node_2 = sdfg.add_access(block3, "C");
    auto& tmp_node_2 = sdfg.add_access(block3, "tmp");
    auto& tasklet3 = sdfg.add_tasklet(block3, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    sdfg.add_computational_memlet(
        block3, C_in_node_2, tasklet3, "_in1", {symbolic::symbol("i"), symbolic::symbol("j_2")}, desc_2d
    );
    sdfg.add_computational_memlet(block3, tmp_node_2, tasklet3, "_in2", {});
    sdfg.add_computational_memlet(
        block3, tasklet3, "_out", C_out_node_2, {symbolic::symbol("i"), symbolic::symbol("j_2")}, desc_2d
    );

    auto sdfg2 = sdfg.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(loop_i.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0; i < N; i = 1 + i\";" << std::endl
        << escapeDotId(loop_i.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(loop_j_1.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(16); // TODO matching the exact NODE ID is HORRIBLE, especially because those UUIDs are not valid
                       // graphviz identifiers
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_1 = 0; j_1 <= i; j_1 = 1 + j_1\";"
        << std::endl
        << escapeDotId(loop_j_1.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(block1.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(beta_node.element_id(), "n_") << " [penwidth=3.0,label=\"beta\"];" << std::endl
        << escapeDotId(C_in_node_1.element_id(), "n_") << " [penwidth=3.0,label=\"C\"];" << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << escapeDotId(C_in_node_1.element_id(), "n_") << " -> " << escapeDotId(tasklet1.element_id(), "n_")
        << " [label=\"   _in1 = C[i][j_1]   \"];" << std::endl
        << escapeDotId(beta_node.element_id(), "n_") << " -> " << escapeDotId(tasklet1.element_id(), "n_")
        << " [label=\"   _in2 = beta   \"];" << std::endl
        << escapeDotId(C_out_node_1.element_id(), "n_") << " [penwidth=3.0,label=\"C\"];" << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " -> " << escapeDotId(C_out_node_1.element_id(), "n_")
        << " [label=\"   C[i][j_1] = _out   \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(loop_k.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: k = 0; k < M; k = 1 + k\";" << std::endl
        << escapeDotId(loop_k.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(loop_j_2.element_id(), "for_") << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_2 = 0; j_2 <= i; j_2 = 1 + j_2\";"
        << std::endl
        << escapeDotId(loop_j_2.element_id(), "for_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(block2.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(A_node.element_id(), "n_") << " [penwidth=3.0,label=\"A\"];" << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << escapeDotId(A_node.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in1 = A[j_2][k]   \"];" << std::endl
        << escapeDotId(A_node.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in2 = A[i][k]   \"];" << std::endl
        << escapeDotId(tmp_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];"
        << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " -> " << escapeDotId(tmp_node.element_id(), "n_")
        << " [label=\"   tmp = _out   \"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(block3.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(tmp_node_2.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];"
        << std::endl
        << escapeDotId(C_in_node_2.element_id(), "n_") << " [penwidth=3.0,label=\"C\"];" << std::endl
        << escapeDotId(tasklet3.element_id(), "n_") << " [shape=octagon,label=\"_out = _in1 + _in2\"];" << std::endl
        << escapeDotId(C_in_node_2.element_id(), "n_") << " -> " << escapeDotId(tasklet3.element_id(), "n_")
        << " [label=\"   _in1 = C[i][j_2]   \"];" << std::endl
        << escapeDotId(tmp_node_2.element_id(), "n_") << " -> " << escapeDotId(tasklet3.element_id(), "n_")
        << " [label=\"   _in2 = tmp   \"];" << std::endl
        << escapeDotId(C_out_node_2.element_id(), "n_") << " [penwidth=3.0,label=\"C\"];" << std::endl
        << escapeDotId(tasklet3.element_id(), "n_") << " -> " << escapeDotId(C_out_node_2.element_id(), "n_")
        << " [label=\"   C[i][j_2] = _out   \"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << escapeDotId(A_node.element_id(), "n_") << " -> " << escapeDotId(tmp_node_2.element_id(), "n_")
        << " [ltail=\"cluster_" << escapeDotId(block2.element_id(), "block_") << "\",lhead=\"cluster_"
        << escapeDotId(block3.element_id(), "block_") << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl
        << escapeDotId(loop_j_1.element_id(), "for_") << " -> " << escapeDotId(loop_k.element_id(), "for_")
        << " [ltail=\"cluster_" << escapeDotId(loop_j_1.element_id(), "for_") << "\",lhead=\"cluster_"
        << escapeDotId(loop_k.element_id(), "for_") << "\",minlen=3];" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}


TEST(DotVisualizerTest, multi_tasklet_block) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");

    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A1, tasklet1, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block, tasklet1, "_out", A2, {symbolic::integer(0)});

    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A2, tasklet2, "_in", {symbolic::integer(0)});
    builder.add_computational_memlet(block, tasklet2, "_out", A3, {symbolic::integer(0)});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(A1.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(A1.element_id(), "n_") << " -> " << escapeDotId(tasklet1.element_id(), "n_")
        << " [label=\"   _in = A[0]   \"];" << std::endl
        << escapeDotId(A2.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " -> " << escapeDotId(A2.element_id(), "n_")
        << " [label=\"   A[0] = _out   \"];" << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(A2.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in = A[0]   \"];" << std::endl
        << escapeDotId(A3.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " -> " << escapeDotId(A3.element_id(), "n_")
        << " [label=\"   A[0] = _out   \"];" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, test_if_else) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& true_case = builder.add_case(if_else, symbolic::Le(symbolic::symbol("A"), symbolic::integer(0)));
    auto& false_case = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("A"), symbolic::integer(0)));

    auto& block = builder.add_block(true_case);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    auto& block2 = builder.add_block(false_case);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {});
    builder.add_computational_memlet(block2, input_node2, tasklet2, "_in", {});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << escapeDotId(if_else.element_id(), "if_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_0 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"A <= 0\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(input_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];"
        << std::endl
        << escapeDotId(tasklet.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(input_node.element_id(), "n_") << " -> " << escapeDotId(tasklet.element_id(), "n_")
        << " [label=\"   _in = A   \"];" << std::endl
        << escapeDotId(output_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];"
        << std::endl
        << escapeDotId(tasklet.element_id(), "n_") << " -> " << escapeDotId(output_node.element_id(), "n_")
        << " [label=\"   B = _out   \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_1 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"0 < A\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block2.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(input_node2.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];"
        << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(input_node2.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in = B   \"];" << std::endl
        << escapeDotId(output_node2.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];"
        << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " -> " << escapeDotId(output_node2.element_id(), "n_")
        << " [label=\"   A = _out   \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}


TEST(DotVisualizerTest, test_while) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2);

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(loop.element_id(), "while_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
        << escapeDotId(loop.element_id(), "while_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << escapeDotId(if_else.element_id(), "if_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_0 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"i < 10\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block1.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(block1.element_id(), "block_") << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << escapeDotId(cont1.element_id(), "cont_") << " [shape=cds,label=\" continue  \"];" << std::endl
        << escapeDotId(block1.element_id(), "block_") << " -> " << escapeDotId(cont1.element_id(), "cont_")
        << " [ltail=\"cluster_" << escapeDotId(block1.element_id(), "block_") << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_1 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"10 <= i\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block2.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(block2.element_id(), "block_") << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << escapeDotId(break1.element_id(), "break_") << " [shape=cds,label=\" break  \"];" << std::endl
        << escapeDotId(block2.element_id(), "block_") << " -> " << escapeDotId(break1.element_id(), "break_")
        << " [ltail=\"cluster_" << escapeDotId(block2.element_id(), "block_") << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, test_return) {
    builder::StructuredSDFGBuilder builder("test_return", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Array desc(base_desc, symbolic::symbol("N"));
    builder.add_container("A", desc, true);

    auto& block1 = builder.add_block(root);
    auto& output1 = builder.add_access(block1, "i");
    auto& zero_node = builder.add_constant(block1, "0", sym_desc);
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, zero_node, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", output1, {});

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("i"), symbolic::symbol("N")));
    auto& case2 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")));

    auto& return_node = builder.add_return(case1, "");

    auto& block2 = builder.add_block(case2);
    auto& input2 = builder.add_access(block2, "A");
    auto& output2 = builder.add_access(block2, "A");
    auto& two_node = builder.add_constant(block2, "2.0", base_desc);
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, input2, tasklet2, "_in1", {symbolic::symbol("i")});
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", output2, {symbolic::symbol("i")});

    auto& block3 = builder.add_block(case2);
    auto& input3 = builder.add_access(block3, "i");
    auto& output3 = builder.add_access(block3, "i");
    auto& one_node = builder.add_constant(block3, "1", sym_desc);
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block3, input3, tasklet3, "_in1", {});
    builder.add_computational_memlet(block3, one_node, tasklet3, "_in2", {});
    builder.add_computational_memlet(block3, tasklet3, "_out", output3, {});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block1.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(zero_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"0\"];"
        << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << escapeDotId(zero_node.element_id(), "n_") << " -> " << escapeDotId(tasklet1.element_id(), "n_")
        << " [label=\"   _in = 0   \"];" << std::endl
        << escapeDotId(output1.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];"
        << std::endl
        << escapeDotId(tasklet1.element_id(), "n_") << " -> " << escapeDotId(output1.element_id(), "n_")
        << " [label=\"   i = _out   \"];" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(loop.element_id(), "while_") << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
        << escapeDotId(loop.element_id(), "while_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << escapeDotId(if_else.element_id(), "if_") << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_0 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"N <= i\";" << std::endl
        << escapeDotId(return_node.element_id(), "return_") << " [shape=cds,label=\" return  \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(if_else.element_id(), "if_") << "_1 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"i < N\";" << std::endl
        << "subgraph cluster_" << escapeDotId(block2.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(two_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"2.0\"];"
        << std::endl
        << escapeDotId(input2.element_id(), "n_") << " [penwidth=3.0,label=\"A\"];" << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << escapeDotId(input2.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in1 = A[i]   \"];" << std::endl
        << escapeDotId(two_node.element_id(), "n_") << " -> " << escapeDotId(tasklet2.element_id(), "n_")
        << " [label=\"   _in2 = 2.0   \"];" << std::endl
        << escapeDotId(output2.element_id(), "n_") << " [penwidth=3.0,label=\"A\"];" << std::endl
        << escapeDotId(tasklet2.element_id(), "n_") << " -> " << escapeDotId(output2.element_id(), "n_")
        << " [label=\"   A[i] = _out   \"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl << "subgraph cluster_" << escapeDotId(block3.element_id(), "block_") << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << escapeDotId(one_node.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"1\"];"
        << std::endl
        << escapeDotId(input3.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];" << std::endl
        << escapeDotId(tasklet3.element_id(), "n_") << " [shape=octagon,label=\"_out = _in1 + _in2\"];" << std::endl
        << escapeDotId(input3.element_id(), "n_") << " -> " << escapeDotId(tasklet3.element_id(), "n_")
        << " [label=\"   _in1 = i   \"];" << std::endl
        << escapeDotId(one_node.element_id(), "n_") << " -> " << escapeDotId(tasklet3.element_id(), "n_")
        << " [label=\"   _in2 = 1   \"];" << std::endl
        << escapeDotId(output3.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];"
        << std::endl
        << escapeDotId(tasklet3.element_id(), "n_") << " -> " << escapeDotId(output3.element_id(), "n_")
        << " [label=\"   i = _out   \"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << escapeDotId(two_node.element_id(), "n_") << " -> " << escapeDotId(one_node.element_id(), "n_")
        << " [ltail=\"cluster_" << escapeDotId(block2.element_id(), "block_") << "\",lhead=\"cluster_"
        << escapeDotId(block3.element_id(), "block_") << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl
        << escapeDotId(zero_node.element_id(), "n_") << " -> " << escapeDotId(loop.element_id(), "while_")
        << " [ltail=\"cluster_" << escapeDotId(block1.element_id(), "block_") << "\",lhead=\"cluster_"
        << escapeDotId(loop.element_id(), "while_") << "\",minlen=3];" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}


TEST(DotVisualizerTest, test_handleTasklet) {
    const std::vector<std::pair<const data_flow::TaskletCode, const std::string>> codes = {
        {data_flow::TaskletCode::assign, "="},
        {data_flow::TaskletCode::fp_neg, "-"},
        {data_flow::TaskletCode::add, "+"},
        {data_flow::TaskletCode::sub, "-"},
        {data_flow::TaskletCode::mul, "*"},
        {data_flow::TaskletCode::div, "/"},
        {data_flow::TaskletCode::fp_fma, "fma"},
        {data_flow::TaskletCode::mod, "%"},
        {data_flow::TaskletCode::max, "max"},
        {data_flow::TaskletCode::min, "min"},
        {data_flow::TaskletCode::minnum, "minnum"},
        {data_flow::TaskletCode::maxnum, "maxnum"},
        {data_flow::TaskletCode::minimum, "minimum"},
        {data_flow::TaskletCode::maximum, "maximum"},
        {data_flow::TaskletCode::trunc, "trunc"},
        {data_flow::TaskletCode::logical_and, "&&"},
        {data_flow::TaskletCode::logical_or, "||"},
        {data_flow::TaskletCode::bitwise_and, "&"},
        {data_flow::TaskletCode::bitwise_or, "|"},
        {data_flow::TaskletCode::bitwise_xor, "^"},
        {data_flow::TaskletCode::bitwise_not, "~"},
        {data_flow::TaskletCode::shift_left, "<<"},
        {data_flow::TaskletCode::shift_right, ">>"},
        {data_flow::TaskletCode::olt, "<"},
        {data_flow::TaskletCode::ole, "<="},
        {data_flow::TaskletCode::oeq, "=="},
        {data_flow::TaskletCode::one, "!="},
        {data_flow::TaskletCode::oge, ">="},
        {data_flow::TaskletCode::ogt, ">"},
        {data_flow::TaskletCode::ord, "=="},
        {data_flow::TaskletCode::ult, "<"},
        {data_flow::TaskletCode::ule, "<="},
        {data_flow::TaskletCode::ueq, "=="},
        {data_flow::TaskletCode::une, "!="},
        {data_flow::TaskletCode::uge, ">="},
        {data_flow::TaskletCode::ugt, ">"},
        {data_flow::TaskletCode::uno, "!="},
        {data_flow::TaskletCode::abs, "abs"},
        {data_flow::TaskletCode::acos, "acos"},
        {data_flow::TaskletCode::acosf, "acosf"},
        {data_flow::TaskletCode::acosl, "acosl"},
        {data_flow::TaskletCode::acosh, "acosh"},
        {data_flow::TaskletCode::acoshf, "acoshf"},
        {data_flow::TaskletCode::acoshl, "acoshl"},
        {data_flow::TaskletCode::asin, "asin"},
        {data_flow::TaskletCode::asinf, "asinf"},
        {data_flow::TaskletCode::asinl, "asinl"},
        {data_flow::TaskletCode::asinh, "asinh"},
        {data_flow::TaskletCode::asinhf, "asinhf"},
        {data_flow::TaskletCode::asinhl, "asinhl"},
        {data_flow::TaskletCode::atan, "atan"},
        {data_flow::TaskletCode::atanf, "atanf"},
        {data_flow::TaskletCode::atanl, "atanl"},
        {data_flow::TaskletCode::atan2, "atan2"},
        {data_flow::TaskletCode::atan2f, "atan2f"},
        {data_flow::TaskletCode::atan2l, "atan2l"},
        {data_flow::TaskletCode::atanh, "atanh"},
        {data_flow::TaskletCode::atanhf, "atanhf"},
        {data_flow::TaskletCode::atanhl, "atanhl"},
        {data_flow::TaskletCode::cabs, "cabs"},
        {data_flow::TaskletCode::cabsf, "cabsf"},
        {data_flow::TaskletCode::cabsl, "cabsl"},
        {data_flow::TaskletCode::ceil, "ceil"},
        {data_flow::TaskletCode::ceilf, "ceilf"},
        {data_flow::TaskletCode::ceill, "ceill"},
        {data_flow::TaskletCode::copysign, "copysign"},
        {data_flow::TaskletCode::copysignf, "copysignf"},
        {data_flow::TaskletCode::copysignl, "copysignl"},
        {data_flow::TaskletCode::cos, "cos"},
        {data_flow::TaskletCode::cosf, "cosf"},
        {data_flow::TaskletCode::cosl, "cosl"},
        {data_flow::TaskletCode::cosh, "cosh"},
        {data_flow::TaskletCode::coshf, "coshf"},
        {data_flow::TaskletCode::coshl, "coshl"},
        {data_flow::TaskletCode::cbrt, "cbrt"},
        {data_flow::TaskletCode::cbrtf, "cbrtf"},
        {data_flow::TaskletCode::cbrtl, "cbrtl"},
        {data_flow::TaskletCode::exp10, "exp10"},
        {data_flow::TaskletCode::exp10f, "exp10f"},
        {data_flow::TaskletCode::exp10l, "exp10l"},
        {data_flow::TaskletCode::exp2, "exp2"},
        {data_flow::TaskletCode::exp2f, "exp2f"},
        {data_flow::TaskletCode::exp2l, "exp2l"},
        {data_flow::TaskletCode::exp, "exp"},
        {data_flow::TaskletCode::expf, "expf"},
        {data_flow::TaskletCode::expl, "expl"},
        {data_flow::TaskletCode::expm1, "expm1"},
        {data_flow::TaskletCode::expm1f, "expm1f"},
        {data_flow::TaskletCode::expm1l, "expm1l"},
        {data_flow::TaskletCode::fabs, "fabs"},
        {data_flow::TaskletCode::fabsf, "fabsf"},
        {data_flow::TaskletCode::fabsl, "fabsl"},
        {data_flow::TaskletCode::floor, "floor"},
        {data_flow::TaskletCode::floorf, "floorf"},
        {data_flow::TaskletCode::floorl, "floorl"},
        {data_flow::TaskletCode::fls, "fls"},
        {data_flow::TaskletCode::flsl, "flsl"},
        {data_flow::TaskletCode::fmax, "fmax"},
        {data_flow::TaskletCode::fmaxf, "fmaxf"},
        {data_flow::TaskletCode::fmaxl, "fmaxl"},
        {data_flow::TaskletCode::fmin, "fmin"},
        {data_flow::TaskletCode::fminf, "fminf"},
        {data_flow::TaskletCode::fminl, "fminl"},
        {data_flow::TaskletCode::fmod, "fmod"},
        {data_flow::TaskletCode::fmodf, "fmodf"},
        {data_flow::TaskletCode::fmodl, "fmodl"},
        {data_flow::TaskletCode::frexp, "frexp"},
        {data_flow::TaskletCode::frexpf, "frexpf"},
        {data_flow::TaskletCode::frexpl, "frexpl"},
        {data_flow::TaskletCode::labs, "labs"},
        {data_flow::TaskletCode::ldexp, "ldexp"},
        {data_flow::TaskletCode::ldexpf, "ldexpf"},
        {data_flow::TaskletCode::ldexpl, "ldexpl"},
        {data_flow::TaskletCode::log10, "log10"},
        {data_flow::TaskletCode::log10f, "log10f"},
        {data_flow::TaskletCode::log10l, "log10l"},
        {data_flow::TaskletCode::log2, "log2"},
        {data_flow::TaskletCode::log2f, "log2f"},
        {data_flow::TaskletCode::log2l, "log2l"},
        {data_flow::TaskletCode::log, "log"},
        {data_flow::TaskletCode::logf, "logf"},
        {data_flow::TaskletCode::logl, "logl"},
        {data_flow::TaskletCode::logb, "logb"},
        {data_flow::TaskletCode::logbf, "logbf"},
        {data_flow::TaskletCode::logbl, "logbl"},
        {data_flow::TaskletCode::log1p, "log1p"},
        {data_flow::TaskletCode::log1pf, "log1pf"},
        {data_flow::TaskletCode::log1pl, "log1pl"},
        {data_flow::TaskletCode::nearbyint, "nearbyint"},
        {data_flow::TaskletCode::nearbyintf, "nearbyintf"},
        {data_flow::TaskletCode::nearbyintl, "nearbyintl"},
        {data_flow::TaskletCode::pow, "pow"},
        {data_flow::TaskletCode::powf, "powf"},
        {data_flow::TaskletCode::powl, "powl"},
        {data_flow::TaskletCode::rint, "rint"},
        {data_flow::TaskletCode::rintf, "rintf"},
        {data_flow::TaskletCode::rintl, "rintl"},
        {data_flow::TaskletCode::round, "round"},
        {data_flow::TaskletCode::roundf, "roundf"},
        {data_flow::TaskletCode::roundl, "roundl"},
        {data_flow::TaskletCode::roundeven, "roundeven"},
        {data_flow::TaskletCode::roundevenf, "roundevenf"},
        {data_flow::TaskletCode::roundevenl, "roundevenl"},
        {data_flow::TaskletCode::sin, "sin"},
        {data_flow::TaskletCode::sinf, "sinf"},
        {data_flow::TaskletCode::sinl, "sinl"},
        {data_flow::TaskletCode::sinh, "sinh"},
        {data_flow::TaskletCode::sinhf, "sinhf"},
        {data_flow::TaskletCode::sinhl, "sinhl"},
        {data_flow::TaskletCode::sqrt, "sqrt"},
        {data_flow::TaskletCode::sqrtf, "sqrtf"},
        {data_flow::TaskletCode::sqrtl, "sqrtl"},
        {data_flow::TaskletCode::rsqrt, "rsqrt"},
        {data_flow::TaskletCode::rsqrtf, "rsqrtf"},
        {data_flow::TaskletCode::rsqrtl, "rsqrtl"},
        {data_flow::TaskletCode::tan, "tan"},
        {data_flow::TaskletCode::tanf, "tanf"},
        {data_flow::TaskletCode::tanl, "tanl"},
        {data_flow::TaskletCode::tanh, "tanh"},
        {data_flow::TaskletCode::tanhf, "tanhf"},
        {data_flow::TaskletCode::tanhl, "tanhl"}
    };
    for (const std::pair<const data_flow::TaskletCode, const std::string> code : codes) {
        const size_t arity = data_flow::arity(code.first);
        builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);

        auto& sdfg = builder.subject();
        auto& root = sdfg.root();

        types::Scalar desc{types::PrimitiveType::Int32};
        builder.add_container("x", desc);

        auto& block = builder.add_block(root);
        auto& output = builder.add_access(block, "x");
        std::vector<std::string> inputs;
        for (size_t i = 0; i < arity; ++i) inputs.push_back(std::to_string(i));
        auto& tasklet = builder.add_tasklet(block, code.first, "_out", inputs);
        builder.add_computational_memlet(block, tasklet, "_out", output, {});

        codegen::PrettyPrinter exp;
        exp << "digraph " << builder.subject().name() << " {" << std::endl;
        exp.setIndent(4);
        exp << "graph [compound=true];" << std::endl
            << "subgraph cluster_" << builder.subject().name() << " {" << std::endl;
        exp.setIndent(8);
        exp << "node [style=filled,fillcolor=white];" << std::endl
            << "style=filled;color=lightblue;label=\"\";" << std::endl
            << "subgraph cluster_" << escapeDotId(block.element_id(), "block_") << " {" << std::endl;
        exp.setIndent(12);
        exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
            << escapeDotId(tasklet.element_id(), "n_") << " [shape=octagon,label=\"";
        if (code.first == data_flow::TaskletCode::assign) {
            exp << "_out = 0";
        } else if (code.first == data_flow::TaskletCode::fma) {
            exp << "_out = 0 * 1 + 2";
        } else if (data_flow::is_infix(code.first) && arity == 1) {
            exp << "_out = " << code.second << " 0";
        } else if (data_flow::is_infix(code.first) && arity == 2) {
            exp << "_out = 0 " << code.second << " 1";
        } else {
            exp << "_out = " << code.second << "(";
            for (size_t i = 0; i < arity; ++i) {
                if (i > 0) exp << ", ";
                exp << std::to_string(i);
            }
            exp << ")";
        }
        exp << "\"];" << std::endl
            << escapeDotId(output.element_id(), "n_") << " [penwidth=3.0,style=\"dashed,filled\",label=\"x\"];"
            << std::endl
            << escapeDotId(tasklet.element_id(), "n_") << " -> " << escapeDotId(output.element_id(), "n_")
            << " [label=\"   x = _out   \"];" << std::endl;
        exp.setIndent(8);
        exp << "}" << std::endl;
        exp.setIndent(4);
        exp << "}" << std::endl;
        exp.setIndent(0);
        exp << "}" << std::endl;

        visualizer::DotVisualizer dot(builder.subject());
        dot.visualize();
        EXPECT_EQ(dot.getStream().str(), exp.str());
    }
}

TEST(DotVisualizerTest, visualizeSubset_does_not_fail_on_incomplete_opaque_ptr) {
    builder::StructuredSDFGBuilder builder("dummy", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array arr_of_float(base_desc, symbolic::symbol("M"));
    types::Pointer ptr_of_arr_of_float(arr_of_float);

    types::Pointer ptr_of_float(base_desc);

    types::Pointer opaque_desc;

    {
        visualizer::DotVisualizer dot(sdfg);

        dot.visualizeSubset(sdfg, {symbolic::zero()}, &opaque_desc);

        EXPECT_EQ(dot.getStream().str(), "[0]");
    }

    {
        visualizer::DotVisualizer dot(sdfg);

        dot.visualizeSubset(sdfg, {symbolic::one()}, &opaque_desc);

        EXPECT_EQ(dot.getStream().str(), "[1]#illgl");
    }

    {
        visualizer::DotVisualizer dot(sdfg);

        dot.visualizeSubset(sdfg, {symbolic::zero(), symbolic::one()}, &opaque_desc);

        EXPECT_EQ(dot.getStream().str(), "[0](rogue)[1]");
    }
}

#include "sdfg/visualizer/dot_visualizer.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

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
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);

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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", base_desc},
                                        {{"_in", base_desc}});
    builder.add_memlet(block, A, "void", tasklet, "_in", {indvar1, indvar2});
    builder.add_memlet(block, tasklet, "_out", B, "void", {indvar2, indvar1});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop1.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0:(M-1)\";"
        << std::endl
        << loop1.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop2.element_id() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j = 0:(N-1)\";"
        << std::endl
        << loop2.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block.element_id() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A.element_id() << " [penwidth=3.0,label=\"A\"];" << std::endl
        << tasklet.element_id() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << A.element_id() << " -> " << tasklet.element_id() << " [label=\"   _in = A[i][j]   \"];"
        << std::endl
        << tasklet.element_id() << " -> " << B.element_id() << " [label=\"   B[j][i] = _out   \"];"
        << std::endl
        << B.element_id() << " [penwidth=3.0,label=\"B\"];" << std::endl;
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

/*
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

    types::Pointer desc_1d(desc_element);
    types::Pointer desc_2d(*desc_1d.clone());
    sdfg.add_container("A", desc_2d, true);
    sdfg.add_container("C", desc_2d, true);

    auto& root = sdfg.subject().root();

    auto& loop_i = sdfg.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));
    auto& body_i = loop_i.root();

    auto& loop_j_1 = sdfg.add_for(body_i, symbolic::symbol("j_1"),
                                  symbolic::Le(symbolic::symbol("j_1"), symbolic::symbol("i")),
                                  symbolic::integer(0),
                                  symbolic::add(symbolic::symbol("j_1"), symbolic::integer(1)));

    auto& block1 = sdfg.add_block(loop_j_1.root());
    auto& C_in_node_1 = sdfg.add_access(block1, "C");
    auto& C_out_node_1 = sdfg.add_access(block1, "C");
    auto& beta_node = sdfg.add_access(block1, "beta");
    auto& tasklet1 = sdfg.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", desc_element},
                                      {{"_in1", desc_element}, {"_in2", desc_element}});
    sdfg.add_memlet(block1, C_in_node_1, "void", tasklet1, "_in1",
                    {symbolic::symbol("i"), symbolic::symbol("j_1")});
    sdfg.add_memlet(block1, beta_node, "void", tasklet1, "_in2", {});
    sdfg.add_memlet(block1, tasklet1, "_out", C_out_node_1, "void",
                    {symbolic::symbol("i"), symbolic::symbol("j_1")});

    auto& loop_k = sdfg.add_for(
        body_i, symbolic::symbol("k"), symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0), symbolic::add(symbolic::symbol("k"), symbolic::integer(1)));

    auto& loop_j_2 = sdfg.add_for(loop_k.root(), symbolic::symbol("j_2"),
                                  symbolic::Le(symbolic::symbol("j_2"), symbolic::symbol("i")),
                                  symbolic::integer(0),
                                  symbolic::add(symbolic::symbol("j_2"), symbolic::integer(1)));

    auto& block2 = sdfg.add_block(loop_j_2.root());
    auto& A_node = sdfg.add_access(block2, "A");
    auto& tmp_node = sdfg.add_access(block2, "tmp");
    auto& tasklet2 = sdfg.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", desc_element},
                                      {{"_in1", desc_element}, {"_in2", desc_element}});
    sdfg.add_memlet(block2, A_node, "void", tasklet2, "_in1",
                    {symbolic::symbol("j_2"), symbolic::symbol("k")});
    sdfg.add_memlet(block2, A_node, "void", tasklet2, "_in2",
                    {symbolic::symbol("i"), symbolic::symbol("k")});
    sdfg.add_memlet(block2, tasklet2, "_out", tmp_node, "void", {});

    auto& block3 = sdfg.add_block(loop_j_2.root());
    auto& C_in_node_2 = sdfg.add_access(block3, "C");
    auto& C_out_node_2 = sdfg.add_access(block3, "C");
    auto& tmp_node_2 = sdfg.add_access(block3, "tmp");
    auto& tasklet3 = sdfg.add_tasklet(block3, data_flow::TaskletCode::add, {"_out", desc_element},
                                      {{"_in1", desc_element}, {"_in2", desc_element}});
    sdfg.add_memlet(block3, C_in_node_2, "void", tasklet3, "_in1",
                    {symbolic::symbol("i"), symbolic::symbol("j_2")});
    sdfg.add_memlet(block3, tmp_node_2, "void", tasklet3, "_in2", {});
    sdfg.add_memlet(block3, tasklet3, "_out", C_out_node_2, "void",
                    {symbolic::symbol("i"), symbolic::symbol("j_2")});

    auto sdfg2 = sdfg.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop_i.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0:(N-1)\";"
        << std::endl
        << loop_i.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop_j_1.element_id() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_1 = 0:i\";"
        << std::endl
        << loop_j_1.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block1.element_id() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << beta_node.element_id() << " [penwidth=3.0,label=\"beta\"];" << std::endl
        << C_in_node_1.element_id() << " [penwidth=3.0,label=\"C\"];" << std::endl
        << tasklet1.element_id() << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << C_in_node_1.element_id() << " -> " << tasklet1.element_id()
        << " [label=\"   _in1 = C[i][j_1]   \"];" << std::endl
        << beta_node.element_id() << " -> " << tasklet1.element_id()
        << " [label=\"   _in2 = beta   \"];" << std::endl
        << tasklet1.element_id() << " -> " << C_out_node_1.element_id()
        << " [label=\"   C[i][j_1] = _out   \"];" << std::endl
        << C_out_node_1.element_id() << " [penwidth=3.0,label=\"C\"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << loop_k.element_id() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: k = 0:(M-1)\";"
        << std::endl
        << loop_k.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop_j_2.element_id() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_2 = 0:i\";"
        << std::endl
        << loop_j_2.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block2.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A_node.element_id() << " [penwidth=3.0,label=\"A\"];" << std::endl
        << tasklet2.element_id() << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << A_node.element_id() << " -> " << tasklet2.element_id()
        << " [label=\"   _in1 = A[j_2][k]   \"];" << std::endl
        << A_node.element_id() << " -> " << tasklet2.element_id()
        << " [label=\"   _in2 = A[i][k]   \"];" << std::endl
        << tasklet2.element_id() << " -> " << tmp_node.element_id()
        << " [label=\"   tmp = _out   \"];" << std::endl
        << tmp_node.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];"
        << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl << "subgraph cluster_" << block3.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << tmp_node_2.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];"
        << std::endl
        << C_in_node_2.element_id() << " [penwidth=3.0,label=\"C\"];" << std::endl
        << tasklet3.element_id() << " [shape=octagon,label=\"_out = _in1 + _in2\"];" << std::endl
        << C_in_node_2.element_id() << " -> " << tasklet3.element_id()
        << " [label=\"   _in1 = C[i][j_2]   \"];" << std::endl
        << tmp_node_2.element_id() << " -> " << tasklet3.element_id()
        << " [label=\"   _in2 = tmp   \"];" << std::endl
        << tasklet3.element_id() << " -> " << C_out_node_2.element_id()
        << " [label=\"   C[i][j_2] = _out   \"];" << std::endl
        << C_out_node_2.element_id() << " [penwidth=3.0,label=\"C\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << tasklet2.element_id() << " -> " << tasklet3.element_id() << " [ltail=\"cluster_"
        << block2.element_id() << "\",lhead=\"cluster_" << block3.element_id() << "\",minlen=3];"
        << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl
        << loop_j_1.element_id() << " -> " << loop_k.element_id() << " [ltail=\"cluster_"
        << loop_j_1.element_id() << "\",lhead=\"cluster_" << loop_k.element_id() << "\",minlen=3];"
        << std::endl;
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
*/

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

    auto& tasklet1 =
        builder.add_tasklet(block, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block, A1, "void", tasklet1, "_in", {symbolic::integer(0)});
    builder.add_memlet(block, tasklet1, "_out", A2, "void", {symbolic::integer(0)});

    auto& tasklet2 =
        builder.add_tasklet(block, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block, A2, "void", tasklet2, "_in", {symbolic::integer(0)});
    builder.add_memlet(block, tasklet2, "_out", A3, "void", {symbolic::integer(0)});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << block.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A1.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << tasklet1.element_id() << " [shape=octagon,label=\"_out = 2 * _in + 1\"];" << std::endl
        << A1.element_id() << " -> " << tasklet1.element_id() << " [label=\"   _in = A[0]   \"];"
        << std::endl
        << tasklet1.element_id() << " -> " << A2.element_id() << " [label=\"   A[0] = _out   \"];"
        << std::endl
        << A2.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << tasklet2.element_id() << " [shape=octagon,label=\"_out = 2 * _in + 1\"];" << std::endl
        << A2.element_id() << " -> " << tasklet2.element_id() << " [label=\"   _in = A[0]   \"];"
        << std::endl
        << tasklet2.element_id() << " -> " << A3.element_id() << " [label=\"   A[0] = _out   \"];"
        << std::endl
        << A3.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl;
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
    auto& true_case =
        builder.add_case(if_else, symbolic::Le(symbolic::symbol("A"), symbolic::integer(0)));
    auto& false_case =
        builder.add_case(if_else, symbolic::Gt(symbolic::symbol("A"), symbolic::integer(0)));

    auto& block = builder.add_block(true_case);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});

    auto& block2 = builder.add_block(false_case);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << if_else.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << if_else.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.element_id() << "_0 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"A <= 0\";" << std::endl
        << "subgraph cluster_" << block.element_id() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input_node.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];"
        << std::endl
        << tasklet.element_id() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << input_node.element_id() << " -> " << tasklet.element_id()
        << " [label=\"   _in = A   \"];" << std::endl
        << tasklet.element_id() << " -> " << output_node.element_id()
        << " [label=\"   B = _out   \"];" << std::endl
        << output_node.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];"
        << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << if_else.element_id() << "_1 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"0 < A\";" << std::endl
        << "subgraph cluster_" << block2.element_id() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input_node2.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];"
        << std::endl
        << tasklet2.element_id() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << input_node2.element_id() << " -> " << tasklet2.element_id()
        << " [label=\"   _in = B   \"];" << std::endl
        << tasklet2.element_id() << " -> " << output_node2.element_id()
        << " [label=\"   A = _out   \"];" << std::endl
        << output_node2.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];"
        << std::endl;
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

/*
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
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
        << loop.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.element_id() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << if_else.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.element_id() << "_0 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"i < 10\";" << std::endl
        << "subgraph cluster_" << block1.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << block1.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << cont1.element_id() << " [shape=cds,label=\" continue  \"];" << std::endl
        << block1.element_id() << " -> " << cont1.element_id() << " [ltail=\"cluster_"
        << block1.element_id() << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl << "subgraph cluster_" << if_else.element_id() << "_1 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"10 <= i\";" << std::endl
        << "subgraph cluster_" << block2.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << block2.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << break1.element_id() << " [shape=cds,label=\" break  \"];" << std::endl
        << block2.element_id() << " -> " << break1.element_id() << " [ltail=\"cluster_"
        << block2.element_id() << "\",minlen=3];" << std::endl;
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
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", sym_desc},
                                         {{"0", sym_desc}});
    builder.add_memlet(block1, tasklet1, "_out", output1, "void", {});

    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& if_else = builder.add_if_else(body);
    auto& case1 =
        builder.add_case(if_else, symbolic::Ge(symbolic::symbol("i"), symbolic::symbol("N")));
    auto& case2 =
        builder.add_case(if_else, symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")));

    auto& return_node = builder.add_return(case1);

    auto& block2 = builder.add_block(case2);
    auto& input2 = builder.add_access(block2, "A");
    auto& output2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"2.0", base_desc}});
    builder.add_memlet(block2, input2, "void", tasklet2, "_in", {symbolic::symbol("i")});
    builder.add_memlet(block2, tasklet2, "_out", output2, "void", {symbolic::symbol("i")});

    auto& block3 = builder.add_block(case2);
    auto& input3 = builder.add_access(block3, "i");
    auto& output3 = builder.add_access(block3, "i");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::add, {"_out", sym_desc},
                                         {{"_in", sym_desc}, {"1", sym_desc}});
    builder.add_memlet(block3, input3, "void", tasklet3, "_in", {});
    builder.add_memlet(block3, tasklet3, "_out", output3, "void", {});

    auto sdfg2 = builder.move();

    codegen::PrettyPrinter exp;
    exp << "digraph " << sdfg2->name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << block1.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << tasklet1.element_id() << " [shape=octagon,label=\"_out = 0\"];" << std::endl
        << tasklet1.element_id() << " -> " << output1.element_id() << " [label=\"   i = _out   \"];"
        << std::endl
        << output1.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];"
        << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl << "subgraph cluster_" << loop.element_id() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
        << loop.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.element_id() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << if_else.element_id() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.element_id() << "_0 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"N <= i\";" << std::endl
        << return_node.element_id() << " [shape=cds,label=\" return  \"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl << "subgraph cluster_" << if_else.element_id() << "_1 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"i < N\";" << std::endl
        << "subgraph cluster_" << block2.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input2.element_id() << " [penwidth=3.0,label=\"A\"];" << std::endl
        << tasklet2.element_id() << " [shape=octagon,label=\"_out = _in * 2.0\"];" << std::endl
        << input2.element_id() << " -> " << tasklet2.element_id()
        << " [label=\"   _in = A[i]   \"];" << std::endl
        << tasklet2.element_id() << " -> " << output2.element_id()
        << " [label=\"   A[i] = _out   \"];" << std::endl
        << output2.element_id() << " [penwidth=3.0,label=\"A\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl << "subgraph cluster_" << block3.element_id() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input3.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];"
        << std::endl
        << tasklet3.element_id() << " [shape=octagon,label=\"_out = _in + 1\"];" << std::endl
        << input3.element_id() << " -> " << tasklet3.element_id() << " [label=\"   _in = i   \"];"
        << std::endl
        << tasklet3.element_id() << " -> " << output3.element_id() << " [label=\"   i = _out   \"];"
        << std::endl
        << output3.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"i\"];"
        << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << tasklet2.element_id() << " -> " << tasklet3.element_id() << " [ltail=\"cluster_"
        << block2.element_id() << "\",lhead=\"cluster_" << block3.element_id() << "\",minlen=3];"
        << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl
        << tasklet1.element_id() << " -> " << loop.element_id() << " [ltail=\"cluster_"
        << block1.element_id() << "\",lhead=\"cluster_" << loop.element_id() << "\",minlen=3];"
        << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(*sdfg2);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}
*/

TEST(DotVisualizerTest, test_handleTasklet) {
    const std::vector<std::pair<const data_flow::TaskletCode, const std::string>> codes = {
        {data_flow::TaskletCode::assign, "="},
        {data_flow::TaskletCode::neg, "-"},
        {data_flow::TaskletCode::add, "+"},
        {data_flow::TaskletCode::sub, "-"},
        {data_flow::TaskletCode::mul, "*"},
        {data_flow::TaskletCode::div, "/"},
        {data_flow::TaskletCode::fma, "fma"},
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
        {data_flow::TaskletCode::modf, "modf"},
        {data_flow::TaskletCode::modff, "modff"},
        {data_flow::TaskletCode::modfl, "modfl"},
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
        {data_flow::TaskletCode::tanhl, "tanhl"}};
    for (const std::pair<const data_flow::TaskletCode, const std::string> code : codes) {
        const size_t arity = data_flow::arity(code.first);
        builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);

        auto& sdfg = builder.subject();
        auto& root = sdfg.root();

        types::Scalar desc{types::PrimitiveType::Int32};
        builder.add_container("x", desc);

        auto& block = builder.add_block(root);
        auto& output = builder.add_access(block, "x");
        std::vector<std::pair<std::string, sdfg::types::Scalar>> inputs;
        for (size_t i = 0; i < arity; ++i) inputs.push_back({std::to_string(i), desc});
        auto& tasklet = builder.add_tasklet(block, code.first, {"_out", desc}, inputs);
        builder.add_memlet(block, tasklet, "_out", output, "void", {});

        auto sdfg2 = builder.move();

        codegen::PrettyPrinter exp;
        exp << "digraph " << sdfg2->name() << " {" << std::endl;
        exp.setIndent(4);
        exp << "graph [compound=true];" << std::endl
            << "subgraph cluster_" << sdfg2->name() << " {" << std::endl;
        exp.setIndent(8);
        exp << "node [style=filled,fillcolor=white];" << std::endl
            << "style=filled;color=lightblue;label=\"\";" << std::endl
            << "subgraph cluster_" << block.element_id() << " {" << std::endl;
        exp.setIndent(12);
        exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
            << tasklet.element_id() << " [shape=octagon,label=\"";
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
            << tasklet.element_id() << " -> " << output.element_id()
            << " [label=\"   x = _out   \"];" << std::endl
            << output.element_id() << " [penwidth=3.0,style=\"dashed,filled\",label=\"x\"];"
            << std::endl;
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
}

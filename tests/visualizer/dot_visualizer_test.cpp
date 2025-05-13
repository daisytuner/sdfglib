#include "sdfg/visualizer/dot_visualizer.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/conditional_schedule.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(DotVisualizerTest, transpose) {
    builder::StructuredSDFGBuilder builder("transpose");

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
    ConditionalSchedule schedule(sdfg2);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg.name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop1.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0:(M-1)\";"
        << std::endl
        << loop1.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop2.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j = 0:(N-1)\";"
        << std::endl
        << loop2.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A.name() << " [penwidth=3.0,label=\"A\"];" << std::endl
        << tasklet.name() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << A.name() << " -> " << tasklet.name() << " [label=\"   _in = A[i][j]   \"];" << std::endl
        << tasklet.name() << " -> " << B.name() << " [label=\"   B[j][i] = _out   \"];" << std::endl
        << B.name() << " [penwidth=3.0,label=\"B\"];" << std::endl;
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, syrk) {
    builder::StructuredSDFGBuilder sdfg("sdfg_1");

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
    sdfg.add_memlet(block1, beta_node, "void", tasklet1, "_in2", {symbolic::integer(0)});
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
    sdfg.add_memlet(block2, tasklet2, "_out", tmp_node, "void", {symbolic::integer(0)});

    auto& block3 = sdfg.add_block(loop_j_2.root());
    auto& C_in_node_2 = sdfg.add_access(block3, "C");
    auto& C_out_node_2 = sdfg.add_access(block3, "C");
    auto& tmp_node_2 = sdfg.add_access(block3, "tmp");
    auto& tasklet3 = sdfg.add_tasklet(block3, data_flow::TaskletCode::add, {"_out", desc_element},
                                      {{"_in1", desc_element}, {"_in2", desc_element}});
    sdfg.add_memlet(block3, C_in_node_2, "void", tasklet3, "_in1",
                    {symbolic::symbol("i"), symbolic::symbol("j_2")});
    sdfg.add_memlet(block3, tmp_node_2, "void", tasklet3, "_in2", {symbolic::integer(0)});
    sdfg.add_memlet(block3, tasklet3, "_out", C_out_node_2, "void",
                    {symbolic::symbol("i"), symbolic::symbol("j_2")});

    auto sdfg2 = sdfg.move();
    ConditionalSchedule schedule(sdfg2);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << schedule.schedule(0).sdfg().name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop_i.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0:(N-1)\";"
        << std::endl
        << loop_i.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop_j_1.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_1 = 0:i\";"
        << std::endl
        << loop_j_1.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block1.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << beta_node.name() << " [penwidth=3.0,label=\"beta\"];" << std::endl
        << C_in_node_1.name() << " [penwidth=3.0,label=\"C\"];" << std::endl
        << tasklet1.name() << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << C_in_node_1.name() << " -> " << tasklet1.name() << " [label=\"   _in1 = C[i][j_1]   \"];"
        << std::endl
        << beta_node.name() << " -> " << tasklet1.name() << " [label=\"   _in2 = beta   \"];"
        << std::endl
        << tasklet1.name() << " -> " << C_out_node_1.name()
        << " [label=\"   C[i][j_1] = _out   \"];" << std::endl
        << C_out_node_1.name() << " [penwidth=3.0,label=\"C\"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << loop_k.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: k = 0:(M-1)\";"
        << std::endl
        << loop_k.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop_j_2.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: j_2 = 0:i\";"
        << std::endl
        << loop_j_2.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block2.name() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A_node.name() << " [penwidth=3.0,label=\"A\"];" << std::endl
        << tasklet2.name() << " [shape=octagon,label=\"_out = _in1 * _in2\"];" << std::endl
        << A_node.name() << " -> " << tasklet2.name() << " [label=\"   _in1 = A[j_2][k]   \"];"
        << std::endl
        << A_node.name() << " -> " << tasklet2.name() << " [label=\"   _in2 = A[i][k]   \"];"
        << std::endl
        << tasklet2.name() << " -> " << tmp_node.name() << " [label=\"   tmp = _out   \"];"
        << std::endl
        << tmp_node.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl << "subgraph cluster_" << block3.name() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << tmp_node_2.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"tmp\"];"
        << std::endl
        << C_in_node_2.name() << " [penwidth=3.0,label=\"C\"];" << std::endl
        << tasklet3.name() << " [shape=octagon,label=\"_out = _in1 + _in2\"];" << std::endl
        << C_in_node_2.name() << " -> " << tasklet3.name() << " [label=\"   _in1 = C[i][j_2]   \"];"
        << std::endl
        << tmp_node_2.name() << " -> " << tasklet3.name() << " [label=\"   _in2 = tmp   \"];"
        << std::endl
        << tasklet3.name() << " -> " << C_out_node_2.name()
        << " [label=\"   C[i][j_2] = _out   \"];" << std::endl
        << C_out_node_2.name() << " [penwidth=3.0,label=\"C\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << tasklet2.name() << " -> " << tasklet3.name() << " [ltail=\"cluster_" << block2.name()
        << "\",lhead=\"cluster_" << block3.name() << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl
        << loop_j_1.name() << " -> " << loop_k.name() << " [ltail=\"cluster_" << loop_j_1.name()
        << "\",lhead=\"cluster_" << loop_k.name() << "\",minlen=3];" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, multi_tasklet_block) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

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
    ConditionalSchedule schedule(sdfg2);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg.name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << block.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << A1.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << tasklet1.name() << " [shape=octagon,label=\"_out = 2 * _in + 1\"];" << std::endl
        << A1.name() << " -> " << tasklet1.name() << " [label=\"   _in = A[0]   \"];" << std::endl
        << tasklet1.name() << " -> " << A2.name() << " [label=\"   A[0] = _out   \"];" << std::endl
        << A2.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << tasklet2.name() << " [shape=octagon,label=\"_out = 2 * _in + 1\"];" << std::endl
        << A2.name() << " -> " << tasklet2.name() << " [label=\"   _in = A[0]   \"];" << std::endl
        << tasklet2.name() << " -> " << A3.name() << " [label=\"   A[0] = _out   \"];" << std::endl
        << A3.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, kernel_test_tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test");

    auto& sdfg = builder.subject();
    auto& kernel = builder.convert_into_kernel();
    auto& root = kernel.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    types::Array array_desc(base_desc, {symbolic::integer(8)});
    types::Array array_desc2(array_desc, {symbolic::integer(512)});
    builder.add_container("B_shared", array_desc2);

    types::Scalar sym_desc(types::PrimitiveType::UInt32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_shared", sym_desc);
    builder.add_container("i_access", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(8));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto indvar_shared = symbolic::symbol("i_shared");
    auto shared_bound =
        symbolic::And(symbolic::Lt(indvar_shared, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_shared, symbolic::max(symbolic::integer(0), bound)));
    auto shared_init = indvar;
    auto shared_update = symbolic::add(indvar_shared, symbolic::integer(1));

    auto& loop_shared =
        builder.add_for(body, indvar_shared, shared_bound, shared_init, shared_update);
    auto& body_shared = loop_shared.root();

    auto& block_shared = builder.add_block(body_shared);
    auto& B = builder.add_access(block_shared, "B");
    auto& B_shared = builder.add_access(block_shared, "B_shared");

    auto& tasklet_shared = builder.add_tasklet(block_shared, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& shared_in = builder.add_memlet(
        block_shared, B, "void", tasklet_shared, "_in",
        {symbolic::add(symbolic::mul(symbolic::integer(512), indvar_shared),
                       symbolic::add(kernel.threadIdx_x(),
                                     symbolic::mul(kernel.blockDim_x(), kernel.blockIdx_x())))});

    auto& shared_out =
        builder.add_memlet(block_shared, tasklet_shared, "_out", B_shared, "void",
                           {kernel.threadIdx_x(), symbolic::sub(indvar_shared, indvar)});

    auto& sync_block = builder.add_block(body);
    auto& libnode =
        builder.add_library_node(sync_block, data_flow::LibraryNodeType::LocalBarrier, {}, {});

    auto indvar_access = symbolic::symbol("i_access");
    auto access_bound =
        symbolic::And(symbolic::Lt(indvar_access, symbolic::add(symbolic::integer(8), indvar)),
                      symbolic::Lt(indvar_access, symbolic::max(symbolic::integer(0), bound)));
    auto access_init = indvar;
    auto access_update = symbolic::add(indvar_access, symbolic::integer(1));

    auto& loop_access =
        builder.add_for(body, indvar_access, access_bound, access_init, access_update);
    auto& body_access = loop_access.root();

    auto& block_access = builder.add_block(body_access);
    auto& B_shared_access = builder.add_access(block_access, "B_shared");
    auto& A = builder.add_access(block_access, "A");

    auto& tasklet_access = builder.add_tasklet(block_access, data_flow::TaskletCode::assign,
                                               {"_out", base_desc}, {{"_in", base_desc}});

    auto& access_in =
        builder.add_memlet(block_access, B_shared_access, "void", tasklet_access, "_in",
                           {kernel.threadIdx_x(), symbolic::sub(indvar_access, indvar)});

    auto& access_out = builder.add_memlet(
        block_access, tasklet_access, "_out", A, "void",
        {symbolic::add(symbolic::mul(symbolic::integer(512), indvar_access),
                       symbolic::add(kernel.threadIdx_x(),
                                     symbolic::mul(kernel.blockDim_x(), kernel.blockIdx_x())))});

    auto sdfgp = builder.move();
    ConditionalSchedule schedule(sdfgp);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << sdfg.name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i = 0:(N-1):8\";"
        << std::endl
        << loop.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << loop_shared.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i_shared = i; And(i_sha"
        << "red < max(0, N), i_shared < 8 + i); i_shared = 1 + i_shared\";" << std::endl
        << loop_shared.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block_shared.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << B.name() << " [penwidth=3.0,label=\"B\"];" << std::endl
        << tasklet_shared.name() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << B.name() << " -> " << tasklet_shared.name()
        << " [label=\"   _in = B[threadIdx.x + 512*i_shared + blockIdx.x*blockDim.x]   \"];"
        << std::endl
        << tasklet_shared.name() << " -> " << B_shared.name()
        << " [label=\"   B_shared[threadIdx.x][-i + i_shared] = _out   \"];" << std::endl
        << B_shared.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B_shared\"];"
        << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << sync_block.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << libnode.name() << " [shape=doubleoctagon,label=\"Local Barrier\"];" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl
        << loop_shared.name() << " -> " << libnode.name() << " [ltail=\"cluster_"
        << loop_shared.name() << "\",lhead=\"cluster_" << sync_block.name() << "\",minlen=3];"
        << std::endl
        << "subgraph cluster_" << loop_access.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: i_access = i; And(i_acc"
        << "ess < max(0, N), i_access < 8 + i); i_access = 1 + i_access\";" << std::endl
        << loop_access.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << block_access.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << B_shared_access.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B_shared\"];"
        << std::endl
        << tasklet_access.name() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << B_shared_access.name() << " -> " << tasklet_access.name()
        << " [label=\"   _in = B_shared[threadIdx.x][-i + i_access]   \"];" << std::endl
        << tasklet_access.name() << " -> " << A.name()
        << " [label=\"   A[threadIdx.x + 512*i_access + blockIdx.x*blockDim.x] = _out   \"];"
        << std::endl
        << A.name() << " [penwidth=3.0,label=\"A\"];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl
        << libnode.name() << " -> " << loop_access.name() << " [ltail=\"cluster_"
        << sync_block.name() << "\",lhead=\"cluster_" << loop_access.name() << "\",minlen=3];"
        << std::endl;
    exp.setIndent(8);
    exp << "}" << std::endl;
    exp.setIndent(4);
    exp << "}" << std::endl;
    exp.setIndent(0);
    exp << "}" << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, test_if_else) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

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
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(false_case);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();
    ConditionalSchedule schedule(sdfg);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << schedule.schedule(0).sdfg().name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << if_else.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << if_else.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.name() << "_0 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"A <= 0\";" << std::endl
        << "subgraph cluster_" << block.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input_node.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];" << std::endl
        << tasklet.name() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << input_node.name() << " -> " << tasklet.name() << " [label=\"   _in = A   \"];"
        << std::endl
        << tasklet.name() << " -> " << output_node.name() << " [label=\"   B = _out   \"];"
        << std::endl
        << output_node.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];"
        << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl;
    exp.setIndent(12);
    exp << "}" << std::endl << "subgraph cluster_" << if_else.name() << "_1 {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"0 < A\";" << std::endl
        << "subgraph cluster_" << block2.name() << " {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << input_node2.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"B\"];" << std::endl
        << tasklet2.name() << " [shape=octagon,label=\"_out = _in\"];" << std::endl
        << input_node2.name() << " -> " << tasklet2.name() << " [label=\"   _in = B   \"];"
        << std::endl
        << tasklet2.name() << " -> " << output_node2.name() << " [label=\"   A = _out   \"];"
        << std::endl
        << output_node2.name() << " [penwidth=3.0,style=\"dashed,filled\",label=\"A\"];"
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

TEST(DotVisualizerTest, test_while) {
    builder::StructuredSDFGBuilder builder("sdfg");

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
    auto& cont1 = builder.add_continue(case1, loop);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2, loop);

    auto sdfg = builder.move();
    ConditionalSchedule schedule(sdfg);

    codegen::PrettyPrinter exp;
    exp << "digraph " << schedule.name() << " {" << std::endl;
    exp.setIndent(4);
    exp << "graph [compound=true];" << std::endl
        << "subgraph cluster_" << schedule.schedule(0).sdfg().name() << " {" << std::endl;
    exp.setIndent(8);
    exp << "node [style=filled,fillcolor=white];" << std::endl
        << "style=filled;color=lightblue;label=\"\";" << std::endl
        << "subgraph cluster_" << loop.name() << " {" << std::endl;
    exp.setIndent(12);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";" << std::endl
        << loop.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.name() << " {" << std::endl;
    exp.setIndent(16);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";" << std::endl
        << if_else.name() << " [shape=point,style=invis,label=\"\"];" << std::endl
        << "subgraph cluster_" << if_else.name() << "_0 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"i < 10\";" << std::endl
        << "subgraph cluster_" << block1.name() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << block1.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << cont1.name() << " [shape=cds,label=\" continue  \"];" << std::endl
        << block1.name() << " -> " << cont1.name() << " [ltail=\"cluster_" << block1.name()
        << "\",minlen=3];" << std::endl;
    exp.setIndent(16);
    exp << "}" << std::endl << "subgraph cluster_" << if_else.name() << "_1 {" << std::endl;
    exp.setIndent(20);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"10 <= i\";" << std::endl
        << "subgraph cluster_" << block2.name() << " {" << std::endl;
    exp.setIndent(24);
    exp << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl
        << block2.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    exp.setIndent(20);
    exp << "}" << std::endl
        << break1.name() << " [shape=cds,label=\" break  \"];" << std::endl
        << block2.name() << " -> " << break1.name() << " [ltail=\"cluster_" << block2.name()
        << "\",minlen=3];" << std::endl;
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), exp.str());
}

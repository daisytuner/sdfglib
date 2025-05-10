#include "sdfg/visualizer/dot_visualizer.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "fixtures/polybench.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/conditional_schedule.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
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
    auto update2 = symbolic::add(indvar2, symbolic::integer(2));

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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph transpose {
    graph [compound=true];
    subgraph cluster_transpose {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_1 {
            style=filled;shape=box;fillcolor=white;color=black;label="for: i = 0:(M-1)";
            __node_1 [shape=point,style=invis,label=""];
            subgraph cluster___node_4 {
                style=filled;shape=box;fillcolor=white;color=black;label="for: j = 0:(N-1):2";
                __node_4 [shape=point,style=invis,label=""];
                subgraph cluster___node_7 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_9 [penwidth=3.0,label="A"];
                    __element_11 [shape=octagon,label="_out = _in"];
                    __element_9 -> __element_11 [label="   _in = A[i][j]   "];
                    __element_11 -> __element_10 [label="   B[j][i] = _out   "];
                    __element_10 [penwidth=3.0,label="B"];
                }
            }
        }
    }
}
)-");
}

TEST(DotVisualizerTest, syrk) {
    auto sdfg = syrk();
    ConditionalSchedule schedule(sdfg);

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph sdfg_1 {
    graph [compound=true];
    subgraph cluster_sdfg_1 {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_1 {
            style=filled;shape=box;fillcolor=white;color=black;label="for: i = 0:(N-1)";
            __node_1 [shape=point,style=invis,label=""];
            subgraph cluster___node_4 {
                style=filled;shape=box;fillcolor=white;color=black;label="for: j_1 = 0:i";
                __node_4 [shape=point,style=invis,label=""];
                subgraph cluster___node_7 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_11 [penwidth=3.0,label="beta"];
                    __element_9 [penwidth=3.0,label="C"];
                    __element_12 [shape=octagon,label="_out = _in1 * _in2"];
                    __element_9 -> __element_12 [label="   _in1 = C[i][j_1]   "];
                    __element_11 -> __element_12 [label="   _in2 = beta   "];
                    __element_12 -> __element_10 [label="   C[i][j_1] = _out   "];
                    __element_10 [penwidth=3.0,label="C"];
                }
            }
            subgraph cluster___node_16 {
                style=filled;shape=box;fillcolor=white;color=black;label="for: k = 0:(M-1)";
                __node_16 [shape=point,style=invis,label=""];
                subgraph cluster___node_19 {
                    style=filled;shape=box;fillcolor=white;color=black;label="for: j_2 = 0:i";
                    __node_19 [shape=point,style=invis,label=""];
                    subgraph cluster___node_22 {
                        style=filled;shape=box;fillcolor=white;color=black;label="";
                        __element_24 [penwidth=3.0,label="A"];
                        __element_26 [shape=octagon,label="_out = _in1 * _in2"];
                        __element_24 -> __element_26 [label="   _in1 = A[j_2][k]   "];
                        __element_24 -> __element_26 [label="   _in2 = A[i][k]   "];
                        __element_26 -> __element_25 [label="   tmp = _out   "];
                        __element_25 [penwidth=3.0,style="dashed,filled",label="tmp"];
                    }
                    subgraph cluster___node_30 {
                        style=filled;shape=box;fillcolor=white;color=black;label="";
                        __element_34 [penwidth=3.0,style="dashed,filled",label="tmp"];
                        __element_32 [penwidth=3.0,label="C"];
                        __element_35 [shape=octagon,label="_out = _in1 + _in2"];
                        __element_32 -> __element_35 [label="   _in1 = C[i][j_2]   "];
                        __element_34 -> __element_35 [label="   _in2 = tmp   "];
                        __element_35 -> __element_33 [label="   C[i][j_2] = _out   "];
                        __element_33 [penwidth=3.0,label="C"];
                    }
                    __element_26 -> __element_35 [ltail="cluster___node_22",lhead="cluster___node_30",minlen=3];
                }
            }
            __node_4 -> __node_16 [ltail="cluster___node_4",lhead="cluster___node_16",minlen=3];
        }
    }
}
)-");
}

TEST(DotVisualizerTest, block_fusion_chain) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {});

    auto& node1_1 = builder.add_access(block1, "A");
    auto& node2_1 = builder.add_access(block1, "A");
    auto& tasklet_1 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block1, node1_1, "void", tasklet_1, "_in", {symbolic::integer(0)});
    builder.add_memlet(block1, tasklet_1, "_out", node2_1, "void", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root, {});

    auto& node1_2 = builder.add_access(block2, "A");
    auto& node2_2 = builder.add_access(block2, "A");
    auto& tasklet_2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", desc_element},
                            {{"2", desc_element}, {"_in", desc_element}, {"1", desc_element}});
    builder.add_memlet(block2, node1_2, "void", tasklet_2, "_in", {symbolic::integer(0)});
    builder.add_memlet(block2, tasklet_2, "_out", node2_2, "void", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    fusion_pass.run(builder_opt, analysis_manager);

    sdfg = builder_opt.move();
    ConditionalSchedule schedule(sdfg);

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph sdfg_1 {
    graph [compound=true];
    subgraph cluster_sdfg_1 {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_1 {
            style=filled;shape=box;fillcolor=white;color=black;label="";
            __element_3 [penwidth=3.0,style="dashed,filled",label="A"];
            __element_5 [shape=octagon,label="_out = 2 * _in + 1"];
            __element_3 -> __element_5 [label="   _in = A[0]   "];
            __element_5 -> __element_4 [label="   A[0] = _out   "];
            __element_4 [penwidth=3.0,style="dashed,filled",label="A"];
            __element_1 [shape=octagon,label="_out = 2 * _in + 1"];
            __element_4 -> __element_1 [label="   _in = A[0]   "];
            __element_1 -> __element_2 [label="   A[0] = _out   "];
            __element_2 [penwidth=3.0,style="dashed,filled",label="A"];
        }
    }
}
)-");
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph sdfg_test {
    graph [compound=true];
    subgraph cluster_sdfg_test {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_5 {
            style=filled;shape=box;fillcolor=white;color=black;label="for: i = 0:(N-1):8";
            __node_5 [shape=point,style=invis,label=""];
            subgraph cluster___node_8 {
                style=filled;shape=box;fillcolor=white;color=black;label="for: i_shared = i; And(i_shared < max(0, N), i_shared < 8 + i); i_shared = 1 + i_shared";
                __node_8 [shape=point,style=invis,label=""];
                subgraph cluster___node_11 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_13 [penwidth=3.0,label="B"];
                    __element_15 [shape=octagon,label="_out = _in"];
                    __element_13 -> __element_15 [label="   _in = B[threadIdx.x + 512*i_shared + blockIdx.x*blockDim.x]   "];
                    __element_15 -> __element_14 [label="   B_shared[threadIdx.x][-i + i_shared] = _out   "];
                    __element_14 [penwidth=3.0,style="dashed,filled",label="B_shared"];
                }
            }
            subgraph cluster___node_18 {
                style=filled;shape=box;fillcolor=white;color=black;label="";
                __element_20 [shape=doubleoctagon,label="Local Barrier"];
            }
            __node_8 -> __element_20 [ltail="cluster___node_8",lhead="cluster___node_18",minlen=3];
            subgraph cluster___node_21 {
                style=filled;shape=box;fillcolor=white;color=black;label="for: i_access = i; And(i_access < max(0, N), i_access < 8 + i); i_access = 1 + i_access";
                __node_21 [shape=point,style=invis,label=""];
                subgraph cluster___node_24 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_26 [penwidth=3.0,style="dashed,filled",label="B_shared"];
                    __element_28 [shape=octagon,label="_out = _in"];
                    __element_26 -> __element_28 [label="   _in = B_shared[threadIdx.x][-i + i_access]   "];
                    __element_28 -> __element_27 [label="   A[threadIdx.x + 512*i_access + blockIdx.x*blockDim.x] = _out   "];
                    __element_27 [penwidth=3.0,label="A"];
                }
            }
            __element_20 -> __node_21 [ltail="cluster___node_18",lhead="cluster___node_21",minlen=3];
        }
    }
}
)-");
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph sdfg_1 {
    graph [compound=true];
    subgraph cluster_sdfg_1 {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_1 {
            style=filled;shape=box;fillcolor=white;color=black;label="if:";
            __node_1 [shape=point,style=invis,label=""];
            subgraph cluster___node_10 {
                style=filled;shape=box;fillcolor=white;color=black;label="A <= 0";
                subgraph cluster___node_5 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_7 [penwidth=3.0,style="dashed,filled",label="A"];
                    __element_9 [shape=octagon,label="_out = _in"];
                    __element_7 -> __element_9 [label="   _in = A   "];
                    __element_9 -> __element_8 [label="   B = _out   "];
                    __element_8 [penwidth=3.0,style="dashed,filled",label="B"];
                }
            }
            subgraph cluster___node_11 {
                style=filled;shape=box;fillcolor=white;color=black;label="0 < A";
                subgraph cluster___node_12 {
                    style=filled;shape=box;fillcolor=white;color=black;label="";
                    __element_14 [penwidth=3.0,style="dashed,filled",label="B"];
                    __element_16 [shape=octagon,label="_out = _in"];
                    __element_14 -> __element_16 [label="   _in = B   "];
                    __element_16 -> __element_15 [label="   A = _out   "];
                    __element_15 [penwidth=3.0,style="dashed,filled",label="A"];
                }
            }
        }
    }
}
)-");
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

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    EXPECT_EQ(dot.getStream().str(), R"-(digraph sdfg {
    graph [compound=true];
    subgraph cluster_sdfg {
        node [style=filled,fillcolor=white];
        style=filled;color=lightblue;label="";
        subgraph cluster___node_1 {
            style=filled;shape=box;fillcolor=white;color=black;label="while:";
            __node_1 [shape=point,style=invis,label=""];
            subgraph cluster___node_4 {
                style=filled;shape=box;fillcolor=white;color=black;label="if:";
                __node_4 [shape=point,style=invis,label=""];
                subgraph cluster___node_40 {
                    style=filled;shape=box;fillcolor=white;color=black;label="i < 10";
                    subgraph cluster___node_7 {
                        style=filled;shape=box;fillcolor=white;color=black;label="";
                        __node_7 [shape=point,style=invis,label=""];
                    }
                    __node_9 [shape=cds,label=" continue  "];
                    __node_7 -> __node_9 [ltail="cluster___node_7",minlen=3];
                }
                subgraph cluster___node_41 {
                    style=filled;shape=box;fillcolor=white;color=black;label="10 <= i";
                    subgraph cluster___node_12 {
                        style=filled;shape=box;fillcolor=white;color=black;label="";
                        __node_12 [shape=point,style=invis,label=""];
                    }
                    __node_14 [shape=cds,label=" break  "];
                    __node_12 -> __node_14 [ltail="cluster___node_12",minlen=3];
                }
            }
        }
    }
}
)-");
}

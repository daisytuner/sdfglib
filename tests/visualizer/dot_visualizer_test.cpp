#include "sdfg/visualizer/dot_visualizer.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "fixtures/polybench.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/c_code_generator.h"
#include "sdfg/codegen/code_generators/cuda_code_generator.h"
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

    codegen::CCodeGenerator cgen(schedule, false);
    EXPECT_TRUE(cgen.generate());
    std::cout << cgen.main().str() << std::endl << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    std::cout << dot.getStream().str() << std::endl;
}

TEST(DotVisualizerTest, syrk) {
    auto sdfg = syrk();
    ConditionalSchedule schedule(sdfg);

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    std::cout << dot.getStream().str() << std::endl;

    auto sdfg2 = syrk();
    builder::StructuredSDFGBuilder builder_opt(sdfg2);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BlockFusionPass fusion_pass;
    fusion_pass.run(builder_opt, analysis_manager);

    auto sdfg3 = builder_opt.move();
    ConditionalSchedule schedule2(sdfg3);

    visualizer::DotVisualizer dot2(schedule2);
    dot2.visualize();
    std::cout << dot2.getStream().str() << std::endl;
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
    std::cout << dot.getStream().str() << std::endl;
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

    codegen::CUDACodeGenerator cgen(schedule, false);
    EXPECT_TRUE(cgen.generate());
    std::cout << cgen.main().str() << std::endl << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    std::cout << dot.getStream().str() << std::endl;
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

    codegen::CCodeGenerator cgen(schedule, false);
    EXPECT_TRUE(cgen.generate());
    std::cout << cgen.main().str() << std::endl << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    std::cout << dot.getStream().str() << std::endl;
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

    codegen::CCodeGenerator cgen(schedule, false);
    EXPECT_TRUE(cgen.generate());
    std::cout << cgen.main().str() << std::endl << std::endl;

    visualizer::DotVisualizer dot(schedule);
    dot.visualize();
    std::cout << dot.getStream().str() << std::endl;
}

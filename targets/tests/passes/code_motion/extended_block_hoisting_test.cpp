#include "sdfg/passes/code_motion/extended_block_hoisting.h"

#include <gtest/gtest.h>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/cuda/cuda.h"
#include "sdfg/cuda/cuda_offloading_node.h"
#include "sdfg/cuda/schedule.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/memory/external_offloading_node.h"
#include "sdfg/memory/offloading_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(ExtendedBlockHoistingTest, Map_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant memcpy
    auto& block1 = builder.add_block(body);
    {
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto& block3 = builder.add_block(body);
    {
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExtendedBlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}
/*
TEST(ExtendedBlockHoistingTest, For_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant memcpy
    auto& block1 = builder.add_block(body);
    {
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto& block3 = builder.add_block(body);
    {
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExtendedBlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}

TEST(ExtendedBlockHoistingTest, IfElse_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant memcpy
    {
        auto& block1 = builder.add_block(case1);
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 1: Branch invariant memcpy
    {
        auto& block3 = builder.add_block(case1);
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 2: Branch invariant memcpy
    {
        auto& block1 = builder.add_block(case2);
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Case 2: Branch invariant memcpy
    {
        auto& block3 = builder.add_block(case2);
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExtendedBlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}
*/

TEST(ExtendedBlockHoistingTest, waxpby_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("n", sym_desc, true);
    auto n = symbolic::symbol("n");
    auto id = symbolic::zero();
    builder.add_container("i", sym_desc);
    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, n);
    auto init = symbolic::zero();
    auto update = symbolic::add(indvar, symbolic::one());

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("beta", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("w", desc, true);
    builder.add_container("tmp", base_desc);

    types::Pointer d_desc(base_desc);
    d_desc.storage_type(types::StorageType::NV_Generic());
    const std::string d_x1 = cuda::CUDA_DEVICE_PREFIX + "x1";
    builder.add_container(d_x1, d_desc);
    const std::string d_x2 = cuda::CUDA_DEVICE_PREFIX + "x2";
    builder.add_container(d_x2, d_desc);
    const std::string d_x3 = cuda::CUDA_DEVICE_PREFIX + "x3";
    builder.add_container(d_x3, d_desc);
    const std::string d_y1 = cuda::CUDA_DEVICE_PREFIX + "y1";
    builder.add_container(d_y1, d_desc);
    const std::string d_y2 = cuda::CUDA_DEVICE_PREFIX + "y2";
    builder.add_container(d_y2, d_desc);
    const std::string d_y3 = cuda::CUDA_DEVICE_PREFIX + "y3";
    builder.add_container(d_y3, d_desc);
    const std::string d_w1 = cuda::CUDA_DEVICE_PREFIX + "w1";
    builder.add_container(d_w1, d_desc);
    const std::string d_w2 = cuda::CUDA_DEVICE_PREFIX + "w2";
    builder.add_container(d_w2, d_desc);
    const std::string d_w3 = cuda::CUDA_DEVICE_PREFIX + "w3";
    builder.add_container(d_w3, d_desc);

    types::Scalar cond_desc(types::PrimitiveType::Bool);
    builder.add_container("cond1", cond_desc);
    builder.add_container("cond2", cond_desc);

    {
        auto& block = builder.add_block(root);
        auto& alpha = builder.add_access(block, "alpha");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond1 = builder.add_access(block, "cond1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond1, {});
    }

    {
        auto& block = builder.add_block(root);
        auto& beta = builder.add_access(block, "beta");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond2 = builder.add_access(block, "cond2");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, beta, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond2, {});
    }

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond1"), symbolic::__true__()));
    auto& case2 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond2"), symbolic::__true__()));
    auto& case3 = builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Ne(symbolic::symbol("cond1"), symbolic::__true__()),
                symbolic::Ne(symbolic::symbol("cond2"), symbolic::__true__()))
    );

    // Case 1

    {
        auto& block = builder.add_block(case1);
        auto& x = builder.add_access(block, "x");
        auto& d_x = builder.add_access(block, d_x1);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_x, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case1);
        auto& y = builder.add_access(block, "y");
        auto& d_y = builder.add_access(block, d_y1);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_y, {}, d_desc);
    }

    {
        auto& map = builder.add_map(case1, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w1);
        auto& d_x = builder.add_access(block, d_x1);
        auto& beta = builder.add_access(block, "beta");
        auto& d_y = builder.add_access(block, d_y1);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, beta, tasklet, "_in1", {});
        builder.add_computational_memlet(block, d_y, tasklet, "_in2", {indvar});
        builder.add_computational_memlet(block, d_x, tasklet, "_in3", {indvar});
        builder.add_computational_memlet(block, tasklet, "_out", d_w, {indvar});
    }

    {
        auto& block = builder.add_block(case1);
        auto& w = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w1);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, d_w, libnode, "_src", {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_dst", w, {}, desc);
    }

    // Case 2

    {
        auto& block = builder.add_block(case2);
        auto& x = builder.add_access(block, "x");
        auto& d_x = builder.add_access(block, d_x2);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_x, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case2);
        auto& y = builder.add_access(block, "y");
        auto& d_y = builder.add_access(block, d_y2);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_y, {}, d_desc);
    }

    {
        auto& map = builder.add_map(case2, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w2);
        auto& alpha = builder.add_access(block, "alpha");
        auto& d_x = builder.add_access(block, d_x2);
        auto& d_y = builder.add_access(block, d_y2);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, alpha, tasklet, "_in1", {});
        builder.add_computational_memlet(block, d_x, tasklet, "_in2", {indvar});
        builder.add_computational_memlet(block, d_y, tasklet, "_in3", {indvar});
        builder.add_computational_memlet(block, tasklet, "_out", d_w, {indvar});
    }

    {
        auto& block = builder.add_block(case2);
        auto& w = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w2);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, d_w, libnode, "_src", {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_dst", w, {}, desc);
    }

    // Case 3

    {
        auto& block = builder.add_block(case3);
        auto& x = builder.add_access(block, "x");
        auto& d_x = builder.add_access(block, d_x3);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_x, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case3);
        auto& y = builder.add_access(block, "y");
        auto& d_y = builder.add_access(block, d_y3);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block, libnode, "_dst", d_y, {}, d_desc);
    }

    {
        auto& map = builder.add_map(case3, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w2);
        auto& alpha = builder.add_access(block, "alpha");
        auto& d_x = builder.add_access(block, d_x3);
        auto& beta = builder.add_access(block, "beta");
        auto& d_y = builder.add_access(block, d_y3);
        auto& tmp = builder.add_access(block, "tmp");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, d_x, tasklet1, "_in2", {indvar});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, beta, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, d_y, tasklet2, "_in2", {indvar});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in3", {});
        builder.add_computational_memlet(block, tasklet2, "_out", d_w, {indvar});
    }

    {
        auto& block = builder.add_block(case3);
        auto& w = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w2);
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block, DebugInfo(), n, id, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, d_w, libnode, "_src", {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_dst", w, {}, desc);
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExtendedBlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 6);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
    EXPECT_EQ(case3.size(), 1);
}

TEST(ExtendedBlockHoistingTest, waxpby_External) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("n", sym_desc, true);
    auto n = symbolic::symbol("n");
    auto id = symbolic::zero();
    builder.add_container("i", sym_desc);
    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, n);
    auto init = symbolic::zero();
    auto update = symbolic::add(indvar, symbolic::one());

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("beta", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("w", desc, true);
    builder.add_container("tmp", base_desc);

    types::Pointer d_desc(base_desc);
    d_desc.storage_type(types::StorageType::NV_Generic());
    const std::string d_x1 = cuda::CUDA_DEVICE_PREFIX + "x1";
    builder.add_container(d_x1, d_desc);
    const std::string d_x2 = cuda::CUDA_DEVICE_PREFIX + "x2";
    builder.add_container(d_x2, d_desc);
    const std::string d_x3 = cuda::CUDA_DEVICE_PREFIX + "x3";
    builder.add_container(d_x3, d_desc);
    const std::string d_y1 = cuda::CUDA_DEVICE_PREFIX + "y1";
    builder.add_container(d_y1, d_desc);
    const std::string d_y2 = cuda::CUDA_DEVICE_PREFIX + "y2";
    builder.add_container(d_y2, d_desc);
    const std::string d_y3 = cuda::CUDA_DEVICE_PREFIX + "y3";
    builder.add_container(d_y3, d_desc);
    const std::string d_w1 = cuda::CUDA_DEVICE_PREFIX + "w1";
    builder.add_container(d_w1, d_desc);
    const std::string d_w2 = cuda::CUDA_DEVICE_PREFIX + "w2";
    builder.add_container(d_w2, d_desc);
    const std::string d_w3 = cuda::CUDA_DEVICE_PREFIX + "w3";
    builder.add_container(d_w3, d_desc);

    types::Pointer opaque_ptr;
    types::Function waxpby_kernel(opaque_ptr);
    waxpby_kernel.add_param(sym_desc);
    waxpby_kernel.add_param(base_desc);
    waxpby_kernel.add_param(opaque_ptr);
    waxpby_kernel.add_param(base_desc);
    waxpby_kernel.add_param(opaque_ptr);
    waxpby_kernel.add_param(opaque_ptr);
    waxpby_kernel.add_param(opaque_ptr);
    waxpby_kernel.add_param(opaque_ptr);
    waxpby_kernel.add_param(opaque_ptr);
    builder.add_container("waxpby_kernel", waxpby_kernel, false, true);

    types::Scalar cond_desc(types::PrimitiveType::Bool);
    builder.add_container("cond1", cond_desc);
    builder.add_container("cond2", cond_desc);

    {
        auto& block = builder.add_block(root);
        auto& alpha = builder.add_access(block, "alpha");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond1 = builder.add_access(block, "cond1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond1, {});
    }

    {
        auto& block = builder.add_block(root);
        auto& beta = builder.add_access(block, "beta");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond2 = builder.add_access(block, "cond2");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, beta, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond2, {});
    }

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond1"), symbolic::__true__()));
    auto& case2 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond2"), symbolic::__true__()));
    auto& case3 = builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Ne(symbolic::symbol("cond1"), symbolic::__true__()),
                symbolic::Ne(symbolic::symbol("cond2"), symbolic::__true__()))
    );

    // Case 1

    {
        auto& block = builder.add_block(case1);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x1);
        auto& d_x_out = builder.add_access(block, d_x1);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_x_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case1);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_y_in = builder.add_access(block, d_y1);
        auto& d_y_out = builder.add_access(block, d_y1);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            4,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_y_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case1);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_constant(block, "1.0", base_desc);
        auto& x_in = builder.add_access(block, "x");
        auto& x_out = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x1);
        auto& d_x_out = builder.add_access(block, d_x1);
        auto& d_y_in = builder.add_access(block, d_y1);
        auto& d_y_out = builder.add_access(block, d_y1);
        auto& d_w_in = builder.add_access(block, d_w1);
        auto& d_w_out = builder.add_access(block, d_w1);
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "waxpby_kernel",
            {"_arg2", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_arg7", {}, desc);
        builder.add_computational_memlet(block, d_w_in, libnode, "_arg8", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg4", y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg6", d_x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg7", d_y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg8", d_w_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case1);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w1);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            5,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_w, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
    }

    // Case 2

    {
        auto& block = builder.add_block(case2);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x2);
        auto& d_x_out = builder.add_access(block, d_x2);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_x_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case2);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_y_in = builder.add_access(block, d_y2);
        auto& d_y_out = builder.add_access(block, d_y2);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            4,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_y_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case2);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x_in = builder.add_access(block, "x");
        auto& x_out = builder.add_access(block, "x");
        auto& beta = builder.add_constant(block, "1.0", base_desc);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x1);
        auto& d_x_out = builder.add_access(block, d_x1);
        auto& d_y_in = builder.add_access(block, d_y1);
        auto& d_y_out = builder.add_access(block, d_y1);
        auto& d_w_in = builder.add_access(block, d_w1);
        auto& d_w_out = builder.add_access(block, d_w1);
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "waxpby_kernel",
            {"_arg2", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_arg7", {}, desc);
        builder.add_computational_memlet(block, d_w_in, libnode, "_arg8", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg4", y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg6", d_x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg7", d_y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg8", d_w_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case2);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w2);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            5,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_w, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
    }

    // Case 3

    {
        auto& block = builder.add_block(case3);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x3);
        auto& d_x_out = builder.add_access(block, d_x3);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_x_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case3);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w = builder.add_access(block, "w");
        auto& d_y_in = builder.add_access(block, d_y3);
        auto& d_y_out = builder.add_access(block, d_y3);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            4,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block, libnode, "_ret", d_y_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case3);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x_in = builder.add_access(block, "x");
        auto& x_out = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_x_in = builder.add_access(block, d_x1);
        auto& d_x_out = builder.add_access(block, d_x1);
        auto& d_y_in = builder.add_access(block, d_y1);
        auto& d_y_out = builder.add_access(block, d_y1);
        auto& d_w_in = builder.add_access(block, d_w1);
        auto& d_w_out = builder.add_access(block, d_w1);
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "waxpby_kernel",
            {"_arg2", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5", "_arg6", "_arg7", "_arg8"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_x_in, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, d_y_in, libnode, "_arg7", {}, desc);
        builder.add_computational_memlet(block, d_w_in, libnode, "_arg8", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg4", y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg6", d_x_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg7", d_y_out, {}, d_desc);
        builder.add_computational_memlet(block, libnode, "_arg8", d_w_out, {}, d_desc);
    }

    {
        auto& block = builder.add_block(case3);
        auto& n = builder.add_access(block, "n");
        auto& alpha = builder.add_access(block, "alpha");
        auto& x = builder.add_access(block, "x");
        auto& beta = builder.add_access(block, "beta");
        auto& y = builder.add_access(block, "y");
        auto& w_in = builder.add_access(block, "w");
        auto& w_out = builder.add_access(block, "w");
        auto& d_w = builder.add_access(block, d_w3);
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4", "_arg5"},
            "waxpby",
            5,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, alpha, libnode, "_arg1", {}, base_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, beta, libnode, "_arg3", {}, base_desc);
        builder.add_computational_memlet(block, y, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, w_in, libnode, "_arg5", {}, desc);
        builder.add_computational_memlet(block, d_w, libnode, "_arg6", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg5", w_out, {}, d_desc);
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ExtendedBlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 6);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
    EXPECT_EQ(case3.size(), 1);
}

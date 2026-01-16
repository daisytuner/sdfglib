#include "sdfg/passes/readonly_transfer_hoisting_pass.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
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

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentCUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_cuda_x", desc);
    builder.add_container("__daisy_cuda_y", desc);

    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& x = builder.add_access(block_in_x, "x");
        auto& device_x = builder.add_access(block_in_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_x, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_dst", device_x, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y = builder.add_access(block_in_y, "__daisy_cuda_y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_y, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_dst", device_y, {}, desc);
    }

    auto& map =
        builder
            .add_map(while_body, i, symbolic::Lt(i, n), zero, symbolic::add(i, one), cuda::ScheduleType_CUDA::create());
    auto& map_body = map.root();

    auto& block = builder.add_block(map_body);
    {
        auto& two = builder.add_constant(block, "2.0f", base_desc);
        auto& device_x = builder.add_access(block, "__daisy_cuda_x");
        auto& device_y = builder.add_access(block, "__daisy_cuda_y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, two, tasklet, "_in1", {});
        builder.add_computational_memlet(block, device_x, tasklet, "_in2", {i});
        builder.add_computational_memlet(block, tasklet, "_out", device_y, {i});
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_x, DebugInfo(), n, zero, memory::DataTransferDirection::NONE, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& device_y = builder.add_access(block_out_y, "__daisy_cuda_y");
        auto& y = builder.add_access(block_out_y, "y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_y, DebugInfo(), n, zero, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_dst", y, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(while_body.size(), 4);
}

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentExternal) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_inline_x", desc);
    builder.add_container("__daisy_inline_y", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Pointer opaque_ptr;
    types::Function double_alloc_type(desc);
    double_alloc_type.add_param(sym_desc);
    double_alloc_type.add_param(desc);
    double_alloc_type.add_param(desc);
    builder.add_container("double_alloc_1", double_alloc_type, false, true);
    builder.add_container("double_alloc_2", double_alloc_type, false, true);
    types::Function double_copy_in_type(desc);
    double_copy_in_type.add_param(sym_desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    builder.add_container("double_in_1", double_copy_in_type, false, true);
    builder.add_container("double_in_2", double_copy_in_type, false, true);
    types::Function double_kernel_type(void_type);
    double_kernel_type.add_param(sym_desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    builder.add_container("double_kernel", double_kernel_type, false, true);
    types::Function double_copy_out_type(void_type);
    double_copy_out_type.add_param(sym_desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    builder.add_container("double_out_2", double_copy_out_type, false, true);
    types::Function double_free_type(desc);
    double_free_type.add_param(sym_desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    builder.add_container("double_free_1", double_free_type, false, true);
    builder.add_container("double_free_2", double_free_type, false, true);

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_x, "n");
        auto& x = builder.add_access(block_in_x, "x");
        auto& y = builder.add_constant(block_in_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_x, device_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_ret", device_x_out, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_y, "n");
        auto& x = builder.add_constant(block_in_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y_in = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_y, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_y, device_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_ret", device_y_out, {}, desc);
    }

    auto& block = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block, "n");
        auto& x = builder.add_constant(block, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& device_x_in = builder.add_access(block, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block, "__daisy_inline_x");
        auto& device_y_in = builder.add_access(block, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "double_kernel",
            {"_arg2", "_arg3", "_arg4"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, device_x_in, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block, device_y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", y_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg3", device_x_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg4", device_y_out, {}, desc);
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_x, "n");
        auto& x = builder.add_access(block_out_x, "x");
        auto& y = builder.add_constant(block_out_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::NONE,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_y, "n");
        auto& x = builder.add_constant(block_out_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block_out_y, "y");
        auto& y_out = builder.add_access(block_out_y, "y");
        auto& device_y = builder.add_access(block_out_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_y, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_arg2", y_out, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(while_body.size(), 4);
}

TEST(ReadonlyTransferHoistingPassTest, HoistLocalUnpackCUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("xy", desc2, true);
    builder.add_container("x_ref", desc2);
    builder.add_container("y_ref", desc2);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("__daisy_cuda_x", desc);
    builder.add_container("__daisy_cuda_y", desc);

    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");

    auto& block_x_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_x_ref, "xy");
        auto& x_ref = builder.add_access(block_x_ref, "x_ref");
        builder.add_reference_memlet(block_x_ref, xy, x_ref, {zero}, desc2);
    }

    auto& block_x = builder.add_block(root);
    {
        auto& x_ref = builder.add_access(block_x, "x_ref");
        auto& x = builder.add_access(block_x, "x");
        builder.add_dereference_memlet(block_x, x_ref, x, true, desc2);
    }

    auto& block_y_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_y_ref, "xy");
        auto& y_ref = builder.add_access(block_y_ref, "y");
        builder.add_reference_memlet(block_y_ref, xy, y_ref, {one}, desc2);
    }

    auto& block_y = builder.add_block(root);
    {
        auto& y_ref = builder.add_access(block_y, "y_ref");
        auto& y = builder.add_access(block_y, "y");
        builder.add_dereference_memlet(block_y, y_ref, y, true, desc2);
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& x = builder.add_access(block_in_x, "x");
        auto& device_x = builder.add_access(block_in_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_x, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_dst", device_x, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y = builder.add_access(block_in_y, "__daisy_cuda_y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_y, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_dst", device_y, {}, desc);
    }

    auto& map =
        builder
            .add_map(while_body, i, symbolic::Lt(i, n), zero, symbolic::add(i, one), cuda::ScheduleType_CUDA::create());
    auto& map_body = map.root();

    auto& block = builder.add_block(map_body);
    {
        auto& two = builder.add_constant(block, "2.0f", base_desc);
        auto& device_x = builder.add_access(block, "__daisy_cuda_x");
        auto& device_y = builder.add_access(block, "__daisy_cuda_y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, two, tasklet, "_in1", {});
        builder.add_computational_memlet(block, device_x, tasklet, "_in2", {i});
        builder.add_computational_memlet(block, tasklet, "_out", device_y, {i});
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_x, DebugInfo(), n, zero, memory::DataTransferDirection::NONE, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& device_y = builder.add_access(block_out_y, "__daisy_cuda_y");
        auto& y = builder.add_access(block_out_y, "y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_y, DebugInfo(), n, zero, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_dst", y, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 7);
    EXPECT_EQ(while_body.size(), 4);
}

TEST(ReadonlyTransferHoistingPassTest, HoistLocalUnpackExternal) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("xy", desc2, true);
    builder.add_container("x_ref", desc2);
    builder.add_container("y_ref", desc2);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("__daisy_inline_x", desc);
    builder.add_container("__daisy_inline_y", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Pointer opaque_ptr;
    types::Function double_alloc_type(desc);
    double_alloc_type.add_param(sym_desc);
    double_alloc_type.add_param(desc);
    double_alloc_type.add_param(desc);
    builder.add_container("double_alloc_1", double_alloc_type, false, true);
    builder.add_container("double_alloc_2", double_alloc_type, false, true);
    types::Function double_copy_in_type(desc);
    double_copy_in_type.add_param(sym_desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    builder.add_container("double_in_1", double_copy_in_type, false, true);
    builder.add_container("double_in_2", double_copy_in_type, false, true);
    types::Function double_kernel_type(void_type);
    double_kernel_type.add_param(sym_desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    builder.add_container("double_kernel", double_kernel_type, false, true);
    types::Function double_copy_out_type(void_type);
    double_copy_out_type.add_param(sym_desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    builder.add_container("double_out_2", double_copy_out_type, false, true);
    types::Function double_free_type(desc);
    double_free_type.add_param(sym_desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    builder.add_container("double_free_1", double_free_type, false, true);
    builder.add_container("double_free_2", double_free_type, false, true);

    auto& block_x_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_x_ref, "xy");
        auto& x_ref = builder.add_access(block_x_ref, "x_ref");
        builder.add_reference_memlet(block_x_ref, xy, x_ref, {symbolic::zero()}, desc2);
    }

    auto& block_x = builder.add_block(root);
    {
        auto& x_ref = builder.add_access(block_x, "x_ref");
        auto& x = builder.add_access(block_x, "x");
        builder.add_dereference_memlet(block_x, x_ref, x, true, desc2);
    }

    auto& block_y_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_y_ref, "xy");
        auto& y_ref = builder.add_access(block_y_ref, "y");
        builder.add_reference_memlet(block_y_ref, xy, y_ref, {symbolic::one()}, desc2);
    }

    auto& block_y = builder.add_block(root);
    {
        auto& y_ref = builder.add_access(block_y, "y_ref");
        auto& y = builder.add_access(block_y, "y");
        builder.add_dereference_memlet(block_y, y_ref, y, true, desc2);
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_x, "n");
        auto& x = builder.add_access(block_in_x, "x");
        auto& y = builder.add_constant(block_in_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_x, device_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_ret", device_x_out, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_y, "n");
        auto& x = builder.add_constant(block_in_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y_in = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_y, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_y, device_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_ret", device_y_out, {}, desc);
    }

    auto& block = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block, "n");
        auto& x = builder.add_constant(block, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& device_x_in = builder.add_access(block, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block, "__daisy_inline_x");
        auto& device_y_in = builder.add_access(block, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "double_kernel",
            {"_arg2", "_arg3", "_arg4"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, device_x_in, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block, device_y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", y_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg3", device_x_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg4", device_y_out, {}, desc);
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_x, "n");
        auto& x = builder.add_access(block_out_x, "x");
        auto& y = builder.add_constant(block_out_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::NONE,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_y, "n");
        auto& x = builder.add_constant(block_out_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block_out_y, "y");
        auto& y_out = builder.add_access(block_out_y, "y");
        auto& device_y = builder.add_access(block_out_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_y, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_arg2", y_out, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 7);
    EXPECT_EQ(while_body.size(), 4);
}

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentCUDA_dependency1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_cuda_x", desc);
    builder.add_container("__daisy_cuda_y", desc);

    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");

    auto& block_n = builder.add_block(root);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& x = builder.add_access(block_in_x, "x");
        auto& device_x = builder.add_access(block_in_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_x, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_dst", device_x, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y = builder.add_access(block_in_y, "__daisy_cuda_y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_y, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_dst", device_y, {}, desc);
    }

    auto& map =
        builder
            .add_map(while_body, i, symbolic::Lt(i, n), zero, symbolic::add(i, one), cuda::ScheduleType_CUDA::create());
    auto& map_body = map.root();

    auto& block = builder.add_block(map_body);
    {
        auto& two = builder.add_constant(block, "2.0f", base_desc);
        auto& device_x = builder.add_access(block, "__daisy_cuda_x");
        auto& device_y = builder.add_access(block, "__daisy_cuda_y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, two, tasklet, "_in1", {});
        builder.add_computational_memlet(block, device_x, tasklet, "_in2", {i});
        builder.add_computational_memlet(block, tasklet, "_out", device_y, {i});
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_x, DebugInfo(), n, zero, memory::DataTransferDirection::NONE, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& device_y = builder.add_access(block_out_y, "__daisy_cuda_y");
        auto& y = builder.add_access(block_out_y, "y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_y, DebugInfo(), n, zero, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_dst", y, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 4);
    EXPECT_EQ(while_body.size(), 4);
    EXPECT_EQ(&root.at(0).first, &block_n);
}

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentCUDA_dependency2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_cuda_x", desc);
    builder.add_container("__daisy_cuda_y", desc);

    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_n = builder.add_block(while_body);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& x = builder.add_access(block_in_x, "x");
        auto& device_x = builder.add_access(block_in_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_x, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_dst", device_x, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y = builder.add_access(block_in_y, "__daisy_cuda_y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_y, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_dst", device_y, {}, desc);
    }

    auto& map =
        builder
            .add_map(while_body, i, symbolic::Lt(i, n), zero, symbolic::add(i, one), cuda::ScheduleType_CUDA::create());
    auto& map_body = map.root();

    auto& block = builder.add_block(map_body);
    {
        auto& two = builder.add_constant(block, "2.0f", base_desc);
        auto& device_x = builder.add_access(block, "__daisy_cuda_x");
        auto& device_y = builder.add_access(block, "__daisy_cuda_y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, two, tasklet, "_in1", {});
        builder.add_computational_memlet(block, device_x, tasklet, "_in2", {i});
        builder.add_computational_memlet(block, tasklet, "_out", device_y, {i});
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_x, DebugInfo(), n, zero, memory::DataTransferDirection::NONE, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& device_y = builder.add_access(block_out_y, "__daisy_cuda_y");
        auto& y = builder.add_access(block_out_y, "y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_y, DebugInfo(), n, zero, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_dst", y, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(while_body.size(), 7);
    EXPECT_EQ(&while_body.at(0).first, &block_n);
}

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentExternal_dependency1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_inline_x", desc);
    builder.add_container("__daisy_inline_y", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Pointer opaque_ptr;
    types::Function double_alloc_type(desc);
    double_alloc_type.add_param(sym_desc);
    double_alloc_type.add_param(desc);
    double_alloc_type.add_param(desc);
    builder.add_container("double_alloc_1", double_alloc_type, false, true);
    builder.add_container("double_alloc_2", double_alloc_type, false, true);
    types::Function double_copy_in_type(desc);
    double_copy_in_type.add_param(sym_desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    builder.add_container("double_in_1", double_copy_in_type, false, true);
    builder.add_container("double_in_2", double_copy_in_type, false, true);
    types::Function double_kernel_type(void_type);
    double_kernel_type.add_param(sym_desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    builder.add_container("double_kernel", double_kernel_type, false, true);
    types::Function double_copy_out_type(void_type);
    double_copy_out_type.add_param(sym_desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    builder.add_container("double_out_2", double_copy_out_type, false, true);
    types::Function double_free_type(desc);
    double_free_type.add_param(sym_desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    builder.add_container("double_free_1", double_free_type, false, true);
    builder.add_container("double_free_2", double_free_type, false, true);

    auto& block_n = builder.add_block(root);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_x, "n");
        auto& x = builder.add_access(block_in_x, "x");
        auto& y = builder.add_constant(block_in_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_x, device_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_ret", device_x_out, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_y, "n");
        auto& x = builder.add_constant(block_in_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y_in = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_y, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_y, device_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_ret", device_y_out, {}, desc);
    }

    auto& block = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block, "n");
        auto& x = builder.add_constant(block, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& device_x_in = builder.add_access(block, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block, "__daisy_inline_x");
        auto& device_y_in = builder.add_access(block, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "double_kernel",
            {"_arg2", "_arg3", "_arg4"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, device_x_in, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block, device_y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", y_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg3", device_x_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg4", device_y_out, {}, desc);
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_x, "n");
        auto& x = builder.add_access(block_out_x, "x");
        auto& y = builder.add_constant(block_out_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::NONE,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_y, "n");
        auto& x = builder.add_constant(block_out_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block_out_y, "y");
        auto& y_out = builder.add_access(block_out_y, "y");
        auto& device_y = builder.add_access(block_out_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_y, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_arg2", y_out, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 4);
    EXPECT_EQ(while_body.size(), 4);
    EXPECT_EQ(&root.at(0).first, &block_n);
}

TEST(ReadonlyTransferHoistingPassTest, HoistArgumentExternal_dependency2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("__daisy_inline_x", desc);
    builder.add_container("__daisy_inline_y", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Pointer opaque_ptr;
    types::Function double_alloc_type(desc);
    double_alloc_type.add_param(sym_desc);
    double_alloc_type.add_param(desc);
    double_alloc_type.add_param(desc);
    builder.add_container("double_alloc_1", double_alloc_type, false, true);
    builder.add_container("double_alloc_2", double_alloc_type, false, true);
    types::Function double_copy_in_type(desc);
    double_copy_in_type.add_param(sym_desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    builder.add_container("double_in_1", double_copy_in_type, false, true);
    builder.add_container("double_in_2", double_copy_in_type, false, true);
    types::Function double_kernel_type(void_type);
    double_kernel_type.add_param(sym_desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    builder.add_container("double_kernel", double_kernel_type, false, true);
    types::Function double_copy_out_type(void_type);
    double_copy_out_type.add_param(sym_desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    builder.add_container("double_out_2", double_copy_out_type, false, true);
    types::Function double_free_type(desc);
    double_free_type.add_param(sym_desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    builder.add_container("double_free_1", double_free_type, false, true);
    builder.add_container("double_free_2", double_free_type, false, true);

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_n = builder.add_block(while_body);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_x, "n");
        auto& x = builder.add_access(block_in_x, "x");
        auto& y = builder.add_constant(block_in_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_x, device_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_ret", device_x_out, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_y, "n");
        auto& x = builder.add_constant(block_in_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y_in = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_y, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_y, device_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_ret", device_y_out, {}, desc);
    }

    auto& block = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block, "n");
        auto& x = builder.add_constant(block, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& device_x_in = builder.add_access(block, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block, "__daisy_inline_x");
        auto& device_y_in = builder.add_access(block, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "double_kernel",
            {"_arg2", "_arg3", "_arg4"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, device_x_in, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block, device_y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", y_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg3", device_x_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg4", device_y_out, {}, desc);
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_x, "n");
        auto& x = builder.add_access(block_out_x, "x");
        auto& y = builder.add_constant(block_out_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::NONE,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_y, "n");
        auto& x = builder.add_constant(block_out_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block_out_y, "y");
        auto& y_out = builder.add_access(block_out_y, "y");
        auto& device_y = builder.add_access(block_out_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_y, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_arg2", y_out, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(while_body.size(), 7);
    EXPECT_EQ(&while_body.at(0).first, &block_n);
}

TEST(ReadonlyTransferHoistingPassTest, HoistLocalUnpackCUDA_dependency) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("xy", desc2, true);
    builder.add_container("x_ref", desc2);
    builder.add_container("y_ref", desc2);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("__daisy_cuda_x", desc);
    builder.add_container("__daisy_cuda_y", desc);

    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");

    auto& block_x_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_x_ref, "xy");
        auto& x_ref = builder.add_access(block_x_ref, "x_ref");
        builder.add_reference_memlet(block_x_ref, xy, x_ref, {zero}, desc2);
    }

    auto& block_x = builder.add_block(root);
    {
        auto& x_ref = builder.add_access(block_x, "x_ref");
        auto& x = builder.add_access(block_x, "x");
        builder.add_dereference_memlet(block_x, x_ref, x, true, desc2);
    }

    auto& block_y_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_y_ref, "xy");
        auto& y_ref = builder.add_access(block_y_ref, "y");
        builder.add_reference_memlet(block_y_ref, xy, y_ref, {one}, desc2);
    }

    auto& block_y = builder.add_block(root);
    {
        auto& y_ref = builder.add_access(block_y, "y_ref");
        auto& y = builder.add_access(block_y, "y");
        builder.add_dereference_memlet(block_y, y_ref, y, true, desc2);
    }

    auto& block_n = builder.add_block(root);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& x = builder.add_access(block_in_x, "x");
        auto& device_x = builder.add_access(block_in_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_x, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, x, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_dst", device_x, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y = builder.add_access(block_in_y, "__daisy_cuda_y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_in_y, DebugInfo(), n, zero, memory::DataTransferDirection::H2D, memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_dst", device_y, {}, desc);
    }

    auto& map =
        builder
            .add_map(while_body, i, symbolic::Lt(i, n), zero, symbolic::add(i, one), cuda::ScheduleType_CUDA::create());
    auto& map_body = map.root();

    auto& block = builder.add_block(map_body);
    {
        auto& two = builder.add_constant(block, "2.0f", base_desc);
        auto& device_x = builder.add_access(block, "__daisy_cuda_x");
        auto& device_y = builder.add_access(block, "__daisy_cuda_y");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, two, tasklet, "_in1", {});
        builder.add_computational_memlet(block, device_x, tasklet, "_in2", {i});
        builder.add_computational_memlet(block, tasklet, "_out", device_y, {i});
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_cuda_x");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_x, DebugInfo(), n, zero, memory::DataTransferDirection::NONE, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& device_y = builder.add_access(block_out_y, "__daisy_cuda_y");
        auto& y = builder.add_access(block_out_y, "y");
        auto& libnode = builder.add_library_node<cuda::CUDAOffloadingNode>(
            block_out_y, DebugInfo(), n, zero, memory::DataTransferDirection::D2H, memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_src", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_dst", y, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 8);
    EXPECT_EQ(while_body.size(), 4);
    EXPECT_EQ(&root.at(4).first, &block_n);
}

TEST(ReadonlyTransferHoistingPassTest, HoistLocalUnpackExternal_dependency) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("xy", desc2, true);
    builder.add_container("x_ref", desc2);
    builder.add_container("y_ref", desc2);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("__daisy_inline_x", desc);
    builder.add_container("__daisy_inline_y", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Pointer opaque_ptr;
    types::Function double_alloc_type(desc);
    double_alloc_type.add_param(sym_desc);
    double_alloc_type.add_param(desc);
    double_alloc_type.add_param(desc);
    builder.add_container("double_alloc_1", double_alloc_type, false, true);
    builder.add_container("double_alloc_2", double_alloc_type, false, true);
    types::Function double_copy_in_type(desc);
    double_copy_in_type.add_param(sym_desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    double_copy_in_type.add_param(desc);
    builder.add_container("double_in_1", double_copy_in_type, false, true);
    builder.add_container("double_in_2", double_copy_in_type, false, true);
    types::Function double_kernel_type(void_type);
    double_kernel_type.add_param(sym_desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    double_kernel_type.add_param(desc);
    builder.add_container("double_kernel", double_kernel_type, false, true);
    types::Function double_copy_out_type(void_type);
    double_copy_out_type.add_param(sym_desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    double_copy_out_type.add_param(desc);
    builder.add_container("double_out_2", double_copy_out_type, false, true);
    types::Function double_free_type(desc);
    double_free_type.add_param(sym_desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    double_free_type.add_param(desc);
    builder.add_container("double_free_1", double_free_type, false, true);
    builder.add_container("double_free_2", double_free_type, false, true);

    auto& block_x_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_x_ref, "xy");
        auto& x_ref = builder.add_access(block_x_ref, "x_ref");
        builder.add_reference_memlet(block_x_ref, xy, x_ref, {symbolic::zero()}, desc2);
    }

    auto& block_x = builder.add_block(root);
    {
        auto& x_ref = builder.add_access(block_x, "x_ref");
        auto& x = builder.add_access(block_x, "x");
        builder.add_dereference_memlet(block_x, x_ref, x, true, desc2);
    }

    auto& block_y_ref = builder.add_block(root);
    {
        auto& xy = builder.add_access(block_y_ref, "xy");
        auto& y_ref = builder.add_access(block_y_ref, "y");
        builder.add_reference_memlet(block_y_ref, xy, y_ref, {symbolic::one()}, desc2);
    }

    auto& block_y = builder.add_block(root);
    {
        auto& y_ref = builder.add_access(block_y, "y_ref");
        auto& y = builder.add_access(block_y, "y");
        builder.add_dereference_memlet(block_y, y_ref, y, true, desc2);
    }

    auto& block_n = builder.add_block(root);
    {
        auto& m = builder.add_access(block_n, "m");
        auto& n = builder.add_access(block_n, "n");
        auto& tasklet = builder.add_tasklet(block_n, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_n, m, tasklet, "_in", {});
        builder.add_computational_memlet(block_n, tasklet, "_out", n, {});
    }

    auto& while_loop = builder.add_while(root);
    auto& while_body = while_loop.root();

    auto& block_in_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_x, "n");
        auto& x = builder.add_access(block_in_x, "x");
        auto& y = builder.add_constant(block_in_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_in_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_x, device_x_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_x, libnode, "_ret", device_x_out, {}, desc);
    }

    auto& block_in_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_in_y, "n");
        auto& x = builder.add_constant(block_in_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y = builder.add_access(block_in_y, "y");
        auto& device_y_in = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block_in_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_in_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::H2D,
            memory::BufferLifecycle::ALLOC
        );
        builder.add_computational_memlet(block_in_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_in_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_in_y, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_in_y, device_y_in, libnode, "_ret", {}, desc);
        builder.add_computational_memlet(block_in_y, libnode, "_ret", device_y_out, {}, desc);
    }

    auto& block = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block, "n");
        auto& x = builder.add_constant(block, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block, "y");
        auto& y_out = builder.add_access(block, "y");
        auto& device_x_in = builder.add_access(block, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block, "__daisy_inline_x");
        auto& device_y_in = builder.add_access(block, "__daisy_inline_y");
        auto& device_y_out = builder.add_access(block, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            data_flow::CallNode,
            const std::string&,
            const std::vector<std::string>&,
            const std::vector<std::string>&>(
            block,
            DebugInfo(),
            "double_kernel",
            {"_arg2", "_arg3", "_arg4"},
            {"_arg0", "_arg1", "_arg2", "_arg3", "_arg4"}
        );
        builder.add_computational_memlet(block, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block, device_x_in, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block, device_y_in, libnode, "_arg4", {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg2", y_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg3", device_x_out, {}, desc);
        builder.add_computational_memlet(block, libnode, "_arg4", device_y_out, {}, desc);
    }

    auto& block_out_x = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_x, "n");
        auto& x = builder.add_access(block_out_x, "x");
        auto& y = builder.add_constant(block_out_x, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& device_x_in = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& device_x_out = builder.add_access(block_out_x, "__daisy_inline_x");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_x,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            1,
            memory::DataTransferDirection::NONE,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_x, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_x, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_x, y, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_x, device_x_in, libnode, "_ptr", {}, desc);
        builder.add_computational_memlet(block_out_x, libnode, "_ptr", device_x_out, {}, desc);
    }

    auto& block_out_y = builder.add_block(while_body);
    {
        auto& n = builder.add_access(block_out_y, "n");
        auto& x = builder.add_constant(block_out_y, symbolic::__nullptr__()->__str__(), opaque_ptr);
        auto& y_in = builder.add_access(block_out_y, "y");
        auto& y_out = builder.add_access(block_out_y, "y");
        auto& device_y = builder.add_access(block_out_y, "__daisy_inline_y");
        auto& libnode = builder.add_library_node<
            memory::ExternalOffloadingNode,
            const std::vector<std::string>&,
            const std::string&,
            size_t,
            memory::DataTransferDirection,
            memory::BufferLifecycle>(
            block_out_y,
            DebugInfo(),
            {"_arg0", "_arg1", "_arg2"},
            "double",
            2,
            memory::DataTransferDirection::D2H,
            memory::BufferLifecycle::FREE
        );
        builder.add_computational_memlet(block_out_y, n, libnode, "_arg0", {}, sym_desc);
        builder.add_computational_memlet(block_out_y, x, libnode, "_arg1", {}, desc);
        builder.add_computational_memlet(block_out_y, y_in, libnode, "_arg2", {}, desc);
        builder.add_computational_memlet(block_out_y, device_y, libnode, "_arg3", {}, desc);
        builder.add_computational_memlet(block_out_y, libnode, "_arg2", y_out, {}, desc);
    }

    builder.add_break(while_body);

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::ReadonlyTransferHoistingPass readonly_transfer_hoisting_pass;
    EXPECT_TRUE(readonly_transfer_hoisting_pass.run(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 8);
    EXPECT_EQ(while_body.size(), 4);
    EXPECT_EQ(&root.at(4).first, &block_n);
}

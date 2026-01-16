#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/cuda/codegen/cuda_map_dispatcher.h"
#include "sdfg/cuda/cuda.h"
#include "sdfg/cuda/nodes/cuda_data_offloading_node.h"

namespace sdfg {
namespace cuda {

inline void register_cuda_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_CUDA::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<CUDAMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        cuda::LibraryNodeType_CUDA_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                cuda::CUDADataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(cuda::LibraryNodeType_CUDA_Offloading.value(), []() {
            return std::make_unique<cuda::CUDADataOffloadingNodeSerializer>();
        });
}

} // namespace cuda
} // namespace sdfg

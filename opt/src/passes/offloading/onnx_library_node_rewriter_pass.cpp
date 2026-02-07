#include "sdfg/passes/offloading/onnx_library_node_rewriter_pass.h"

#include "sdfg/data_flow/library_nodes/math/tensor/broadcast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/transpose_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace passes {

ONNXLibraryNodeRewriter::
    ONNXLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool ONNXLibraryNodeRewriter::visit() {
    DEBUG_PRINTLN("Running ONNXLibraryNodeRewriterPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool ONNXLibraryNodeRewriter::accept(structured_control_flow::Block& block) {
    bool applied = false;

    for (auto* lib_node : block.dataflow().library_nodes()) {
        // Check if this is a tensor node by attempting dynamic_cast
        // TensorNode is the base class for all tensor operations
        auto* tensor_node = dynamic_cast<math::tensor::TensorNode*>(lib_node);

        if (tensor_node != nullptr) {
            // Set implementation type to ONNX
            tensor_node->implementation_type() = onnx::tensor::ImplementationType_ONNX;
            applied = true;
        }
    }

    return applied;
}

} // namespace passes
} // namespace sdfg

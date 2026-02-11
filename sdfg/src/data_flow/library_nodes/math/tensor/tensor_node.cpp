#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/types/tensor.h"

namespace sdfg {
namespace math {
namespace tensor {

TensorNode::TensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    data_flow::ImplementationType impl_type
)
    : MathNode(element_id, debug_info, vertex, parent, code, outputs, inputs, impl_type) {}

void TensorNode::validate(const Function& function) const {
    MathNode::validate(function);

    auto& graph = this->get_parent();

    // Check that all input memlets are tensors
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Tensor &&
            iedge.base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "TensorNode: Input memlet must be of tensor or scalar type. Found type: " + iedge.base_type().print()
            );
        }
    }

    // Check that all output memlets are tensors
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Tensor &&
            oedge.base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "TensorNode: Output memlet must be of tensor or scalar type. Found type: " + oedge.base_type().print()
            );
        }
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg

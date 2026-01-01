#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

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
    auto& graph = this->get_parent();

    // Get the set of actually connected input connectors
    std::set<std::string> connected_inputs;
    for (auto& iedge : graph.in_edges(*this)) {
        connected_inputs.insert(iedge.dst_conn());
    }

    // Check that all input memlets are scalar or pointer of scalar
    for (auto& iedge : graph.in_edges(*this)) {
        // Skip validation for optional inputs that aren't connected
        // (this check is redundant here since we iterate over connected edges,
        // but kept for clarity)
        
        if (iedge.base_type().type_id() != types::TypeID::Scalar &&
            iedge.base_type().type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException(
                "TensorNode: Input memlet must be of scalar or pointer type. Found type: " + iedge.base_type().print()
            );
        }
        if (iedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(iedge.base_type());
            if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException(
                    "TensorNode: Input memlet pointer must be flat (pointer to scalar). Found type: " +
                    ptr_type.pointee_type().print()
                );
            }
            if (!iedge.subset().empty()) {
                throw InvalidSDFGException("TensorNode: Input memlet pointer must not be dereferenced.");
            }
        }
    }

    // Check that all required inputs are connected
    for (const auto& input_name : inputs_) {
        // Skip optional inputs
        if (is_input_optional(input_name)) {
            continue;
        }
        
        // Check if this required input is connected
        if (connected_inputs.find(input_name) == connected_inputs.end()) {
            throw InvalidSDFGException(
                "TensorNode: Required input '" + input_name + "' is not connected"
            );
        }
    }

    // Check that all output memlets are scalar or pointer of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Scalar &&
            oedge.base_type().type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException(
                "TensorNode: Output memlet must be of scalar or pointer type. Found type: " + oedge.base_type().print()
            );
        }
        if (oedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(oedge.base_type());
            if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException(
                    "TensorNode: Output memlet pointer must be flat (pointer to scalar). Found type: " +
                    ptr_type.pointee_type().print()
                );
            }
            if (!oedge.subset().empty()) {
                throw InvalidSDFGException("TensorNode: Output memlet pointer must not be dereferenced.");
            }
        }
    }

    // Validate that all memlets have the same primitive type
    types::PrimitiveType prim_type = primitive_type(graph);

    // Check if this operation supports integer types
    if (!supports_integer_types() && types::is_integer(prim_type)) {
        throw InvalidSDFGException(
            "TensorNode: This operation does not support integer types. Found type: " +
            std::string(types::primitive_type_to_string(prim_type))
        );
    }
}

types::PrimitiveType TensorNode::primitive_type(const data_flow::DataFlowGraph& graph) const {
    types::PrimitiveType result_type = types::PrimitiveType::Void;
    bool first = true;

    // Check all input edges
    for (auto& iedge : graph.in_edges(*this)) {
        types::PrimitiveType edge_type;
        if (iedge.base_type().type_id() == types::TypeID::Scalar) {
            auto& scalar_type = static_cast<const types::Scalar&>(iedge.base_type());
            edge_type = scalar_type.primitive_type();
        } else if (iedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(iedge.base_type());
            auto& pointee = ptr_type.pointee_type();
            if (pointee.type_id() == types::TypeID::Scalar) {
                auto& scalar_type = static_cast<const types::Scalar&>(pointee);
                edge_type = scalar_type.primitive_type();
            } else {
                throw InvalidSDFGException("TensorNode: Pointer must point to scalar type");
            }
        } else {
            throw InvalidSDFGException("TensorNode: Edge must be scalar or pointer type");
        }

        if (first) {
            result_type = edge_type;
            first = false;
        } else if (result_type != edge_type) {
            throw InvalidSDFGException(
                "TensorNode: All input memlets must have the same primitive type. Found " +
                std::string(types::primitive_type_to_string(result_type)) + " and " +
                std::string(types::primitive_type_to_string(edge_type))
            );
        }
    }

    // Check all output edges
    for (auto& oedge : graph.out_edges(*this)) {
        types::PrimitiveType edge_type;
        if (oedge.base_type().type_id() == types::TypeID::Scalar) {
            auto& scalar_type = static_cast<const types::Scalar&>(oedge.base_type());
            edge_type = scalar_type.primitive_type();
        } else if (oedge.base_type().type_id() == types::TypeID::Pointer) {
            auto& ptr_type = static_cast<const types::Pointer&>(oedge.base_type());
            auto& pointee = ptr_type.pointee_type();
            if (pointee.type_id() == types::TypeID::Scalar) {
                auto& scalar_type = static_cast<const types::Scalar&>(pointee);
                edge_type = scalar_type.primitive_type();
            } else {
                throw InvalidSDFGException("TensorNode: Pointer must point to scalar type");
            }
        } else {
            throw InvalidSDFGException("TensorNode: Edge must be scalar or pointer type");
        }

        if (first) {
            result_type = edge_type;
            first = false;
        } else if (result_type != edge_type) {
            throw InvalidSDFGException(
                "TensorNode: All output memlets must have the same primitive type. Found " +
                std::string(types::primitive_type_to_string(result_type)) + " and " +
                std::string(types::primitive_type_to_string(edge_type))
            );
        }
    }

    if (first) {
        throw InvalidSDFGException("TensorNode: No edges found to determine primitive type");
    }

    return result_type;
}

data_flow::TaskletCode TensorNode::get_integer_minmax_tasklet(types::PrimitiveType prim_type, bool is_max) {
    bool is_signed = types::is_signed(prim_type);
    if (is_max) {
        return is_signed ? data_flow::TaskletCode::int_smax : data_flow::TaskletCode::int_umax;
    } else {
        return is_signed ? data_flow::TaskletCode::int_smin : data_flow::TaskletCode::int_umin;
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg

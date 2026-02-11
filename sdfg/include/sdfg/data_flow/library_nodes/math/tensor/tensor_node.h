#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

class TensorNode : public math::MathNode {
public:
    TensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        data_flow::ImplementationType impl_type
    );

    void validate(const Function& function) const override;

    virtual bool supports_integer_types() const = 0;
};

} // namespace tensor
} // namespace math
} // namespace sdfg

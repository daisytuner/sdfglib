#pragma once

#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace deepcopy {

class StructuredSDFGDeepCopy {
private:
    builder::StructuredSDFGBuilder& builder_;
    structured_control_flow::Sequence& root_;
    structured_control_flow::ControlFlowNode& source_;

    std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>
        node_mapping;

    void append(structured_control_flow::Sequence& root, structured_control_flow::Sequence& source);

    void insert(structured_control_flow::Sequence& root, structured_control_flow::ControlFlowNode& source);

public:
    StructuredSDFGDeepCopy(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& root,
        structured_control_flow::ControlFlowNode& source
    );

    std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>
    copy();

    std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>
    insert();
};

} // namespace deepcopy
} // namespace sdfg

#pragma once

#include <memory>

#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {

class StructuredSDFG;

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class While;
class StructuredLoop;
class Sequence;

class Transition : public Element {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Sequence* parent_;
    control_flow::Assignments assignments_;

    Transition(size_t element_id, const DebugInfo& debug_info, Sequence& parent);

    Transition(
        size_t element_id, const DebugInfo& debug_info, Sequence& parent, const control_flow::Assignments& assignments
    );

public:
    Transition(const Transition& node) = delete;
    Transition& operator=(const Transition&) = delete;

    void validate(const Function& function) const override;

    const control_flow::Assignments& assignments() const;

    control_flow::Assignments& assignments();

    Sequence& parent();

    const Sequence& parent() const;

    bool empty() const;

    size_t size() const;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

class Sequence : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

    friend class sdfg::StructuredSDFG;

    friend class sdfg::structured_control_flow::While;
    friend class sdfg::structured_control_flow::StructuredLoop;

private:
    std::vector<std::unique_ptr<ControlFlowNode>> children_;
    std::vector<std::unique_ptr<Transition>> transitions_;

    Sequence(size_t element_id, const DebugInfo& debug_info);

public:
    Sequence(const Sequence& node) = delete;
    Sequence& operator=(const Sequence&) = delete;

    void validate(const Function& function) const override;

    size_t size() const;

    std::pair<const ControlFlowNode&, const Transition&> at(size_t i) const;

    std::pair<ControlFlowNode&, Transition&> at(size_t i);

    int index(const ControlFlowNode& child) const;

    int index(const Transition& transition) const;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace structured_control_flow
} // namespace sdfg

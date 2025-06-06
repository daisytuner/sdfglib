#pragma once

#include <list>
#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

class StructuredSDFG;

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class While;
class For;
class Map;

class Transition : public Element {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Assignments assignments_;

    Transition(const DebugInfo& debug_info);

    Transition(const DebugInfo& debug_info, const symbolic::Assignments& assignments);

   public:
    Transition(const Transition& node) = delete;
    Transition& operator=(const Transition&) = delete;

    const symbolic::Assignments& assignments() const;

    symbolic::Assignments& assignments();

    bool empty() const;

    size_t size() const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

class Sequence : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

    friend class sdfg::StructuredSDFG;

    friend class sdfg::structured_control_flow::While;
    friend class sdfg::structured_control_flow::For;
    friend class sdfg::structured_control_flow::Map;

   private:
    std::vector<std::unique_ptr<ControlFlowNode>> children_;
    std::vector<std::unique_ptr<Transition>> transitions_;

    Sequence(const DebugInfo& debug_info);

   public:
    Sequence(const Sequence& node) = delete;
    Sequence& operator=(const Sequence&) = delete;

    size_t size() const;

    std::pair<const ControlFlowNode&, const Transition&> at(size_t i) const;

    std::pair<ControlFlowNode&, Transition&> at(size_t i);

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg

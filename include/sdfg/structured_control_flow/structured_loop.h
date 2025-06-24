#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class StructuredLoop : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   protected:
    symbolic::Symbol indvar_;
    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    StructuredLoop(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
                   symbolic::Expression init, symbolic::Expression update,
                   symbolic::Condition condition);

   public:
    virtual ~StructuredLoop() = default;

    StructuredLoop(const StructuredLoop& node) = delete;
    StructuredLoop& operator=(const StructuredLoop&) = delete;

    const symbolic::Symbol& indvar() const;

    symbolic::Symbol& indvar();

    const symbolic::Expression& init() const;

    symbolic::Expression& init();

    const symbolic::Expression& update() const;

    symbolic::Expression& update();

    const symbolic::Condition& condition() const;

    symbolic::Condition& condition();

    Sequence& root() const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression);
};

}  // namespace structured_control_flow
}  // namespace sdfg
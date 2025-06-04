#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class For : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Symbol indvar_;
    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    For(const DebugInfo& debug_info, symbolic::Symbol indvar, symbolic::Expression init,
        symbolic::Expression update, symbolic::Condition condition);

   public:
    For(const For& node) = delete;
    For& operator=(const For&) = delete;

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
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class For : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Symbol indvar_;
    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    For(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
        symbolic::Expression init, symbolic::Expression update, symbolic::Condition condition);

   public:
    For(const For& node) = delete;
    For& operator=(const For&) = delete;

    const symbolic::Symbol& indvar() const override;

    symbolic::Symbol& indvar();

    const symbolic::Expression& init() const override;

    symbolic::Expression& init();

    const symbolic::Expression& update() const override;

    symbolic::Expression& update();

    const symbolic::Condition& condition() const override;

    symbolic::Condition& condition();

    Sequence& root() const override;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
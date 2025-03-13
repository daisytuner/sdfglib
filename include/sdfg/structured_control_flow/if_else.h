#pragma once

#include <memory>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class IfElse : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    std::vector<std::unique_ptr<Sequence>> cases_;
    std::vector<symbolic::Condition> conditions_;

    IfElse(size_t element_id, const DebugInfo& debug_info);

   public:
    IfElse(const IfElse& node) = delete;
    IfElse& operator=(const IfElse&) = delete;

    size_t size() const;

    std::pair<const Sequence&, const symbolic::Condition&> at(size_t i) const;

    std::pair<Sequence&, symbolic::Condition&> at(size_t i);

    bool is_complete();

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
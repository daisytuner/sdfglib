#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class ForEach : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    symbolic::Symbol iterator_;
    symbolic::Symbol end_;

    std::unique_ptr<Sequence> root_;

    ForEach(
        size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol iterator,
        symbolic::Symbol end
    );

public:
    virtual ~ForEach() = default;

    ForEach(const ForEach& node) = delete;
    ForEach& operator=(const ForEach&) = delete;

    void validate(const Function& function) const override;

    const symbolic::Symbol iterator() const;

    const symbolic::Symbol end() const;

    Sequence& root() const;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

};

} // namespace structured_control_flow
} // namespace sdfg

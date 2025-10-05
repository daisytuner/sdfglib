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

class While : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::unique_ptr<Sequence> root_;

    While(size_t element_id, const DebugInfo& debug_info);

public:
    While(const While& node) = delete;
    While& operator=(const While&) = delete;

    void validate(const Function& function) const override;

    const Sequence& root() const;

    Sequence& root();

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

class Break : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Break(size_t element_id, const DebugInfo& debug_info);

public:
    void validate(const Function& function) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

class Continue : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Continue(size_t element_id, const DebugInfo& debug_info);

public:
    void validate(const Function& function) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace structured_control_flow
} // namespace sdfg

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

    While(const DebugInfo& debug_info);

   public:
    While(const While& node) = delete;
    While& operator=(const While&) = delete;

    const Sequence& root() const;

    Sequence& root();

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

class Break : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    Break(const DebugInfo& debug_info);

   public:
    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

class Continue : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    Continue(const DebugInfo& debug_info);

   public:
    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
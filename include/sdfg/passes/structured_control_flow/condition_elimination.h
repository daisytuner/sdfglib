#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class ConditionElimination : public visitor::StructuredSDFGVisitor {
private:
    bool eliminate_condition(
        structured_control_flow::Sequence& root,
        structured_control_flow::IfElse& match,
        structured_control_flow::StructuredLoop& loop,
        const symbolic::Condition& condition
    );

public:
    ConditionElimination(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& node) override;
};

typedef VisitorPass<ConditionElimination> ConditionEliminationPass;

} // namespace passes
} // namespace sdfg

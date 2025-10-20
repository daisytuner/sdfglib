#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace visitor {

class StructuredSDFGVisitor {
protected:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;

    virtual bool visit_internal(structured_control_flow::Sequence& parent);

public:
    StructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    virtual ~StructuredSDFGVisitor() = default;

    bool visit();

    virtual bool accept(structured_control_flow::Block& node);

    virtual bool accept(structured_control_flow::Sequence& node);

    virtual bool accept(structured_control_flow::Return& node);

    virtual bool accept(structured_control_flow::IfElse& node);

    virtual bool accept(structured_control_flow::For& node);

    virtual bool accept(structured_control_flow::While& node);

    virtual bool accept(structured_control_flow::Continue& node);

    virtual bool accept(structured_control_flow::Break& node);

    virtual bool accept(structured_control_flow::Map& node);
};

class NonStoppingStructuredSDFGVisitor : public StructuredSDFGVisitor {
private:
    bool applied_;

protected:
    bool visit_internal(structured_control_flow::Sequence& parent) override;

public:
    NonStoppingStructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);
};

} // namespace visitor
} // namespace sdfg

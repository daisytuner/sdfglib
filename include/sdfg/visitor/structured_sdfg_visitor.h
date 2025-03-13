#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace visitor {

class StructuredSDFGVisitor {
   private:
    bool visit(structured_control_flow::Sequence& parent);

   protected:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;

   public:
    StructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder,
                          analysis::AnalysisManager& analysis_manager);

    bool visit();

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Block& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Sequence& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Return& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::IfElse& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::For& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::While& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Continue& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Break& node);

    virtual bool accept(structured_control_flow::Sequence& parent,
                        structured_control_flow::Kernel& node);
};

}  // namespace visitor
}  // namespace sdfg

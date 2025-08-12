#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class For2Map : public visitor::StructuredSDFGVisitor {
public:
    For2Map(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::For& node) override;

private:
    bool can_be_applied(structured_control_flow::For& for_stmt, analysis::AnalysisManager& analysis_manager);
    void apply(
        structured_control_flow::For& for_stmt,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager
    );
};

typedef VisitorPass<For2Map> For2MapPass;

} // namespace passes
} // namespace sdfg

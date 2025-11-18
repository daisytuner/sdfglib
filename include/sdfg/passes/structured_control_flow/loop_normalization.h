#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class LoopNormalization : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    LoopNormalization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "LoopNormalization"; };

    bool accept(structured_control_flow::For& node) override;
};

typedef VisitorPass<LoopNormalization> LoopNormalizationPass;

} // namespace passes
} // namespace sdfg

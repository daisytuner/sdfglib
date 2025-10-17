#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class Expansion : public visitor::StructuredSDFGVisitor {
public:
    Expansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "Expansion"; };

    bool accept(structured_control_flow::Block& node) override;
};

typedef VisitorPass<Expansion> ExpansionPass;

} // namespace passes
} // namespace sdfg

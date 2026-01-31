#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"


namespace sdfg {
namespace passes {

class GemmExpansion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    GemmExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "GemmExpansion"; }
    virtual bool visit() override;

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<GemmExpansion> GemmExpansionPass;

} // namespace passes
} // namespace sdfg

#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class AllocationHoisting : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool can_be_applied(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

    void apply(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

public:
    AllocationHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "AllocationHoisting"; };

    bool accept(structured_control_flow::Block& node) override;
};

typedef VisitorPass<AllocationHoisting> AllocationHoistingPass;

} // namespace passes
} // namespace sdfg

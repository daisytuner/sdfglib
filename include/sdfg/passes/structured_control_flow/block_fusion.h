#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockFusion : public visitor::StructuredSDFGVisitor {
   private:
    bool can_be_applied(data_flow::DataFlowGraph& first_graph,
                        control_flow::Assignments& first_assignments,
                        data_flow::DataFlowGraph& second_graph,
                        control_flow::Assignments& second_assignments);

    void apply(structured_control_flow::Block& first_block,
               control_flow::Assignments& first_assignments,
               structured_control_flow::Block& second_block,
               control_flow::Assignments& second_assignments);

   public:
    BlockFusion(builder::StructuredSDFGBuilder& builder,
                analysis::AnalysisManager& analysis_manager);

    bool accept(structured_control_flow::Sequence& parent,
                structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<BlockFusion> BlockFusionPass;

}  // namespace passes
}  // namespace sdfg

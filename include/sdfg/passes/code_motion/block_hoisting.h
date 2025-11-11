#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockHoisting : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool is_invariant_move(structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg);
    bool is_invariant_view(structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg);

    bool equal_moves(structured_control_flow::Block& block1, structured_control_flow::Block& block2);
    bool equal_views(structured_control_flow::Block& block1, structured_control_flow::Block& block2);

    bool map_invariant_move(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );

    bool map_invariant_view(
        structured_control_flow::Sequence& parent,
        structured_control_flow::Map& map_stmt,
        structured_control_flow::Block& block
    );

    bool for_invariant_move(
        structured_control_flow::Sequence& parent,
        structured_control_flow::For& for_stmt,
        structured_control_flow::Block& block
    );

    bool for_invariant_view(
        structured_control_flow::Sequence& parent,
        structured_control_flow::For& for_stmt,
        structured_control_flow::Block& block
    );

    void if_else_extract_invariant(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else);

public:
    BlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "BlockHoisting"; }

    virtual bool accept(structured_control_flow::Map& map_stmt) override;

    virtual bool accept(structured_control_flow::For& for_stmt) override;

    virtual bool accept(structured_control_flow::IfElse& if_else) override;
};

typedef VisitorPass<BlockHoisting> BlockHoistingPass;

} // namespace passes
} // namespace sdfg

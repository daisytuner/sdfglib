#pragma once

#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockSorting : public visitor::StructuredSDFGVisitor {
private:
    bool is_reference_block(structured_control_flow::Block& block);

    bool is_libnode_block(structured_control_flow::Block& next_block);

protected:
    virtual bool is_libnode_side_effect_white_listed(data_flow::LibraryNode* libnode);

    virtual bool can_be_bubbled_up(structured_control_flow::Block& block);
    virtual bool can_be_bubbled_down(structured_control_flow::Block& block);

    virtual std::pair<int, std::string> get_prio_and_order(structured_control_flow::ControlFlowNode& node);

public:
    BlockSorting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "BlockSorting"; }

    virtual bool accept(structured_control_flow::Sequence& sequence_stmt) override;
};

typedef VisitorPass<BlockSorting> BlockSortingPass;

} // namespace passes
} // namespace sdfg

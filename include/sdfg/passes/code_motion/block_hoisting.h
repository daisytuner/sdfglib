#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockHoisting : public visitor::NonStoppingStructuredSDFGVisitor {
private:
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

public:
    BlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "BlockHoisting"; }

    virtual bool accept(structured_control_flow::Map& map_stmt) override;

    virtual bool accept(structured_control_flow::For& for_stmt) override;
};

typedef VisitorPass<BlockHoisting> BlockHoistingPass;

} // namespace passes
} // namespace sdfg

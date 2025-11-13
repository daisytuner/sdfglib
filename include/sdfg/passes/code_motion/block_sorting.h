#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockSorting : public visitor::StructuredSDFGVisitor {
private:
    bool is_reference_block(structured_control_flow::Block& next_block);

    bool is_libnode_block(structured_control_flow::Block& next_block);

public:
    BlockSorting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "BlockSorting"; }

    virtual bool accept(structured_control_flow::Sequence& sequence_stmt) override;
};

typedef VisitorPass<BlockSorting> BlockSortingPass;

} // namespace passes
} // namespace sdfg

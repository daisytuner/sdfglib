#include "sdfg/passes/dot_expansion_pass.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/while.h"


namespace sdfg {
namespace passes {

DotExpansion::DotExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool DotExpansion::visit() {
    DEBUG_PRINTLN("Running DotExpansionPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool DotExpansion::accept(structured_control_flow::Block& block) {
    bool expanded = false;

    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();
    for (auto lib_node : block.dataflow().library_nodes()) {
        std::cerr << "DotExpansionPass visiting libnode " << lib_node->code().value() << "\n";
        if (lib_node->code() == math::blas::LibraryNodeType_DOT) {
            auto dot_node = static_cast<math::blas::DotNode*>(lib_node);
            if (dot_node->expand(builder_, analysis_manager_)) {
                std::cerr << "  expanded!\n";
                return true;
            }
        }
    }

    return expanded;
}

} // namespace passes
} // namespace sdfg

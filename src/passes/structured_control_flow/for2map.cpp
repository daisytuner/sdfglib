#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pipeline.h"

namespace sdfg {
namespace passes {

For2Map::For2Map(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {

      };

bool For2Map::can_be_applied(structured_control_flow::For& for_stmt, analysis::AnalysisManager& analysis_manager) {
    // Criterion: loop must be data-parallel w.r.t containers
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto dependencies = data_dependency_analysis.dependencies(for_stmt);

    // a. No true dependencies (RAW) between iterations
    for (auto& dep : dependencies) {
        if (dep.second == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
            return false;
        }
    }

    // b. False dependencies (WAW) are limited to loop-local variables
    auto& users = analysis_manager.get<analysis::Users>();
    auto locals = users.locals(for_stmt.root());
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        if (locals.find(container) == locals.end()) {
            return false;
        }
    }

    // c. indvar not used after for
    if (locals.find(for_stmt.indvar()->get_name()) != locals.end()) {
        return false;
    }

    return true;
}

void For2Map::apply(
    structured_control_flow::For& for_stmt,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&for_stmt));

    // convert for to map
    builder.convert_for(*parent, for_stmt);

    analysis_manager.invalidate_all();
}

bool For2Map::accept(structured_control_flow::For& node) {
    if (!this->can_be_applied(node, analysis_manager_)) {
        return false;
    }

    this->apply(node, builder_, analysis_manager_);
    return true;
}

} // namespace passes
} // namespace sdfg

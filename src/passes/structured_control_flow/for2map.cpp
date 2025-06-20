#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace passes {

For2Map::For2Map(builder::StructuredSDFGBuilder& builder,
                 analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {

      };

bool For2Map::can_be_applied(structured_control_flow::For& for_stmt,
                             analysis::AnalysisManager& analysis_manager) {
    std::cout << "For2Map: " << for_stmt.indvar()->get_name() << std::endl;

    // Criterion: loop must be contiguous
    // Simplification to reason about memory offsets and bounds
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    if (!analysis::LoopAnalysis::is_contiguous(&for_stmt, assumptions_analysis)) {
        std::cout << "Loop is not contiguous" << std::endl;
        return false;
    }

    // Criterion: loop condition can be written as a closed-form expression.
    // Closed-form: i < expression_no_i
    // Example: i < N && i < M -> i < min(N, M)
    auto bound = analysis::LoopAnalysis::canonical_bound(&for_stmt, assumptions_analysis);
    if (bound == SymEngine::null) {
        std::cout << "Loop has no canonical bound" << std::endl;
        return false;
    }

    // Criterion: loop must be data-parallel w.r.t containers
    return false;

    /*
    auto& loop_dependency_analysis = analysis_manager.get<analysis::LoopDependencyAnalysis>();
    auto dependencies = loop_dependency_analysis.get(for_stmt);

    // a. No true dependencies (RAW) between iterations
    for (auto& dep : dependencies) {
        if (dep.second == analysis::LoopCarriedDependency::RAW) {
            std::cout << "Loop has true dependency: " << dep.first << std::endl;
            return false;
        }
    }

    // b. False dependencies (WAW) are limited to loop-local variables
    auto& users = analysis_manager.get<analysis::Users>();
    auto locals = users.locals(for_stmt.root());
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        if (locals.find(container) == locals.end()) {
            std::cout << "Loop has false dependency: " << container << " is a local variable"
                      << std::endl;
            return false;
        }
    }
    */

    return true;
}

void For2Map::apply(structured_control_flow::For& for_stmt, builder::StructuredSDFGBuilder& builder,
                    analysis::AnalysisManager& analysis_manager) {
    // Contiguous and canonical bound -> we can compute the number of iterations
    auto init = for_stmt.init();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto num_iterations = symbolic::sub(
        analysis::LoopAnalysis::canonical_bound(&for_stmt, assumptions_analysis), init);

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent =
        static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&for_stmt));

    // convert for to map
    auto& map = builder.convert_for(*parent, for_stmt, num_iterations);
    auto& indvar = map.indvar();

    // Shift indvar by init in body
    auto shift = symbolic::add(indvar, init);
    map.root().replace(indvar, shift);

    // set indvar to last value of a sequential loop
    auto successor = builder.add_block_after(*parent, map);
    auto last_value = symbolic::add(init, num_iterations);
    successor.second.assignments().insert({indvar, last_value});
}

bool For2Map::accept(structured_control_flow::Sequence& parent,
                     structured_control_flow::For& node) {
    if (!this->can_be_applied(node, analysis_manager_)) {
        return false;
    }

    this->apply(node, builder_, analysis_manager_);
    return true;
}

}  // namespace passes
}  // namespace sdfg

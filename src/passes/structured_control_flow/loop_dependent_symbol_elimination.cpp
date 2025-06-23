#include "sdfg/passes/structured_control_flow/loop_dependent_symbol_elimination.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace passes {

bool LoopDependentSymbolElimination::eliminate_symbols(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    structured_control_flow::Transition& transition) {
    if (loop.root().size() == 0) {
        return false;
    }

    bool applied = false;

    auto indvar = loop.indvar();
    auto update = loop.update();
    auto init = loop.init();
    auto condition = loop.condition();

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(loop.root(), true);

    // Assume simple loops: i = 0; i < N; i++
    if (!SymEngine::eq(*init, *symbolic::integer(0))) {
        return false;
    }
    if (!analysis::LoopAnalysis::is_contiguous(&loop, assumptions_analysis)) {
        return false;
    }
    auto bound = analysis::DataParallelismAnalysis::bound(loop);
    if (bound == SymEngine::null || !SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        return false;
    }
    for (auto atom : symbolic::atoms(bound)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
        if (transition.assignments().find(sym) != transition.assignments().end()) {
            return false;
        }
    }

    // Find all symbolic upates
    auto& last_transition = loop.root().at(loop.root().size() - 1).second;
    auto& last_assignments = last_transition.assignments();
    std::unordered_set<std::string> loop_dependent_symbols;
    for (auto& entry : last_assignments) {
        auto& sym = entry.first;
        auto& assign = entry.second;
        if (!symbolic::series::is_contiguous(assign, sym, assumptions)) {
            continue;
        }
        loop_dependent_symbols.insert(sym->get_name());
    }
    if (loop_dependent_symbols.empty()) {
        return false;
    }

    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users(all_users, loop.root());
    for (auto& cand : loop_dependent_symbols) {
        auto writes = users.writes(cand);
        if (writes.size() != 1) {
            continue;
        }
        auto reads = users.reads(cand);
        bool has_dataflow = false;
        for (auto& read : reads) {
            if (dynamic_cast<data_flow::AccessNode*>(read->element())) {
                has_dataflow = true;
                break;
            }
        }
        if (has_dataflow) {
            continue;
        }
        auto sym = symbolic::symbol(cand);
        last_assignments.erase(sym);
        loop.root().replace(sym, symbolic::add(indvar, sym));

        transition.assignments().insert({sym, symbolic::add(sym, bound)});

        applied = true;
    }

    return applied;
};

LoopDependentSymbolElimination::LoopDependentSymbolElimination()
    : Pass() {

      };

std::string LoopDependentSymbolElimination::name() { return "LoopDependentSymbolElimination"; };

bool LoopDependentSymbolElimination::run_pass(builder::StructuredSDFGBuilder& builder,
                                              analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Add children to queue
        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                if (auto match = dynamic_cast<structured_control_flow::For*>(&child.first)) {
                    applied |=
                        this->eliminate_symbols(builder, analysis_manager, *match, child.second);
                }
            }
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

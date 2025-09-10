#include "sdfg/passes/symbolic/symbol_evolution.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace passes {

symbolic::Expression scalar_evolution(
    structured_control_flow::StructuredLoop& loop,
    symbolic::Symbol indvar,
    symbolic::Expression indvar_update,
    symbolic::Expression indvar_init,
    symbolic::Symbol sym,
    symbolic::Expression sym_update,
    symbolic::Expression sym_init,
    const std::unordered_set<std::string>& moving_symbols
) {
    // Check if expr is safe
    for (auto& atom : symbolic::atoms(sym_update)) {
        if (symbolic::eq(atom, sym)) {
            continue;
        }
        if (moving_symbols.find(atom->get_name()) != moving_symbols.end()) {
            return SymEngine::null;
        }
    }

    // Pattern 1: Loop Alias
    if (symbolic::eq(sym_update, indvar_update) && symbolic::eq(sym_init, indvar_init)) {
        return indvar;
    }

    if (symbolic::uses(sym_update, indvar)) {
        return SymEngine::null;
    }

    // Pattern 2: Constant
    if (!symbolic::uses(sym_update, sym) && symbolic::eq(sym_update, sym_init)) {
        return sym_update;
    }

    // Pattern 3: Affine update
    auto stride = analysis::LoopAnalysis::stride(&loop);
    if (stride == SymEngine::null) {
        return SymEngine::null;
    }

    symbolic::SymbolVec gens = {sym};
    auto poly = symbolic::polynomial(sym_update, gens);
    auto coeffs = symbolic::affine_coefficients(poly, gens);
    if (coeffs.empty()) {
        return SymEngine::null;
    }
    auto mul = coeffs.at(sym);
    if (!symbolic::eq(mul, symbolic::one())) {
        return SymEngine::null;
    }
    auto offset = coeffs.at(symbolic::symbol("__daisy_constant__"));

    auto iter = symbolic::div(symbolic::sub(indvar, indvar_init), stride);
    auto inv = symbolic::add(symbolic::mul(iter, offset), sym_init);
    return inv;
}

bool SymbolEvolution::eliminate_symbols(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    structured_control_flow::Transition& transition
) {
    if (loop.root().size() == 0) {
        return false;
    }

    bool applied = false;

    auto indvar = loop.indvar();
    auto update = loop.update();
    auto init = loop.init();
    auto condition = loop.condition();

    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, loop.root());

    // Loop-dependent symbols are written in the loop body
    std::unordered_set<std::string> candidates;
    for (auto& entry : body_users.writes()) {
        auto& type = builder.subject().type(entry->container());
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }
        if (!types::is_integer(type.primitive_type())) {
            continue;
        }
        candidates.insert(entry->container());
    }

    // Filter out candidates that are not loop-dependent
    std::unordered_map<std::string, structured_control_flow::Transition*> aliases;
    std::unordered_map<std::string, structured_control_flow::Transition*> pseudo_iterators;
    for (auto& sym : candidates) {
        // Criterion: Must have place after loop
        if (transition.assignments().find(symbolic::symbol(sym)) != transition.assignments().end()) {
            continue;
        }

        // Criterion: Must be written once
        if (body_users.writes(sym).size() != 1) {
            continue;
        }
        // Criterion: Must be read to avoid infinite loop
        if (body_users.reads(sym).size() == 0) {
            continue;
        }

        // Criterion: Write must be a symbolic assignment
        auto update_write = body_users.writes(sym).at(0);
        auto update_write_element = update_write->element();
        if (!dynamic_cast<structured_control_flow::Transition*>(update_write_element)) {
            continue;
        }
        auto& update_transition = static_cast<structured_control_flow::Transition&>(*update_write_element);
        auto update_sym = update_transition.assignments().at(symbolic::symbol(sym));

        // Criterion: Not in a nested loop
        bool nested_loop = false;
        auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
        structured_control_flow::ControlFlowNode* scope = &update_transition.parent();
        while (scope != nullptr && scope != &loop.root()) {
            if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(scope)) {
                nested_loop = true;
                break;
            } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(scope)) {
                nested_loop = true;
                break;
            }
            scope = scope_analysis.parent_scope(scope);
        }
        if (nested_loop) {
            continue;
        }

        // Criterion: Candidate is not read after the update
        auto uses = body_users.all_uses_after(*update_write);
        bool used_after_update = false;
        for (auto& use : uses) {
            if (use->container() == sym) {
                used_after_update = true;
                break;
            }
        }
        if (used_after_update) {
            continue;
        }

        // Criterion: Candidate is initialized before the loop
        auto all_writes = users.writes(sym);
        if (all_writes.size() != 2) {
            continue;
        }
        auto init_write = all_writes.at(0);
        if (init_write == update_write) {
            init_write = all_writes.at(1);
        }
        if (!users.dominates(*init_write, *update_write)) {
            continue;
        }
        if (!dynamic_cast<structured_control_flow::Transition*>(init_write->element())) {
            continue;
        }
        auto& init_transition = static_cast<structured_control_flow::Transition&>(*init_write->element());
        auto init_sym = init_transition.assignments().at(symbolic::symbol(sym));

        // Criterion: Infer scalar evolution
        auto evolution =
            scalar_evolution(loop, indvar, update, init, symbolic::symbol(sym), update_sym, init_sym, candidates);
        if (evolution == SymEngine::null) {
            continue;
        }

        // Apply by inserting redefinition before the loop body
        auto& old_first_block = loop.root().at(0).first;
        builder.add_block_before(
            loop.root(),
            old_first_block,
            {{symbolic::symbol(sym), evolution}},
            builder.debug_info().get_region(old_first_block.debug_info().indices())
        );
        update_transition.assignments().erase(symbolic::symbol(sym));
        transition.assignments().insert({symbolic::symbol(sym), update_sym});

        applied = true;
    }

    return applied;
};

SymbolEvolution::SymbolEvolution()
    : Pass() {

      };

std::string SymbolEvolution::name() { return "SymbolEvolution"; };

bool SymbolEvolution::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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
                if (auto match = dynamic_cast<structured_control_flow::StructuredLoop*>(&child.first)) {
                    applied |= this->eliminate_symbols(builder, analysis_manager, *match, child.second);
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
        } else if (auto sloop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg

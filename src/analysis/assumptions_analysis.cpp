#include "sdfg/analysis/assumptions_analysis.h"

#include <utility>
#include <vector>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace analysis {

symbolic::Expression AssumptionsAnalysis::cnf_to_upper_bound(const symbolic::CNF& cnf,
                                                             const symbolic::Symbol& indvar) {
    std::vector<symbolic::Expression> candidates;

    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            // Comparison: indvar < expr
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), indvar) &&
                    !symbolic::uses(lt->get_arg2(), indvar)) {
                    auto ub = symbolic::sub(lt->get_arg2(), symbolic::one());
                    candidates.push_back(ub);
                }
            }
            // Comparison: indvar <= expr
            else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
                if (symbolic::eq(le->get_arg1(), indvar) &&
                    !symbolic::uses(le->get_arg2(), indvar)) {
                    candidates.push_back(le->get_arg2());
                }
            }
            // Comparison: indvar == expr
            else if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(literal);
                if (symbolic::eq(eq->get_arg1(), indvar) &&
                    !symbolic::uses(eq->get_arg2(), indvar)) {
                    candidates.push_back(eq->get_arg2());
                }
            }
        }
    }

    if (candidates.empty()) {
        return SymEngine::null;
    }

    // Return the smallest upper bound across all candidate constraints
    symbolic::Expression result = candidates[0];
    for (size_t i = 1; i < candidates.size(); ++i) {
        result = symbolic::min(result, candidates[i]);
    }

    return result;
}

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

void AssumptionsAnalysis::visit_block(structured_control_flow::Block* block,
                                      analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::visit_sequence(structured_control_flow::Sequence* sequence,
                                         analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::visit_if_else(structured_control_flow::IfElse* if_else,
                                        analysis::AnalysisManager& analysis_manager) {
    auto& users = analysis_manager.get<analysis::Users>();
    for (size_t i = 0; i < if_else->size(); i++) {
        auto& scope = if_else->at(i).first;
        auto condition = if_else->at(i).second;
        auto symbols = symbolic::atoms(condition);

        // Assumption: symbols are read-only in scope
        bool read_only = true;
        analysis::UsersView scope_users(users, scope);
        for (auto& sym : symbols) {
            if (scope_users.writes(sym->get_name()).size() > 0) {
                read_only = false;
                break;
            }
        }
        if (!read_only) {
            continue;
        }

        try {
            auto cnf = symbolic::conjunctive_normal_form(condition);

            // Assumption: no or conditions
            bool has_complex_clauses = false;
            for (auto& clause : cnf) {
                if (clause.size() > 1) {
                    has_complex_clauses = true;
                    break;
                }
            }
            if (has_complex_clauses) {
                continue;
            }

            for (auto& sym : symbols) {
                symbolic::Expression ub = symbolic::infty(1);
                symbolic::Expression lb = symbolic::infty(-1);
                for (auto& clause : cnf) {
                    auto& literal = clause[0];
                    // Literal does not use symbol
                    if (!symbolic::uses(literal, sym)) {
                        continue;
                    }

                    if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                        auto eq = SymEngine::rcp_dynamic_cast<const SymEngine::Equality>(literal);
                        auto lhs = eq->get_args()[0];
                        auto rhs = eq->get_args()[1];
                        if (SymEngine::eq(*lhs, *sym) && !symbolic::uses(rhs, sym)) {
                            ub = rhs;
                            lb = rhs;
                            break;
                        } else if (SymEngine::eq(*rhs, *sym) && !symbolic::uses(lhs, sym)) {
                            ub = lhs;
                            lb = lhs;
                            break;
                        }
                    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                        auto lt =
                            SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(literal);
                        auto lhs = lt->get_args()[0];
                        auto rhs = lt->get_args()[1];
                        if (SymEngine::eq(*lhs, *sym) && !symbolic::uses(rhs, sym)) {
                            if (symbolic::eq(ub, symbolic::infty(1))) {
                                ub = rhs;
                            } else {
                                ub = symbolic::min(ub, rhs);
                            }
                        } else if (SymEngine::eq(*rhs, *sym) && !symbolic::uses(lhs, sym)) {
                            if (symbolic::eq(lb, symbolic::infty(-1))) {
                                lb = lhs;
                            } else {
                                lb = symbolic::max(lb, lhs);
                            }
                        }
                    } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                        auto lt = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(literal);
                        auto lhs = lt->get_args()[0];
                        auto rhs = lt->get_args()[1];
                        if (SymEngine::eq(*lhs, *sym) && !symbolic::uses(rhs, sym)) {
                            if (symbolic::eq(ub, symbolic::infty(1))) {
                                ub = rhs;
                            } else {
                                ub = symbolic::min(ub, rhs);
                            }
                        } else if (SymEngine::eq(*rhs, *sym) && !symbolic::uses(lhs, sym)) {
                            if (symbolic::eq(lb, symbolic::infty(-1))) {
                                lb = lhs;
                            } else {
                                lb = symbolic::max(lb, lhs);
                            }
                        }
                    }
                }

                // Failed to infer anything
                if (symbolic::eq(ub, symbolic::infty(1)) && symbolic::eq(lb, symbolic::infty(-1))) {
                    continue;
                }

                if (this->assumptions_.find(&scope) == this->assumptions_.end()) {
                    this->assumptions_.insert({&scope, symbolic::Assumptions()});
                }
                auto& scope_assumptions = this->assumptions_[&scope];
                if (scope_assumptions.find(sym) == scope_assumptions.end()) {
                    scope_assumptions.insert({sym, symbolic::Assumption(sym)});
                }

                if (!symbolic::eq(ub, symbolic::infty(1))) {
                    scope_assumptions[sym].upper_bound(ub);
                }
                if (!symbolic::eq(lb, symbolic::infty(-1))) {
                    scope_assumptions[sym].lower_bound(lb);
                }
            }
        } catch (const symbolic::CNFException& e) {
            continue;
        }
    }
};

void AssumptionsAnalysis::visit_while(structured_control_flow::While* while_loop,
                                      analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::visit_for(structured_control_flow::For* for_loop,
                                    analysis::AnalysisManager& analysis_manager) {
    auto indvar = for_loop->indvar();
    auto update = for_loop->update();

    // Add new assumptions
    auto& body = for_loop->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }
    auto& body_assumptions = this->assumptions_[&body];
    if (body_assumptions.find(indvar) == body_assumptions.end()) {
        body_assumptions.insert({indvar, symbolic::Assumption(indvar)});
    }

    // Assumption 1: indvar moves according to update
    body_assumptions[indvar].map(update);

    // Prove that update is monotonic -> assume bounds
    auto& assums = this->get(*for_loop);
    if (!symbolic::is_monotonic(update, indvar, assums)) {
        return;
    }

    // Assumption 2: init is lower bound
    body_assumptions[indvar].lower_bound(for_loop->init());
    try {
        auto cnf = symbolic::conjunctive_normal_form(for_loop->condition());
        auto ub = cnf_to_upper_bound(cnf, indvar);
        if (ub == SymEngine::null) {
            return;
        }
        // Assumption 3: ub is upper bound
        body_assumptions[indvar].upper_bound(ub);
    } catch (const symbolic::CNFException& e) {
        return;
    }
}

void AssumptionsAnalysis::visit_map(structured_control_flow::Map* map,
                                    analysis::AnalysisManager& analysis_manager) {
    auto indvar = map->indvar();

    auto& body = map->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }
    auto& body_assumptions = this->assumptions_[&body];
    if (body_assumptions.find(indvar) == body_assumptions.end()) {
        body_assumptions.insert({indvar, symbolic::Assumption(indvar)});
    }
    body_assumptions[indvar].lower_bound(symbolic::zero());
    body_assumptions[indvar].upper_bound(symbolic::sub(map->num_iterations(), symbolic::one()));
    body_assumptions[indvar].map(map->update());
};

void AssumptionsAnalysis::traverse(structured_control_flow::Sequence& root,
                                   analysis::AnalysisManager& analysis_manager) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(current)) {
            this->visit_block(block_stmt, analysis_manager);
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            this->visit_sequence(sequence_stmt, analysis_manager);
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            this->visit_if_else(if_else_stmt, analysis_manager);
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->visit_while(while_stmt, analysis_manager);
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            this->visit_for(for_stmt, analysis_manager);
            queue.push_back(&for_stmt->root());
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(current)) {
            this->visit_map(map_stmt, analysis_manager);
            queue.push_back(&map_stmt->root());
        }
    }
};

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->assumptions_.clear();

    // Add sdfg assumptions
    this->assumptions_.insert({&sdfg_.root(), symbolic::Assumptions()});

    // Add additional assumptions
    for (auto& entry : this->additional_assumptions_) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }

    // Forward propagate for each node
    this->traverse(sdfg_.root(), analysis_manager);
};

const symbolic::Assumptions AssumptionsAnalysis::get(structured_control_flow::ControlFlowNode& node,
                                                     bool include_trivial_bounds) {
    // Compute assumptions on the fly

    // Node-level assumptions
    symbolic::Assumptions assums;
    if (this->assumptions_.find(&node) != this->assumptions_.end()) {
        for (auto& entry : this->assumptions_[&node]) {
            assums.insert({entry.first, entry.second});
        }
    }

    AnalysisManager manager(this->sdfg_);
    auto& scope_analysis = manager.get<ScopeAnalysis>();

    auto scope = scope_analysis.parent_scope(&node);
    while (scope != nullptr) {
        // Don't overwrite lower scopes' assumptions
        if (this->assumptions_.find(scope) != this->assumptions_.end()) {
            for (auto& entry : this->assumptions_[scope]) {
                if (assums.find(entry.first) == assums.end()) {
                    assums.insert({entry.first, entry.second});
                }
            }
        }
        scope = scope_analysis.parent_scope(scope);
    }

    if (include_trivial_bounds) {
        for (auto& entry : sdfg_.assumptions()) {
            if (assums.find(entry.first) == assums.end()) {
                assums.insert({entry.first, entry.second});
            }
        }
    }

    return assums;
};

void AssumptionsAnalysis::add(symbolic::Assumptions& assums,
                              structured_control_flow::ControlFlowNode& node) {
    if (this->assumptions_.find(&node) == this->assumptions_.end()) {
        return;
    }

    for (auto& entry : this->assumptions_[&node]) {
        if (assums.find(entry.first) == assums.end()) {
            assums.insert({entry.first, entry.second});
        } else {
            assums[entry.first] = entry.second;
        }
    }
}

}  // namespace analysis
}  // namespace sdfg

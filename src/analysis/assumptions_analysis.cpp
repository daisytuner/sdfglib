#include "sdfg/analysis/assumptions_analysis.h"

#include <utility>
#include <vector>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace analysis {

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

void AssumptionsAnalysis::visit_block(const structured_control_flow::Block* block) { return; };

void AssumptionsAnalysis::visit_sequence(const structured_control_flow::Sequence* sequence) {
    return;
};

void AssumptionsAnalysis::visit_if_else(const structured_control_flow::IfElse* if_else) { return; };

void AssumptionsAnalysis::visit_while(const structured_control_flow::While* while_loop) { return; };

void AssumptionsAnalysis::visit_for(const structured_control_flow::For* for_loop) {
    auto& assums = this->get(*for_loop);

    // Prove that update is monotonic
    auto indvar = for_loop->indvar();
    auto update = for_loop->update();
    if (!symbolic::is_monotonic(update, indvar, assums)) {
        return;
    }

    // Add new assumptions
    auto& body = for_loop->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }
    auto& body_assumptions = this->assumptions_[&body];
    if (body_assumptions.find(indvar) == body_assumptions.end()) {
        body_assumptions.insert({indvar, symbolic::Assumption(indvar)});
    }

    // monotonic => init is lower bound
    body_assumptions[indvar].lower_bound(for_loop->init());
    try {
        auto cnf = symbolic::conjunctive_normal_form(for_loop->condition());
        auto ub = symbolic::upper_bound(cnf, indvar);
        if (ub == SymEngine::null) {
            return;
        }
        body_assumptions[indvar].upper_bound(ub);
    } catch (const symbolic::CNFException& e) {
        return;
    }
}

void AssumptionsAnalysis::visit_map(const structured_control_flow::Map* map) {
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
};

void AssumptionsAnalysis::traverse(const structured_control_flow::Sequence& root) {
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block_stmt = dynamic_cast<const structured_control_flow::Block*>(current)) {
            this->visit_block(block_stmt);
        } else if (auto sequence_stmt =
                       dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            this->visit_sequence(sequence_stmt);
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt =
                       dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            this->visit_if_else(if_else_stmt);
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(current)) {
            this->visit_while(while_stmt);
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::For*>(current)) {
            this->visit_for(for_stmt);
            queue.push_back(&for_stmt->root());
        } else if (auto map_stmt = dynamic_cast<const structured_control_flow::Map*>(current)) {
            this->visit_map(map_stmt);
            queue.push_back(&map_stmt->root());
        }
    }
};

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->assumptions_.clear();

    // Add sdfg assumptions
    this->assumptions_.insert({&sdfg_.root(), symbolic::Assumptions()});
    for (auto& entry : sdfg_.assumptions()) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }

    // Add additional assumptions
    for (auto& entry : this->additional_assumptions_) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }

    // Forward propagate for each node
    this->traverse(sdfg_.root());
};

const symbolic::Assumptions AssumptionsAnalysis::get(
    const structured_control_flow::ControlFlowNode& node) {
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

    return assums;
};

}  // namespace analysis
}  // namespace sdfg

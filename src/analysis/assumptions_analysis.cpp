#include "sdfg/analysis/assumptions_analysis.h"

#include <utility>
#include <vector>

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/analysis.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/basic.h"
#include "symengine/symbol.h"

namespace sdfg {
namespace analysis {

void AssumptionsAnalysis::traverse(structured_control_flow::Sequence& root) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(current)) {
            this->visit_block(block_stmt);
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            this->visit_sequence(sequence_stmt);
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            this->visit_if_else(if_else_stmt);
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->visit_while(while_stmt);
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            this->visit_for(for_stmt);
            queue.push_back(&for_stmt->root());
        } else if (auto map_stmt = dynamic_cast<const structured_control_flow::Map*>(current)) {
            this->visit_map(map_stmt);
            queue.push_back(&map_stmt->root());
        }
    }
};

void AssumptionsAnalysis::visit_block(structured_control_flow::Block* block) { return; };

void AssumptionsAnalysis::visit_sequence(structured_control_flow::Sequence* sequence) { return; };

void AssumptionsAnalysis::visit_if_else(structured_control_flow::IfElse* if_else) { return; };

void AssumptionsAnalysis::visit_while(structured_control_flow::While* while_loop) { return; };

void AssumptionsAnalysis::visit_for(structured_control_flow::For* for_loop) {
    if (symbolic::strict_monotonicity(for_loop->update(), for_loop->indvar()) !=
        symbolic::Sign::POSITIVE) {
        return;
    }

    auto& body = for_loop->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }
    auto& body_assumptions = this->assumptions_[&body];

    auto indvar = for_loop->indvar();
    this->iterators_.push_back(indvar->get_name());
    auto sym = indvar;

    auto update = for_loop->update();
    if (body_assumptions.find(sym) == body_assumptions.end()) {
        body_assumptions.insert({sym, symbolic::Assumption(sym)});
    }
    body_assumptions[sym].map(update);

    auto init = for_loop->init();
    body_assumptions[sym].lower_bound(init);

    auto condition = for_loop->condition();
    auto args = condition->get_args();
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto arg_0 = args[0];
        auto arg_1 = args[1];
        if (symbolic::eq(arg_0, indvar)) {
            body_assumptions[sym].upper_bound(symbolic::sub(arg_1, symbolic::integer(1)));
        }
    } else if (SymEngine::is_a<SymEngine::LessThan>(*condition)) {
        auto arg_0 = args[0];
        auto arg_1 = args[1];
        if (symbolic::eq(arg_0, indvar)) {
            body_assumptions[sym].upper_bound(arg_1);
        }
    } else if (SymEngine::is_a<SymEngine::And>(*condition)) {
        auto args = condition->get_args();
        body_assumptions[sym].upper_bound(symbolic::infty(1));
        for (auto arg : args) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*arg)) {
                auto args = arg->get_args();
                auto arg_0 = args[0];
                auto arg_1 = args[1];
                if (symbolic::eq(arg_0, indvar)) {
                    auto old_ub = body_assumptions[sym].upper_bound();
                    auto new_ub = symbolic::min(old_ub, symbolic::sub(arg_1, symbolic::integer(1)));
                    body_assumptions[sym].upper_bound(new_ub);
                }
            } else if (SymEngine::is_a<SymEngine::LessThan>(*arg)) {
                auto args = arg->get_args();
                auto arg_0 = args[0];
                auto arg_1 = args[1];
                if (symbolic::eq(arg_0, indvar)) {
                    auto old_ub = body_assumptions[sym].upper_bound();
                    auto new_ub = symbolic::min(old_ub, arg_1);
                    body_assumptions[sym].upper_bound(new_ub);
                }
            }
        }
    }
};

void AssumptionsAnalysis::visit_map(const structured_control_flow::Map* map) {
    auto& body = map->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }

    auto& body_assumptions = this->assumptions_[&body];
    auto indvar = map->indvar();
    this->iterators_.push_back(indvar->get_name());
    auto sym = indvar;
    auto num_iterations = map->num_iterations();

    if (body_assumptions.find(sym) == body_assumptions.end()) {
        body_assumptions.insert({sym, symbolic::Assumption(sym)});
    }

    body_assumptions[sym].lower_bound(symbolic::zero());
    body_assumptions[sym].upper_bound(num_iterations);
}

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->assumptions_.clear();

    this->assumptions_.insert({&sdfg_.root(), symbolic::Assumptions()});
    for (auto& entry : sdfg_.assumptions()) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }
    for (auto& entry : this->additional_assumptions_) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }

    this->traverse(sdfg_.root());
};

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

const symbolic::Assumptions AssumptionsAnalysis::get(
    structured_control_flow::ControlFlowNode& node) {
    // TODO: Proper inference based on scope nesting
    symbolic::Assumptions assums;
    for (auto& entry : this->assumptions_) {
        if (entry.first == &sdfg_.root()) {
            continue;
        }
        for (auto& entry_ : entry.second) {
            assums[entry_.first] = entry_.second;
        }
    }
    for (auto entry : this->assumptions_[&sdfg_.root()]) {
        if (assums.find(entry.first) != assums.end()) {
            continue;
        }
        assums[entry.first] = entry.second;
    }

    return assums;
};

}  // namespace analysis
}  // namespace sdfg

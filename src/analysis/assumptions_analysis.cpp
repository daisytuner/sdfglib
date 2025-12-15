#include "sdfg/analysis/assumptions_analysis.h"

#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/series.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace analysis {

symbolic::Expression AssumptionsAnalysis::cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar) {
    std::vector<symbolic::Expression> candidates;

    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            // Comparison: indvar < expr
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), indvar) && !symbolic::uses(lt->get_arg2(), indvar)) {
                    auto ub = symbolic::sub(lt->get_arg2(), symbolic::one());
                    candidates.push_back(ub);
                }
            }
            // Comparison: indvar <= expr
            else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
                if (symbolic::eq(le->get_arg1(), indvar) && !symbolic::uses(le->get_arg2(), indvar)) {
                    candidates.push_back(le->get_arg2());
                }
            }
            // Comparison: indvar == expr
            else if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(literal);
                if (symbolic::eq(eq->get_arg1(), indvar) && !symbolic::uses(eq->get_arg2(), indvar)) {
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

void AssumptionsAnalysis::visit_block(structured_control_flow::Block* block, analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::
    visit_sequence(structured_control_flow::Sequence* sequence, analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::
    visit_if_else(structured_control_flow::IfElse* if_else, analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::
    visit_while(structured_control_flow::While* while_loop, analysis::AnalysisManager& analysis_manager) {
    return;
};

void AssumptionsAnalysis::
    visit_structured_loop(structured_control_flow::StructuredLoop* loop, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop->indvar();
    auto update = loop->update();
    auto init = loop->init();

    // Add new assumptions
    auto& body = loop->root();
    if (this->assumptions_.find(&body) == this->assumptions_.end()) {
        this->assumptions_.insert({&body, symbolic::Assumptions()});
    }
    auto& body_assumptions = this->assumptions_[&body];

    // Define all constant symbols

    // By definition, all symbols in the loop condition are constant within the loop body
    symbolic::SymbolSet syms = {indvar};
    for (auto& sym : symbolic::atoms(loop->condition())) {
        syms.insert(sym);
    }
    for (auto& sym : syms) {
        if (body_assumptions.find(sym) == body_assumptions.end()) {
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
        body_assumptions[sym].constant(true);
    }

    // Collect other constant symbols based on uses
    UsersView users_view(*this->users_analysis_, body);
    std::unordered_set<std::string> visited;
    for (auto& read : users_view.reads()) {
        if (!sdfg_.exists(read->container())) {
            continue;
        }

        if (visited.find(read->container()) != visited.end()) {
            continue;
        }
        visited.insert(read->container());

        auto& type = this->sdfg_.type(read->container());
        if (!type.is_symbol() || type.type_id() != types::TypeID::Scalar) {
            continue;
        }

        if (users_view.reads(read->container()) != users_view.uses(read->container())) {
            continue;
        }

        if (body_assumptions.find(symbolic::symbol(read->container())) == body_assumptions.end()) {
            symbolic::Symbol sym = symbolic::symbol(read->container());
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
            body_assumptions[sym].constant(true);
        }
    }

    // Define map of indvar
    body_assumptions[indvar].map(update);

    // Determine non-tight lower and upper bounds from inverse index access
    std::vector<symbolic::Expression> lbs, ubs;
    for (auto user : this->users_analysis_->reads(indvar->get_name())) {
        if (auto* memlet = dynamic_cast<data_flow::Memlet*>(user->element())) {
            const types::IType* memlet_type = &memlet->base_type();
            for (long long i = memlet->subset().size() - 1; i >= 0; i--) {
                symbolic::Expression num_elements = SymEngine::null;
                if (auto* pointer_type = dynamic_cast<const types::Pointer*>(memlet_type)) {
                    memlet_type = &pointer_type->pointee_type();
                } else if (auto* array_type = dynamic_cast<const types::Array*>(memlet_type)) {
                    memlet_type = &array_type->element_type();
                    num_elements = array_type->num_elements();
                } else {
                    break;
                }
                if (!symbolic::uses(memlet->subset().at(i), indvar)) {
                    continue;
                }
                symbolic::Expression inverse = symbolic::inverse(memlet->subset().at(i), indvar);
                if (inverse.is_null()) {
                    continue;
                }
                lbs.push_back(symbolic::subs(inverse, indvar, symbolic::zero()));
                if (num_elements.is_null()) {
                    std::string container;
                    if (memlet->src_conn() == "void") {
                        container = dynamic_cast<data_flow::AccessNode&>(memlet->src()).data();
                    } else {
                        container = dynamic_cast<data_flow::AccessNode&>(memlet->dst()).data();
                    }
                    if (this->is_parameter(container)) {
                        ubs.push_back(symbolic::sub(
                            symbolic::subs(inverse, indvar, symbolic::dynamic_sizeof(symbolic::symbol(container))),
                            symbolic::one()
                        ));
                    }
                } else {
                    ubs.push_back(symbolic::sub(symbolic::subs(inverse, indvar, num_elements), symbolic::one()));
                }
            }
        }
    }
    for (auto lb : lbs) {
        body_assumptions[indvar].add_lower_bound(lb);
    }
    for (auto ub : ubs) {
        body_assumptions[indvar].add_upper_bound(ub);
    }

    // Prove that update is monotonic -> assume bounds
    auto& assums = this->get(*loop);
    if (!symbolic::series::is_monotonic(update, indvar, assums)) {
        return;
    }

    // Assumption: init is tight lower bound for indvar
    body_assumptions[indvar].add_lower_bound(init);
    body_assumptions[indvar].tight_lower_bound(init);
    body_assumptions[indvar].lower_bound_deprecated(init);

    // Assumption: inverse index access lower bounds are lower bound for init
    if (SymEngine::is_a<SymEngine::Symbol>(*init) && !lbs.empty()) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(init);
        if (!body_assumptions.contains(sym)) {
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
        for (auto lb : lbs) {
            body_assumptions[sym].add_lower_bound(lb);
        }
    }

    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(loop->condition());
    } catch (const symbolic::CNFException& e) {
        return;
    }
    auto ub = cnf_to_upper_bound(cnf, indvar);
    if (ub.is_null()) {
        return;
    }
    // Assumption: upper bound ub is tight for indvar if
    // body_assumptions[indvar].add_upper_bound(ub);
    body_assumptions[indvar].upper_bound_deprecated(ub);
    // TODO: handle non-contiguous tight upper bounds with modulo
    // Example: for (i = 0; i < n; i += 3) -> tight_upper_bound = (n - 1) - ((n - 1) % 3)
    if (symbolic::series::is_contiguous(update, indvar, assums)) {
        body_assumptions[indvar].tight_upper_bound(ub);
    }

    // Assumption: inverse index access upper bounds are upper bound for ub
    if (SymEngine::is_a<SymEngine::Symbol>(*ub) && !ubs.empty()) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(ub);
        if (!body_assumptions.contains(sym)) {
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
        for (auto ub : ubs) {
            body_assumptions[sym].add_upper_bound(ub);
        }
    }

    // Assumption: any ub symbol is at least init
    for (auto& sym : symbolic::atoms(ub)) {
        body_assumptions[sym].add_lower_bound(symbolic::add(init, symbolic::one()));
        body_assumptions[sym].lower_bound_deprecated(symbolic::add(init, symbolic::one()));
    }
}

void AssumptionsAnalysis::traverse(structured_control_flow::Sequence& root, analysis::AnalysisManager& analysis_manager) {
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
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->visit_structured_loop(loop_stmt, analysis_manager);
            queue.push_back(&loop_stmt->root());
        }
    }
};

void AssumptionsAnalysis::determine_parameters(analysis::AnalysisManager& analysis_manager) {
    for (auto& container : this->sdfg_.arguments()) {
        bool readonly = true;
        Use not_allowed;
        switch (this->sdfg_.type(container).type_id()) {
            case types::TypeID::Scalar:
                not_allowed = Use::WRITE;
                break;
            case types::TypeID::Pointer:
                not_allowed = Use::MOVE;
                break;
            case types::TypeID::Array:
            case types::TypeID::Structure:
            case types::TypeID::Reference:
            case types::TypeID::Function:
                continue;
        }
        for (auto user : this->users_analysis_->uses(container)) {
            if (user->use() == not_allowed) {
                readonly = false;
                break;
            }
        }
        if (readonly) {
            this->parameters_.insert(symbolic::symbol(container));
        }
    }
}

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->assumptions_.clear();
    this->parameters_.clear();
    this->cache_nodes_.clear();
    this->cache_range_.clear();

    // Add sdfg assumptions
    this->assumptions_.insert({&sdfg_.root(), symbolic::Assumptions()});

    // Add additional assumptions
    for (auto& entry : this->additional_assumptions_) {
        this->assumptions_[&sdfg_.root()][entry.first] = entry.second;
    }

    this->scope_analysis_ = &analysis_manager.get<ScopeAnalysis>();
    this->users_analysis_ = &analysis_manager.get<Users>();

    // Determine parameters
    this->determine_parameters(analysis_manager);

    // Forward propagate for each node
    this->traverse(sdfg_.root(), analysis_manager);
};

const symbolic::Assumptions AssumptionsAnalysis::
    get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds) {
    if (include_trivial_bounds) {
        std::string key = std::to_string(node.element_id());
        if (this->cache_nodes_.find(key) != this->cache_nodes_.end()) {
            return this->cache_nodes_[key];
        }
    }

    // Compute assumptions on the fly

    // Node-level assumptions
    symbolic::Assumptions assums;
    if (this->assumptions_.find(&node) != this->assumptions_.end()) {
        for (auto& entry : this->assumptions_[&node]) {
            assums.insert({entry.first, entry.second});
        }
    }

    auto scope = scope_analysis_->parent_scope(&node);
    while (scope != nullptr) {
        if (this->assumptions_.find(scope) == this->assumptions_.end()) {
            scope = scope_analysis_->parent_scope(scope);
            continue;
        }
        for (auto& entry : this->assumptions_[scope]) {
            if (assums.find(entry.first) == assums.end()) {
                // New assumption
                assums.insert({entry.first, entry.second});
                continue;
            }

            // Merge assumptions from lower scopes
            auto& lower_assum = assums[entry.first];

            // Deprecated: combine with min
            auto lower_ub_deprecated = lower_assum.upper_bound_deprecated();
            auto lower_lb_deprecated = lower_assum.lower_bound_deprecated();
            auto new_ub_deprecated = symbolic::min(entry.second.upper_bound_deprecated(), lower_ub_deprecated);
            auto new_lb_deprecated = symbolic::max(entry.second.lower_bound_deprecated(), lower_lb_deprecated);
            lower_assum.upper_bound_deprecated(new_ub_deprecated);
            lower_assum.lower_bound_deprecated(new_lb_deprecated);

            // Add to set of bounds
            for (auto ub : entry.second.upper_bounds()) {
                lower_assum.add_upper_bound(ub);
            }
            for (auto lb : entry.second.lower_bounds()) {
                lower_assum.add_lower_bound(lb);
            }

            // Set tight bounds
            if (lower_assum.tight_upper_bound().is_null()) {
                lower_assum.tight_upper_bound(entry.second.tight_upper_bound());
            }
            if (lower_assum.tight_lower_bound().is_null()) {
                lower_assum.tight_lower_bound(entry.second.tight_lower_bound());
            }

            // Set map
            if (lower_assum.map().is_null()) {
                lower_assum.map(entry.second.map());
            }

            // Set constant
            if (!lower_assum.constant()) {
                lower_assum.constant(entry.second.constant());
            }
        }
        scope = scope_analysis_->parent_scope(scope);
    }

    if (include_trivial_bounds) {
        for (auto& entry : sdfg_.assumptions()) {
            if (assums.find(entry.first) == assums.end()) {
                assums.insert({entry.first, entry.second});
            } else {
                for (auto& lb : entry.second.lower_bounds()) {
                    assums.at(entry.first).add_lower_bound(lb);
                }
                for (auto& ub : entry.second.upper_bounds()) {
                    assums.at(entry.first).add_upper_bound(ub);
                }
            }
        }
    }

    if (include_trivial_bounds) {
        std::string key = std::to_string(node.element_id());
        this->cache_nodes_.insert({key, assums});
    }
    return assums;
};

const symbolic::Assumptions AssumptionsAnalysis::
    get(structured_control_flow::ControlFlowNode& from,
        structured_control_flow::ControlFlowNode& to,
        bool include_trivial_bounds) {
    if (include_trivial_bounds) {
        std::string key = std::to_string(from.element_id()) + "." + std::to_string(to.element_id());
        if (this->cache_range_.find(key) != this->cache_range_.end()) {
            return this->cache_range_[key];
        }
    }

    auto assums_from = this->get(from, include_trivial_bounds);
    auto assums_to = this->get(to, include_trivial_bounds);

    // Add lower scope assumptions to outer
    // ignore constants assumption
    for (auto& entry : assums_from) {
        if (assums_to.find(entry.first) == assums_to.end()) {
            auto assums_safe = assums_to;
            assums_safe.at(entry.first).constant(false);
            assums_to.insert({entry.first, assums_safe.at(entry.first)});
        } else {
            auto lower_assum = assums_to[entry.first];
            auto lower_ub_deprecated = lower_assum.upper_bound_deprecated();
            auto lower_lb_deprecated = lower_assum.lower_bound_deprecated();
            auto new_ub_deprecated = symbolic::min(entry.second.upper_bound_deprecated(), lower_ub_deprecated);
            auto new_lb_deprecated = symbolic::max(entry.second.lower_bound_deprecated(), lower_lb_deprecated);
            lower_assum.upper_bound_deprecated(new_ub_deprecated);
            lower_assum.lower_bound_deprecated(new_lb_deprecated);

            for (auto ub : entry.second.upper_bounds()) {
                lower_assum.add_upper_bound(ub);
            }
            for (auto lb : entry.second.lower_bounds()) {
                lower_assum.add_lower_bound(lb);
            }

            auto lower_tight_ub = lower_assum.tight_upper_bound();
            if (!entry.second.tight_upper_bound().is_null() && !lower_tight_ub.is_null()) {
                auto new_tight_ub = symbolic::min(entry.second.tight_upper_bound(), lower_tight_ub);
                lower_assum.tight_upper_bound(new_tight_ub);
            }
            auto lower_tight_lb = lower_assum.tight_lower_bound();
            if (!entry.second.tight_lower_bound().is_null() && !lower_tight_lb.is_null()) {
                auto new_tight_lb = symbolic::max(entry.second.tight_lower_bound(), lower_tight_lb);
                lower_assum.tight_lower_bound(new_tight_lb);
            }

            if (lower_assum.map() == SymEngine::null) {
                lower_assum.map(entry.second.map());
            }
            lower_assum.constant(entry.second.constant());
            assums_to[entry.first] = lower_assum;
        }
    }

    if (include_trivial_bounds) {
        std::string key = std::to_string(from.element_id()) + "." + std::to_string(to.element_id());
        this->cache_range_.insert({key, assums_to});
    }
    return assums_to;
}

const symbolic::SymbolSet& AssumptionsAnalysis::parameters() { return this->parameters_; }

bool AssumptionsAnalysis::is_parameter(const symbolic::Symbol& container) {
    return this->parameters_.contains(container);
}

bool AssumptionsAnalysis::is_parameter(const std::string& container) {
    return this->is_parameter(symbolic::symbol(container));
}

void AssumptionsAnalysis::add(symbolic::Assumptions& assums, structured_control_flow::ControlFlowNode& node) {
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

} // namespace analysis
} // namespace sdfg

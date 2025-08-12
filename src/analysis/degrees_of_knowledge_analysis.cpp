#include "sdfg/analysis/degrees_of_knowledge_analysis.h"

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/analysis/work_depth_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/sets.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace analysis {

DegreesOfKnowledgeAnalysis::DegreesOfKnowledgeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}


void DegreesOfKnowledgeAnalysis::run(AnalysisManager& analysis_manager) {
    // Initialize the analysis
    this->number_analysis(analysis_manager, symbolic::one(), false, &sdfg_.root());
    this->size_analysis(analysis_manager);
    this->load_analysis(analysis_manager);
    this->balance_analysis(analysis_manager);
}

void DegreesOfKnowledgeAnalysis::number_analysis(
    AnalysisManager& analysis_manager,
    symbolic::Expression base_iterations,
    bool branched,
    structured_control_flow::ControlFlowNode* node
) {
    if (dynamic_cast<structured_control_flow::Map*>(node)) {
        auto* map_node = dynamic_cast<structured_control_flow::Map*>(node);
        if (branched) {
            number_of_maps_.insert({map_node, {symbolic::zero(), base_iterations}});
        } else {
            number_of_maps_.insert({map_node, {base_iterations, base_iterations}});
        }

        auto stride = analysis::LoopAnalysis::stride(map_node);
        auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
        auto bound = analysis::LoopAnalysis::canonical_bound(map_node, assumptions_analysis);
        auto num_iterations = symbolic::sub(bound, map_node->init());
        num_iterations = symbolic::div(num_iterations, stride);

        number_analysis(analysis_manager, symbolic::mul(num_iterations, base_iterations), branched, &map_node->root());
    } else if (dynamic_cast<structured_control_flow::For*>(node)) {
        auto* for_loop = dynamic_cast<structured_control_flow::For*>(node);

        auto stride = analysis::LoopAnalysis::stride(for_loop);
        auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
        symbolic::Expression num_iterations = symbolic::zero();
        auto bound = analysis::LoopAnalysis::canonical_bound(for_loop, assumptions_analysis);
        if (bound == SymEngine::null) {
            std::string while_symbol_name = "while_" + std::to_string(for_loop->element_id());
            symbolic::Symbol while_symbol = symbolic::symbol(while_symbol_name);
            while_symbols_.insert(while_symbol);
            num_iterations = while_symbol;
        } else {
            num_iterations = symbolic::sub(bound, for_loop->init());
            num_iterations = symbolic::div(num_iterations, stride);
        }

        number_analysis(analysis_manager, symbolic::mul(num_iterations, base_iterations), branched, &for_loop->root());
    } else if (dynamic_cast<structured_control_flow::While*>(node)) {
        auto* while_loop = dynamic_cast<structured_control_flow::While*>(node);
        std::string while_symbol_name = "while_" + std::to_string(while_loop->element_id());
        symbolic::Symbol while_symbol = symbolic::symbol(while_symbol_name);
        while_symbols_.insert(while_symbol);

        number_analysis(analysis_manager, symbolic::mul(while_symbol, base_iterations), branched, &while_loop->root());
    } else if (dynamic_cast<structured_control_flow::IfElse*>(node)) {
        auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(node);

        for (size_t i = 0; i < if_else->size(); i++) {
            number_analysis(analysis_manager, base_iterations, true, &if_else->at(i).first);
        }
    } else if (dynamic_cast<structured_control_flow::Sequence*>(node)) {
        auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(node);

        for (size_t i = 0; i < sequence->size(); ++i) {
            number_analysis(analysis_manager, base_iterations, branched, &sequence->at(i).first);
        }
    } else {
        // not spanning a scope, so we can stop here
        return;
    }
}

void DegreesOfKnowledgeAnalysis::size_analysis(AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<LoopAnalysis>();
    for (auto& loop : loop_analysis.loops()) {
        if (auto* map_node = dynamic_cast<structured_control_flow::Map*>(loop)) {
            auto stride = analysis::LoopAnalysis::stride(map_node);
            auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
            auto bound = analysis::LoopAnalysis::canonical_bound(map_node, assumptions_analysis);
            auto num_iterations = symbolic::sub(bound, map_node->init());
            num_iterations = symbolic::div(num_iterations, stride);

            symbolic::SymbolSet unbound_symbols;
            if (SymEngine::is_a<SymEngine::Integer>(*num_iterations)) {
                auto integer_iterations = SymEngine::rcp_static_cast<const SymEngine::Integer>(num_iterations);
                size_of_a_map_.insert({map_node, {integer_iterations, unbound_symbols}});
                continue;
            }

            auto atoms = symbolic::atoms(num_iterations);

            auto& users = analysis_manager.get<Users>();
            auto& dependencies = analysis_manager.get<DataDependencyAnalysis>();

            std::unordered_set<ForUser*> for_users;
            for (auto atom : atoms) {
                for (auto read : users.reads(atom->get_name())) {
                    if (ForUser* for_user = dynamic_cast<ForUser*>(read)) {
                        if (for_user->element() == map_node) {
                            for_users.insert(for_user);
                        }
                    }
                }
            }

            for (auto user : for_users) {
                auto defined_by = dependencies.defined_by(user->container());
                auto after = users.all_uses_after(*user);
                if (defined_by.find(user) == defined_by.end()) {
                    continue;
                }
                auto definition = defined_by.at(user);
                std::unordered_set<User*> intersection;
                for (auto& use : after) {
                    if (definition.contains(use)) {
                        intersection.insert(use);
                        unbound_symbols.insert(symbolic::symbol(user->container()));
                        break;
                    }
                }
            }

            size_of_a_map_.insert({map_node, {num_iterations, unbound_symbols}});
        }
    }
}

void DegreesOfKnowledgeAnalysis::load_analysis(AnalysisManager& analysis_manager) {
    auto& work_depth_analysis = analysis_manager.get<WorkDepthAnalysis>();

    auto& loop_analysis = analysis_manager.get<LoopAnalysis>();
    for (auto& loop : loop_analysis.loops()) {
        if (auto* map_node = dynamic_cast<structured_control_flow::Map*>(loop)) {
            auto depth = work_depth_analysis.work(&map_node->root());
            auto while_symbols = work_depth_analysis.while_symbols(depth);
            while_symbols_.insert(while_symbols.begin(), while_symbols.end());

            load_of_a_map_.insert({map_node, depth});
        }
    }
}

bool contains_jump(const structured_control_flow::ControlFlowNode* node) {
    std::list<const structured_control_flow::ControlFlowNode*> stack;
    stack.push_back(node);
    while (!stack.empty()) {
        auto current_node = stack.front();
        stack.pop_front();

        if (dynamic_cast<const structured_control_flow::Break*>(current_node) ||
            dynamic_cast<const structured_control_flow::Continue*>(current_node)) {
            return true;
        } else if (auto loop_node = dynamic_cast<const structured_control_flow::StructuredLoop*>(current_node)) {
            stack.push_back(&loop_node->root());
        } else if (auto sequence_node = dynamic_cast<const structured_control_flow::Sequence*>(current_node)) {
            for (size_t i = 0; i < sequence_node->size(); ++i) {
                stack.push_back(&sequence_node->at(i).first);
            }
        } else if (auto if_else_node = dynamic_cast<const structured_control_flow::IfElse*>(current_node)) {
            for (size_t i = 0; i < if_else_node->size(); i++) {
                stack.push_back(&if_else_node->at(i).first);
            }
        } else if (auto while_loop = dynamic_cast<const structured_control_flow::While*>(current_node)) {
            stack.push_back(&while_loop->root());
        }
    }

    return false;
}

std::unordered_set<User*> get_dependent_writes(
    AnalysisManager& analysis_manager,
    Users& users,
    DataDependencyAnalysis& data_dependency_analysis,
    data_flow::DataFlowNode& node
) {
    std::unordered_set<User*> dependent_writes;

    auto succs = node.get_parent().successors(node);
    std::list<data_flow::DataFlowNode*> successors(succs.begin(), succs.end());
    while (!successors.empty()) {
        auto succ = successors.front();
        successors.pop_front();
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(succ)) {
            dependent_writes.insert(users.get_user(access_node->data(), access_node, Use::WRITE));
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(succ)) {
            auto added_succs = tasklet->get_parent().successors(*tasklet);
            successors.insert(successors.end(), added_succs.begin(), added_succs.end());
        }
    }

    return dependent_writes;
}

std::unordered_set<User*> dynamic_writes(
    AnalysisManager& analysis_manager,
    Users& users,
    DataDependencyAnalysis& data_dependency_analysis,
    structured_control_flow::Map* map_node
) {
    std::unordered_set<User*> writes;
    writes.insert(users.get_user(map_node->indvar()->get_name(), map_node, Use::WRITE, true, false, false));
    writes.insert(users.get_user(map_node->indvar()->get_name(), map_node, Use::WRITE, false, false, true));

    auto users_view = UsersView(users, map_node->root());

    bool updated = true;
    while (updated) {
        updated = false;
        std::unordered_set<User*> reads;
        for (auto write : writes) {
            auto read_users = data_dependency_analysis.defines(*write);
            reads.insert(read_users.begin(), read_users.end());
        }

        for (auto read : reads) {
            auto element = read->element();
            if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element)) {
                auto stride = analysis::LoopAnalysis::stride(loop);
                auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
                auto bound = analysis::LoopAnalysis::canonical_bound(loop, assumptions_analysis);
                auto num_iterations = symbolic::sub(bound, loop->init());
                auto atoms = symbolic::atoms(num_iterations);
                if (atoms.find(symbolic::symbol(read->container())) != atoms.end()) {
                    auto write_init = users.get_user(loop->indvar()->get_name(), loop, Use::WRITE, true, false, false);
                    auto write_update =
                        users.get_user(loop->indvar()->get_name(), loop, Use::WRITE, false, false, true);

                    if (writes.find(write_init) == writes.end()) {
                        writes.insert(write_init);
                        updated = true;
                    }
                    if (writes.find(write_update) == writes.end()) {
                        writes.insert(write_update);
                        updated = true;
                    }
                }
            } else if (auto access = dynamic_cast<data_flow::AccessNode*>(element)) {
                auto dependent_writes =
                    get_dependent_writes(analysis_manager, users, data_dependency_analysis, *access);
                if (!helpers::sets_subset(writes, dependent_writes)) {
                    writes.insert(dependent_writes.begin(), dependent_writes.end());
                    updated = true;
                }
            } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
                if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&memlet->dst())) {
                    if (memlet->type() != data_flow::MemletType::Reference) {
                        auto write = users.get_user(access_node->data(), access_node, Use::WRITE);
                        if (writes.find(write) == writes.end()) {
                            writes.insert(write);
                            updated = true;
                        }
                    }
                } else {
                    auto dependent_writes =
                        get_dependent_writes(analysis_manager, users, data_dependency_analysis, memlet->dst());

                    if (!helpers::sets_subset(writes, dependent_writes)) {
                        writes.insert(dependent_writes.begin(), dependent_writes.end());
                        updated = true;
                    }
                }
            } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(element)) {
                auto dependent_writes =
                    get_dependent_writes(analysis_manager, users, data_dependency_analysis, *access);
                if (!helpers::sets_subset(writes, dependent_writes)) {
                    writes.insert(dependent_writes.begin(), dependent_writes.end());
                    updated = true;
                }
            }
        }
    }
    return writes;
}

void DegreesOfKnowledgeAnalysis::balance_analysis(AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<LoopAnalysis>();
    auto& users = analysis_manager.get<Users>();
    auto& data_dependency_analysis = analysis_manager.get<DataDependencyAnalysis>();

    symbolic::SymbolSet if_else_symbols;

    for (auto& loop : loop_analysis.loops()) {
        if (auto* map_node = dynamic_cast<structured_control_flow::Map*>(loop)) {
            auto writes = dynamic_writes(analysis_manager, users, data_dependency_analysis, map_node);
            std::unordered_set<User*> reads;
            for (auto write : writes) {
                auto read_users = data_dependency_analysis.defines(*write);
                reads.insert(read_users.begin(), read_users.end());
            }

            std::stack<structured_control_flow::ControlFlowNode*> nodes;
            std::queue<structured_control_flow::ControlFlowNode*> queue;

            queue.push(&map_node->root());

            // Create bottom-up order of nodes
            while (!queue.empty()) {
                auto* node = queue.front();
                queue.pop();

                nodes.push(node);

                if (dynamic_cast<structured_control_flow::Sequence*>(node)) {
                    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(node);
                    for (size_t i = 0; i < sequence->size(); ++i) {
                        queue.push(&sequence->at(i).first);
                    }
                } else if (dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    auto* strutured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(node);
                    queue.push(&strutured_loop->root());
                } else if (dynamic_cast<structured_control_flow::While*>(node)) {
                    auto* while_loop = dynamic_cast<structured_control_flow::While*>(node);
                    queue.push(&while_loop->root());
                } else if (dynamic_cast<structured_control_flow::IfElse*>(node)) {
                    auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(node);
                    for (size_t i = 0; i < if_else->size(); ++i) {
                        queue.push(&if_else->at(i).first);
                    }
                }
            }

            std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Expression> cost;

            while (!nodes.empty()) {
                auto* node = nodes.top();
                nodes.pop();

                if (auto loop_node = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    std::unordered_set<User*> loop_reads;
                    bool found = false;
                    symbolic::Expression total_cost = symbolic::one();
                    if (writes.find(users.get_user(loop_node->indvar()->get_name(), loop_node, Use::WRITE, true)) !=
                        writes.end()) {
                        found = true;
                        auto stride = analysis::LoopAnalysis::stride(loop_node);
                        auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
                        auto bound = analysis::LoopAnalysis::canonical_bound(loop_node, assumptions_analysis);
                        auto num_iterations = symbolic::sub(bound, loop_node->init());

                        total_cost = symbolic::mul(num_iterations, cost.at(&loop_node->root()));
                    }
                    cost.insert({node, total_cost});

                } else if (auto sequence_node = dynamic_cast<structured_control_flow::Sequence*>(node)) {
                    symbolic::Expression total_cost = symbolic::one();
                    for (size_t i = 0; i < sequence_node->size(); ++i) {
                        auto& body = sequence_node->at(i).first;
                        total_cost = symbolic::add(total_cost, cost.at(&body));
                    }
                    cost.insert({node, total_cost});
                } else if (auto if_else_node = dynamic_cast<structured_control_flow::IfElse*>(node)) {
                    symbolic::Expression total_cost = symbolic::one();
                    bool is_dynamic = false;
                    if (contains_jump(if_else_node)) {
                        for (size_t i = 0; i < if_else_node->size(); ++i) {
                            auto& condition = if_else_node->at(i).second;
                            for (auto atom : symbolic::atoms(condition)) {
                                if (reads.find(users.get_user(atom->get_name(), if_else_node, Use::READ)) !=
                                    reads.end()) {
                                    is_dynamic = true;
                                    break;
                                }
                            }
                            if (is_dynamic) {
                                break;
                            }
                        }
                        if (is_dynamic) {
                            for (size_t i = 0; i < if_else_node->size(); ++i) {
                                auto& body = if_else_node->at(i).first;
                                symbolic::Symbol if_else_symbol =
                                    symbolic::symbol("if_else_" + std::to_string(if_else_node->element_id()));
                                total_cost = symbolic::mul(symbolic::max(total_cost, cost.at(&body)), if_else_symbol);
                                if_else_symbols.insert(if_else_symbol);
                            }
                        }
                    }
                    cost.insert({node, total_cost});
                } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(node)) {
                    symbolic::Expression total_cost = symbolic::one();

                    auto body_cost = cost.at(&while_loop->root());
                    bool is_dynamic = false;
                    for (auto atom : symbolic::atoms(body_cost)) {
                        if (if_else_symbols.find(atom) != if_else_symbols.end()) {
                            is_dynamic = true;
                            break;
                        }
                    }
                    if (is_dynamic) {
                        symbolic::Symbol while_symbol =
                            symbolic::symbol("while_" + std::to_string(while_loop->element_id()));
                        total_cost = symbolic::mul(while_symbol, body_cost);
                        while_symbols_.insert(while_symbol);
                    }
                    cost.insert({node, total_cost});
                } else {
                    cost.insert({node, symbolic::one()});
                }
            }

            this->balance_of_a_map_.insert({map_node, cost.at(&map_node->root())});
        }
    }
}


std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> DegreesOfKnowledgeAnalysis::
    number_of_maps(const structured_control_flow::Map& node) const {
    if (number_of_maps_.find(&node) == number_of_maps_.end()) {
        throw std::runtime_error("Map node not found in DegreesOfKnowledgeAnalysis.");
    }

    auto bounds = number_of_maps_.at(&node);
    auto atoms = symbolic::atoms(bounds.second);

    if (!symbolic::intersects(atoms, while_symbols_) && symbolic::eq(bounds.first, bounds.second)) {
        if (SymEngine::is_a<SymEngine::Integer>(*bounds.second)) {
            return {bounds.second, DegreesOfKnowledgeClassification::Scalar};
        }

        return {bounds.second, DegreesOfKnowledgeClassification::Bound};
    }

    return {bounds.second, DegreesOfKnowledgeClassification::Unbound};
}

std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> DegreesOfKnowledgeAnalysis::
    size_of_a_map(const structured_control_flow::Map& node) const {
    if (size_of_a_map_.find(&node) == size_of_a_map_.end()) {
        throw std::runtime_error("Map node not found in DegreesOfKnowledgeAnalysis.");
    }

    auto [expr, unbound_symbols] = size_of_a_map_.at(&node);

    if (unbound_symbols.empty() || !symbolic::intersects(symbolic::atoms(expr), unbound_symbols)) {
        if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
            return {expr, DegreesOfKnowledgeClassification::Scalar};
        }

        return {expr, DegreesOfKnowledgeClassification::Bound};
    }

    return {expr, DegreesOfKnowledgeClassification::Unbound};
}

std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> DegreesOfKnowledgeAnalysis::
    load_of_a_map(const structured_control_flow::Map& node) const {
    if (load_of_a_map_.find(&node) == load_of_a_map_.end()) {
        throw std::runtime_error("Map node not found in DegreesOfKnowledgeAnalysis.");
    }

    auto expr = load_of_a_map_.at(&node);
    if (!symbolic::intersects(symbolic::atoms(expr), while_symbols_)) {
        if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
            return {expr, DegreesOfKnowledgeClassification::Scalar};
        }
        return {expr, DegreesOfKnowledgeClassification::Bound};
    }

    return {expr, DegreesOfKnowledgeClassification::Unbound};
}

std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> DegreesOfKnowledgeAnalysis::
    balance_of_a_map(const structured_control_flow::Map& node) const {
    if (balance_of_a_map_.find(&node) == balance_of_a_map_.end()) {
        throw std::runtime_error("Map node not found in DegreesOfKnowledgeAnalysis.");
    }

    auto expr = balance_of_a_map_.at(&node);

    if (!symbolic::intersects(symbolic::atoms(expr), while_symbols_)) {
        if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
            return {expr, DegreesOfKnowledgeClassification::Scalar};
        }
        return {expr, DegreesOfKnowledgeClassification::Bound};
    }

    return {expr, DegreesOfKnowledgeClassification::Unbound};
}

} // namespace analysis
} // namespace sdfg

#include "sdfg/analysis/degrees_of_knowledge_analysis.h"

#include <cstddef>
#include <list>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/analysis/work_depth_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"

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
        auto bound = analysis::LoopAnalysis::canonical_bound(for_loop, assumptions_analysis);
        auto num_iterations = symbolic::sub(bound, for_loop->init());
        num_iterations = symbolic::div(num_iterations, stride);

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
                        if (for_user->parent()->get_parent() == map_node) {
                            for_users.insert(for_user);
                        }
                    }
                }
            }

            for (auto user : for_users) {
                auto defined_by = dependencies.defined_by(user->container());
                auto after = users.all_uses_after(*user);
                auto definition = defined_by.at(user);
                std::unordered_set<User *> intersection;
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
            auto depth = work_depth_analysis.depth(map_node);
            auto while_symbols = work_depth_analysis.while_symbols(depth);

            load_of_a_map_.insert({map_node, {depth, while_symbols}});
        }
    }    
}

bool contains_jump(
    const structured_control_flow::ControlFlowNode* node
) {
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

std::unordered_set<User*> dynamic_writes(Users& users, structured_control_flow::Map* map_node) {
    std::unordered_set<User*> writes;
    writes.insert(users.get_user(map_node->indvar()->get_name(), map_node, Use::WRITE, true, false, false));
    writes.insert(users.get_user(map_node->indvar()->get_name(), map_node, Use::WRITE, false, false, true));

    std::unordered_set<std::string> write_containers;
    for (auto& write : writes) {
        write_containers.insert(write->container());
    }

    auto users_view = UsersView(users, map_node->root());

    bool updated = true;
    while (updated) {
        updated = false;
        auto reads = users_view.reads();
        for (auto read : reads) {
            if (write_containers.find(read->container()) != write_containers.end()) {
                auto element = read->element();
                if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element)) {
                    // TODO: Compare number of iterations instead of individual reads. Since they might be canceled out (e.g. Loop tiling)
                    if (loop->indvar()->get_name() != read->container()) {
                        writes.insert(users.get_user(read->container(), loop, Use::WRITE, true, false, false));
                        writes.insert(users.get_user(read->container(), loop, Use::WRITE, false, false, true));
                        updated = true;
                    }
                } else if (auto if_else = dynamic_cast<data_flow::AccessNode*>(element)) {
                    // TODO: Propagate access node through to the next access node
                } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
                    // TODO: Propagate memlet through to the next access node
                } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(element)) {
                    // TODO: Propagate tasklet through to the next access node
                }
            }
        }
    }
    return writes;
}

void DegreesOfKnowledgeAnalysis::balance_analysis(AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<LoopAnalysis>();
    auto& users = analysis_manager.get<Users>();

    for (auto& loop : loop_analysis.loops()) {
        if (auto* map_node = dynamic_cast<structured_control_flow::Map*>(loop)) {
            auto writes = dynamic_writes(users, map_node);
            std::unordered_set<std::string> write_containers;
            for (auto& write : writes) {
                write_containers.insert(write->container());
            }

            // TODO: create buttom up order of nodes
            // to avoid traversing the whole SDFG multiple times
            // Then create differential depth expression
            std::list<structured_control_flow::ControlFlowNode*> stack;
            stack.push_back(&map_node->root());
            while (!stack.empty()) {
                auto node = stack.front();
                stack.pop_front();
                
                if (auto loop_node = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    stack.push_back(&loop_node->root());
                    
                    bool is_dependent = false;

                    for (auto atom : symbolic::atoms(loop_node->init())) {
                        if (write_containers.find(atom->get_name()) != write_containers.end()) {
                            auto user = users.get_user(atom->get_name(), loop_node, Use::READ, true, false, false);
                            for (auto write : writes) {
                                if (write->container() == atom->get_name()) {
                                    if (users.all_uses_after(*user).find(write) != users.all_uses_after(*user).end()) {
                                        is_dependent = true;
                                        break;                                
                                    }
                                }                                
                            }
                        }
                    }

                    if (!is_dependent) {
                        for (auto atom : symbolic::atoms(loop_node->condition())) {
                            if (symbolic::eq(atom, loop_node->indvar())) {
                                continue;                            
                            }

                            if (write_containers.find(atom->get_name()) != write_containers.end()) {
                                auto user = users.get_user(atom->get_name(), loop_node, Use::READ, false, true, false);
                                for (auto write : writes) {
                                    if (write->container() == atom->get_name()) {
                                        if (users.all_uses_after(*user).find(write) != users.all_uses_after(*user).end()) {
                                            is_dependent = true;
                                            break;                                
                                        }
                                    }                                
                                }
                            }
                        }
                    }
                } else if (auto sequence_node = dynamic_cast<structured_control_flow::Sequence*>(node)) {
                    for (size_t i = 0; i < sequence_node->size(); ++i) {
                        stack.push_back(&sequence_node->at(i).first);
                    }
                } else if (auto if_else_node = dynamic_cast<structured_control_flow::IfElse*>(node)) {
                    for (size_t i = 0; i < if_else_node->size(); i++) {
                        stack.push_back(&if_else_node->at(i).first);
                    }
                } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(node)) {
                    stack.push_back(&while_loop->root());
                } else {
                    // NOP
                }
                
                //TODO: think
            }
        }
    }
}

} // namespace analysis
} // namespace sdfg

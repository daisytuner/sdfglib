#include "sdfg/analysis/degrees_of_knowledge_analysis.h"

#include <cstddef>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/analysis/work_depth_analysis.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/structured_control_flow/block.h"
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
    this->load_analysis();
    this->balance_analysis();
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
            auto& stride = analysis::LoopAnalysis::stride(map_node);
            auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
            auto bound = analysis::LoopAnalysis::canonical_bound(map_node, assumptions_analysis);
            auto num_iterations = symbolic::sub(bound, map_node->init());
            num_iterations = symbolic::div(num_iterations, stride);

            symbolic::SymbolSet unbound_symbols;

            if (SymEngine::is_a<SymEngine::Integer>(num_iterations)) {
                auto integer_iterations = SymEngine::rcp_static_cast<const SymEngine::Integer>(num_iterations);
                size_of_a_map_.insert({map_node, {integer_iterations}, unbound_symbols});
                continue;
            }

            auto atoms = symbolic::atoms(num_iterations);

            auto& users = analysis_manager.get<Users>();
            auto& dependencies = analysis_manager.get<DataDependencyAnalysis>();

            std::unordered_set<ForUser*> for_users;
            for (auto& atom : atoms) {
                for (auto& read : users.reads(atom.get_name())) {
                    if (ForUser* for_user = dynamic_cast<ForUser*>(&read)) {
                        if (for_user->parent() == map_node) {
                            for_users.insert(for_user);
                        }
                    }
                }
            }

            for (auto& user : for_users) {
                auto defined_by = dependencies.defined_by(user);
                auto after = users.all_uses_after(user);
                if (after.contains(defined_by)) {
                    unbound_symbols.insert(symbolic::symbol(user->container()));
                }
            }

            size_of_a_map_.insert({map_node, {num_iterations, unbound_symbols}});
        }
    }
}

void DegreesOfKnowledgeAnalysis::load_of_a_map(AnalysisManager& analysis_manager) {
    auto& work_depth_analysis = analysis_manager.get<WorkDepthAnalysis>();
}

} // namespace analysis
} // namespace sdfg

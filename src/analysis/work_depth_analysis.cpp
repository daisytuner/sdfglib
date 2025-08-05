#include "sdfg/analysis/work_depth_analysis.h"
#include <cstddef>
#include <stack>
#include <string>
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/code_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

WorkDepthAnalysis::WorkDepthAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}


void WorkDepthAnalysis::run(AnalysisManager& analysis_manager) {
    std::stack<structured_control_flow::ControlFlowNode*> nodes;
    std::queue<structured_control_flow::ControlFlowNode*> queue;

    queue.push(&sdfg_.root());

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

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // Calculate work and depth for each node
    while (!nodes.empty()) {
        auto* node = nodes.top();
        nodes.pop();

        symbolic::Expression work;
        symbolic::Expression depth;

        if (dynamic_cast<structured_control_flow::Sequence*>(node)) {
            auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(node);
            work = symbolic::zero();
            depth = symbolic::zero();
            for (size_t i = 0; i < sequence->size(); ++i) {
                const auto& child_work = this->work_.at(&sequence->at(i).first);
                const auto& child_depth = this->depth_.at(&sequence->at(i).first);

                work = symbolic::add(work, child_work);
                depth = symbolic::add(depth, child_depth);
            }
        } else if (dynamic_cast<structured_control_flow::Block*>(node)) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(node);
            int code_nodes = 0;
            for (auto& inst : block->dataflow().nodes()) {
                if (dynamic_cast<data_flow::CodeNode*>(&inst)) {
                    code_nodes++;
                }
            }
            work = symbolic::integer(code_nodes);
            depth = symbolic::integer(code_nodes);
        } else if (dynamic_cast<structured_control_flow::While*>(node)) {
            auto* while_loop = dynamic_cast<structured_control_flow::While*>(node);

            std::string while_symbol_name = "while_" + std::to_string(while_loop->element_id());
            symbolic::Symbol while_symbol = symbolic::symbol(while_symbol_name);
            while_symbols_.insert(while_symbol);
            symbolic::Expression body_work = this->work_.at(&while_loop->root());
            symbolic::Expression body_depth = this->depth_.at(&while_loop->root());

            work = symbolic::mul(while_symbol, body_work);
            depth = symbolic::mul(while_symbol, body_depth);
        } else if (dynamic_cast<structured_control_flow::For*>(node)) {
            auto* for_loop = dynamic_cast<structured_control_flow::For*>(node);
            symbolic::Expression body_work = this->work_.at(&for_loop->root());
            symbolic::Expression body_depth = this->depth_.at(&for_loop->root());

            auto stride = analysis::LoopAnalysis::stride(for_loop);
            auto bound = analysis::LoopAnalysis::canonical_bound(for_loop, assumptions_analysis);
            auto num_iterations = symbolic::sub(bound, for_loop->init());
            num_iterations = symbolic::div(num_iterations, stride);

            work = symbolic::mul(num_iterations, body_work);
            depth = symbolic::mul(num_iterations, body_depth);
        } else if (dynamic_cast<structured_control_flow::Map*>(node)) {
            auto* map_node = dynamic_cast<structured_control_flow::Map*>(node);

            symbolic::Expression body_work = this->work_.at(&map_node->root());
            symbolic::Expression body_depth = this->depth_.at(&map_node->root());

            auto stride = analysis::LoopAnalysis::stride(map_node);
            auto bound = analysis::LoopAnalysis::canonical_bound(map_node, assumptions_analysis);
            auto num_iterations = symbolic::sub(bound, map_node->init());
            num_iterations = symbolic::div(num_iterations, stride);

            work = symbolic::mul(num_iterations, body_work);
            depth = body_depth;
        } else if (dynamic_cast<structured_control_flow::IfElse*>(node)) {
            auto* if_else = dynamic_cast<structured_control_flow::IfElse*>(node);

            work = symbolic::zero();
            depth = symbolic::zero();

            for (size_t i = 0; i < if_else->size(); i++) {
                symbolic::Expression body_work = this->work_.at(&if_else->at(i).first);
                symbolic::Expression body_depth = this->depth_.at(&if_else->at(i).first);

                work = symbolic::max(body_work, work);
                depth = symbolic::max(body_depth, depth);
            }
        } else {
            work = symbolic::zero();
            depth = symbolic::zero();
        }

        this->work_.insert({node, work});
        this->depth_.insert({node, depth});
    }
}

const symbolic::Expression& WorkDepthAnalysis::work(const structured_control_flow::ControlFlowNode* node) const {
    auto it = work_.find(node);
    if (it != work_.end()) {
        return it->second;
    }
    throw std::runtime_error("Work not found for the given node.");
}

const symbolic::Expression& WorkDepthAnalysis::depth(const structured_control_flow::ControlFlowNode* node) const {
    auto it = depth_.find(node);
    if (it != depth_.end()) {
        return it->second;
    }
    throw std::runtime_error("Depth not found for the given node.");
}

const symbolic::SymbolSet WorkDepthAnalysis::while_symbols(symbolic::Expression expression) const {
    auto atoms = symbolic::atoms(expression);
    symbolic::SymbolSet contained_symbols;
    for (const auto& atom : atoms) {
        if (while_symbols_.find(atom) != while_symbols_.end()) {
            contained_symbols.insert(atom);
        }
    }
    return contained_symbols;
}

} // namespace analysis
} // namespace sdfg

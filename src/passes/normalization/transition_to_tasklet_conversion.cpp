#include "sdfg/passes/normalization/transition_to_tasklet_conversion.h"

namespace sdfg {
namespace passes {

TransitionToTaskletConversion::TransitionToTaskletConversion(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager){};

bool TransitionToTaskletConversion::can_be_applied(const symbolic::Symbol& lhs,
                                                   const symbolic::Expression& rhs) const {
    return true;
};

void TransitionToTaskletConversion::apply(structured_control_flow::Block& block,
                                          const symbolic::Symbol& lhs,
                                          const symbolic::Expression& rhs) {
    auto& sdfg = builder_.subject();

    auto& input_node = builder_.symbolic_expression_to_dataflow(block, rhs);
    auto& tasklet = builder_.add_tasklet(
        block, data_flow::TaskletCode::assign,
        {"_out", static_cast<const types::Scalar&>(sdfg.type(lhs->get_name()))},
        {{"_in", static_cast<const types::Scalar&>(sdfg.type(input_node.data()))}});
    auto& output_node = builder_.add_access(block, lhs->get_name());
    builder_.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});
    builder_.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
};

bool TransitionToTaskletConversion::accept(structured_control_flow::Sequence& parent,
                                           structured_control_flow::Sequence& node) {
    bool applied = false;
    for (size_t i = 0; i < node.size();) {
        auto& transition = node.at(i).second;
        if (transition.assignments().empty()) {
            i++;
            continue;
        }

        structured_control_flow::ControlFlowNode* next = nullptr;
        if (i < node.size() - 1) {
            next = &node.at(i + 1).first;
        }

        std::unordered_set<std::string> erased_symbols;
        for (auto& entry : transition.assignments()) {
            if (this->can_be_applied(entry.first, entry.second)) {
                if (next != nullptr) {
                    auto& new_block = builder_.add_block_before(node, *next, {}).first;
                    this->apply(new_block, entry.first, entry.second);
                } else {
                    auto& new_block = builder_.add_block(node);
                    this->apply(new_block, entry.first, entry.second);
                }
                erased_symbols.insert(entry.first->get_name());
            }
        }
        for (auto& symbol : erased_symbols) {
            transition.assignments().erase(symbolic::symbol(symbol));
        }
        if (!erased_symbols.empty()) {
            applied = true;
        }

        i = i + erased_symbols.size() + 1;
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

#include "sdfg/passes/symbolic/condition_propagation.h"

#include "sdfg/analysis/users.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace passes {

bool BackwardConditionPropagation::eliminate_condition(builder::StructuredSDFGBuilder& builder,
                                                       structured_control_flow::Sequence& root,
                                                       structured_control_flow::IfElse& match,
                                                       structured_control_flow::For& loop,
                                                       const symbolic::Condition& condition) {
    auto loop_indvar = loop.indvar();
    auto loop_init = loop.init();
    auto loop_condition = loop.condition();

    // If loop condition equals true => condition true, we can eliminate the match
    auto assumption_1 = loop_condition;
    auto assumption_2 = symbolic::subs(loop_condition, loop_indvar, loop_init);
    if (symbolic::eq(assumption_1, condition) || symbolic::eq(assumption_2, condition)) {
        auto& new_seq = builder.add_sequence_before(root, match).first;
        builder.insert(loop, match.at(0).first, new_seq, match.debug_info());
        builder.remove_child(root, match);

        return true;
    }

    return false;
};

BackwardConditionPropagation::BackwardConditionPropagation()
    : Pass() {

      };

std::string BackwardConditionPropagation::name() { return "BackwardConditionPropagation"; };

bool BackwardConditionPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                            analysis::AnalysisManager& analysis_manager) {
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
                if (!child.second.assignments().empty()) {
                    continue;
                }
                auto& body = child.first;
                if (auto match = dynamic_cast<structured_control_flow::IfElse*>(&body)) {
                    // Must be a simple if
                    if (match->size() != 1) {
                        continue;
                    }
                    auto branch = match->at(0);
                    auto& condition = branch.second;

                    // Branch must contain a single for loop
                    auto& root = branch.first;
                    if (root.size() != 1) {
                        continue;
                    }
                    if (dynamic_cast<structured_control_flow::For*>(&root.at(0).first) == nullptr) {
                        continue;
                    }
                    auto& loop = dynamic_cast<structured_control_flow::For&>(root.at(0).first);
                    bool eliminated =
                        this->eliminate_condition(builder, *sequence_stmt, *match, loop, condition);
                    if (eliminated) {
                        applied = true;
                    }
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
        } else if (auto sloop_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

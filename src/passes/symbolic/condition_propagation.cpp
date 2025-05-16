#include "sdfg/passes/symbolic/condition_propagation.h"

#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace passes {

ForwardConditionPropagation::ForwardConditionPropagation()
    : Pass(){

      };

std::string ForwardConditionPropagation::name() { return "ForwardConditionPropagation"; };

bool ForwardConditionPropagation::propagate_condition(builder::StructuredSDFGBuilder& builder,
                                                      analysis::AnalysisManager& analysis_manager,
                                                      structured_control_flow::Sequence& parent,
                                                      const symbolic::Condition& condition) {
    auto& sdfg = builder.subject();
    bool applied = false;

    // Condition's symbols must not be written to in parent
    std::unordered_set<std::string> symbols;
    for (auto& atom : symbolic::atoms(condition)) {
        auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
        if (symbolic::is_pointer(sym)) {
            return false;
        }
        if (!symbolic::is_nvptx(sym)) {
            if (!dynamic_cast<const types::Scalar*>(&sdfg.type(sym->get_name()))) {
                return false;
            }
        }
        symbols.insert(sym->get_name());
    }

    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, parent);
    for (auto& sym : symbols) {
        if (!body_users.writes(sym).empty()) {
            return false;
        }
    }

    std::list<structured_control_flow::ControlFlowNode*> queue = {&parent};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();

        // Pattern-Match
        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(curr)) {
            // Case 1: Assignment - sym = true | false -> sym = condition
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                auto& transition = child.second;
                for (auto& entry : transition.assignments()) {
                    if (symbolic::is_true(entry.second)) {
                        entry.second = condition;
                        applied = true;
                    } else if (symbolic::is_false(entry.second)) {
                        entry.second = symbolic::Not(condition);
                        applied = true;
                    }
                }
                queue.push_back(&child.first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(curr)) {
            // Case 2: if (condition) | if (!condition) -> if (true) | if (false)
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                auto branch = if_else_stmt->at(i);
                auto& nested_condition = branch.second;
                if (symbolic::eq(nested_condition, condition)) {
                    nested_condition = symbolic::__true__();
                    applied = true;
                } else if (symbolic::eq(nested_condition, symbolic::Not(condition))) {
                    nested_condition = symbolic::__false__();
                    applied = true;
                }
            }
        }

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(curr)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                queue.push_back(&child.first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(curr)) {
            for (size_t j = 0; j < if_else_stmt->size(); j++) {
                if (symbolic::is_false(if_else_stmt->at(j).second)) {
                    continue;
                }
                queue.push_back(&if_else_stmt->at(j).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(curr)) {
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(curr)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kernel_stmt = dynamic_cast<structured_control_flow::Kernel*>(curr)) {
            queue.push_back(&kernel_stmt->root());
        }
    }

    return applied;
};

bool ForwardConditionPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                           analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    std::list<structured_control_flow::ControlFlowNode*> queue = {&sdfg.root()};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(curr)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                queue.push_back(&child.first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(curr)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                if (symbolic::is_false(if_else_stmt->at(i).second)) {
                    continue;
                }

                auto branch = if_else_stmt->at(i);
                applied |= this->propagate_condition(builder, analysis_manager, branch.first,
                                                     branch.second);
            }
            for (size_t j = 0; j < if_else_stmt->size(); j++) {
                queue.push_back(&if_else_stmt->at(j).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(curr)) {
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(curr)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kernel_stmt = dynamic_cast<structured_control_flow::Kernel*>(curr)) {
            queue.push_back(&kernel_stmt->root());
        }
    }

    return applied;
};

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
        deepcopy::StructuredSDFGDeepCopy copy(builder, new_seq, loop);
        copy.copy();
        builder.remove_child(root, match);

        return true;
    }

    return false;
};

BackwardConditionPropagation::BackwardConditionPropagation()
    : Pass(){

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
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

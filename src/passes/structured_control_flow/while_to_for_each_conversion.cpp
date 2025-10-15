#include "sdfg/passes/structured_control_flow/while_to_for_each_conversion.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace passes {

bool WhileToForEachConversion::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop
) {
    auto& sdfg = builder.subject();
    auto& body = loop.root();
    if (loop.root().size() < 2) {
        return false;
    }

    // Identify break and continue conditions
    auto end_of_body = body.at(body.size() - 1);
    if (end_of_body.second.size() > 0) {
        return false;
    }
    auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&end_of_body.first);
    if (!if_else_stmt || if_else_stmt->size() != 2) {
        return false;
    }

    bool first_is_continue = false;
    bool first_is_break = false;
    auto& first_branch = if_else_stmt->at(0).first;
    if (first_branch.size() != 1) {
        return false;
    }
    auto first_condition = if_else_stmt->at(0).second;
    if (dynamic_cast<structured_control_flow::Break*>(&first_branch.at(0).first)) {
        first_is_break = true;
    } else if (dynamic_cast<structured_control_flow::Continue*>(&first_branch.at(0).first)) {
        first_is_continue = true;
    }
    if (!first_is_break && !first_is_continue) {
        return false;
    }

    bool second_is_continue = false;
    bool second_is_break = false;
    auto& second_branch = if_else_stmt->at(1).first;
    if (second_branch.size() != 1) {
        return false;
    }
    auto second_condition = if_else_stmt->at(1).second;
    if (dynamic_cast<structured_control_flow::Break*>(&second_branch.at(0).first)) {
        second_is_break = true;
    } else if (dynamic_cast<structured_control_flow::Continue*>(&second_branch.at(0).first)) {
        second_is_continue = true;
    }
    if (!second_is_break && !second_is_continue) {
        return false;
    }
    if (first_is_break == second_is_break) {
        return false;
    }

    if (symbolic::atoms(first_condition).size() != 2 || 
        symbolic::atoms(second_condition).size() != 2) {
        return false;
    }

    // Criterion: Continue condition is an equality between two symbols
    auto sym1 = *symbolic::atoms(first_condition).begin();
    auto sym2 = *(++symbolic::atoms(first_condition).begin());
    auto cont_condition = symbolic::Ne(sym1, sym2);
    auto cont_condition_alt = symbolic::Eq(symbolic::__false__(), symbolic::Eq(sym1, sym2));
    if (first_is_continue && !(symbolic::eq(cont_condition, first_condition) || symbolic::eq(cont_condition_alt, first_condition))) {
        return false;
    }
    if (second_is_continue && !(symbolic::eq(cont_condition, second_condition) || symbolic::eq(cont_condition_alt, second_condition))) {
        return false;
    }
    if (!symbolic::eq(first_condition, symbolic::Not(second_condition))) {
        return false;
    }

    // We know that the while body ends with an if-else continue-break structure
    // We now check that there is exactly one iterator, which is moved once per
    // iteration
    // All other variables in the condition must be constants

    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(all_users, body);

    // Candidates: all symbols in the condition which are pointers
    analysis::User* update = nullptr;
    for (auto& sym : symbolic::atoms(first_condition)) {
        if (symbolic::eq(sym, symbolic::__nullptr__())) {
            continue;
        }
        auto& type = sdfg.type(sym->get_name());
        if (!dynamic_cast<const types::Pointer*>(&type)) {
            return false;
        }

        auto moves = body_users.moves(sym->get_name());
        // Not an iterator
        if (moves.empty()) {
            continue;
        }
        // Not well-formed
        if (moves.size() > 1) {
            return false;
        }
        // Exactly one iterator
        if (update != nullptr) {
            return false;
        }
        update = moves.at(0);
    }
    if (update == nullptr) {
        return false;
    }
    auto iterator = symbolic::symbol(update->container());

    // Criterion: Update is a dereference memlet
    // iterator = *ptr
    auto move_dst = dynamic_cast<data_flow::AccessNode*>(update->element());
    auto& graph = move_dst->get_parent();
    auto& block = static_cast<const structured_control_flow::Block&>(*graph.get_parent());
    auto& move_edge = *graph.in_edges(*move_dst).begin();
    auto& move_src = static_cast<data_flow::AccessNode&>(move_edge.src());
    if (move_edge.type() != data_flow::MemletType::Dereference_Src) {
        return false;
    }

    // Criterion: Update happens right before the condition
    if (body.index(block) != body.size() - 2) {
        return false;
    }

    // No other continue, break or return inside loop body
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&loop.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (dynamic_cast<const structured_control_flow::Break*>(current)) {
            return false;
        } else if (dynamic_cast<const structured_control_flow::Continue*>(current)) {
            return false;
        } else if (dynamic_cast<const structured_control_flow::Return*>(current)) {
            return false;
        }

        if (auto sequence_stmt = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            // Ignore the if_else_stmt
            if (if_else == if_else_stmt) {
                continue;
            }
            for (size_t i = 0; i < if_else->size(); i++) {
                queue.push_back(&if_else->at(i).first);
            }
        }
    }

    return true;
}

void WhileToForEachConversion::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& parent,
    structured_control_flow::While& loop
) {
    auto& sdfg = builder.subject();
    auto& body = loop.root();

    // Identify break and continue conditions
    auto last_element = body.at(body.size() - 1);
    auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&last_element.first);

    auto first_condition = if_else_stmt->at(0).second;
    auto second_condition = if_else_stmt->at(1).second;

    bool second_is_break = false;
    auto& second_branch = if_else_stmt->at(1).first;
    if (dynamic_cast<structured_control_flow::Break*>(&second_branch.at(0).first)) {
        second_is_break = true;
    }
    
    // Identify iterator
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(all_users, body);
    analysis::User* update = nullptr;
    for (auto& sym : symbolic::atoms(first_condition)) {
        if (symbolic::eq(sym, symbolic::__nullptr__())) {
            continue;
        }
        auto moves = body_users.moves(sym->get_name());
        if (moves.size() == 1) {
            update = moves.at(0);
            break;
        }
    }
    symbolic::Symbol iterator = symbolic::symbol(update->container());
    symbolic::Symbol end = SymEngine::null;
    for (auto& atom : symbolic::atoms(second_condition)) {
        if (atom->get_name() != iterator->get_name()) {
            end = atom;
            break;
        }
    }

    // Identify update / move statement
    auto move_dst = dynamic_cast<data_flow::AccessNode*>(update->element());
    auto& graph = move_dst->get_parent();
    auto& move_edge = *graph.in_edges(*move_dst).begin();
    auto& move_src = static_cast<data_flow::AccessNode&>(move_edge.src());
    symbolic::Symbol update_ptr = symbolic::symbol(move_src.data());

    // Remove update from block
    builder.remove_child(body, body.size() - 2);
    // Remove the if-else statement
    builder.remove_child(body, body.size() - 1);

    // find index of while
    int while_index = parent.index(loop);
    auto& transition = parent.at(while_index).second;

    // Create for-each loop
    auto& for_each = builder.add_for_each_after(
        parent,
        loop,
        iterator,
        end,
        update_ptr,
        SymEngine::null,
        transition.assignments(),
        loop.debug_info()
    );
    builder.move_children(body, for_each.root());

    // Remove while loop
    builder.remove_child(parent, while_index);
};

WhileToForEachConversion::WhileToForEachConversion()
    : Pass() {

      };

std::string WhileToForEachConversion::name() { return "WhileToForEachConversion"; };

bool WhileToForEachConversion::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Add children to queue
        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                if (auto match = dynamic_cast<structured_control_flow::While*>(&sequence_stmt->at(i).first)) {
                    if (this->can_be_applied(builder, analysis_manager, *match)) {
                        this->apply(builder, analysis_manager, *sequence_stmt, *match);
                        return true;
                    }
                }

                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else->size(); i++) {
                queue.push_back(&if_else->at(i).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        } else if (auto for_each = dynamic_cast<structured_control_flow::ForEach*>(current)) {
            queue.push_back(&for_each->root());
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg

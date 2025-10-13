#include "sdfg/passes/dataflow/reference_propagation.h"

#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

ReferencePropagation::ReferencePropagation()
    : Pass() {

      };

std::string ReferencePropagation::name() { return "ReferencePropagation"; };

bool ReferencePropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    // Replaces all views
    auto& users = analysis_manager.get<analysis::Users>();
    std::unordered_set<std::string> reduced;
    for (auto& container : sdfg.containers()) {
        if (reduced.find(container) != reduced.end()) {
            continue;
        }
        if (!sdfg.is_transient(container)) {
            continue;
        }

        // By definition, a view is a pointer
        auto& type = sdfg.type(container);
        if (type.type_id() != types::TypeID::Pointer) {
            continue;
        }

        // Criterion: Must have at least one move
        auto moves = users.moves(container);
        if (moves.empty()) {
            continue;
        }

        // Eliminate views
        auto uses = users.uses(container);
        for (auto& move : moves) {
            auto& access_node = static_cast<data_flow::AccessNode&>(*move->element());
            auto& dataflow = *move->parent();
            auto& move_edge = *dataflow.in_edges(access_node).begin();

            // Criterion: Must be a reference memlet
            if (move_edge.type() != data_flow::MemletType::Reference) {
                continue;
            }

            // Retrieve underlying container
            auto& viewed_node = static_cast<const data_flow::AccessNode&>(move_edge.src());
            auto& viewed_container = viewed_node.data();

            // Criterion: Must not be constant data
            if (helpers::is_number(viewed_container) || symbolic::is_nullptr(symbolic::symbol(viewed_container))) {
                continue;
            }

            // Criterion: Must not be address of
            auto& move_subset = move_edge.subset();
            if (move_subset.empty()) {
                continue;
            }

            // Replace all uses of the view by the pointer
            for (auto& user : uses) {
                // Criterion: Cannot be a move
                if (user->use() == analysis::Use::MOVE) {
                    continue;
                }

                // Criterion: Must be an access node
                if (!dynamic_cast<data_flow::AccessNode*>(user->element())) {
                    continue;
                }

                // Criterion: Must be dominated by the move
                if (!users.dominates(*move, *user)) {
                    continue;
                }

                // Criterion: No reassignment of pointer or view in between
                auto uses_between = users.all_uses_between(*move, *user);
                bool unsafe = false;
                for (auto& use : uses_between) {
                    if (use->use() != analysis::Use::MOVE) {
                        continue;
                    }
                    // Pointer is not constant
                    if (use->container() == viewed_container) {
                        unsafe = true;
                        break;
                    }
                    // View is reassigned
                    if (use->container() == container) {
                        unsafe = true;
                        break;
                    }
                }
                if (unsafe) {
                    continue;
                }

                auto& user_node = static_cast<data_flow::AccessNode&>(*user->element());

                // Simple case: No arithmetic on pointer, just replace container
                if (move_subset.size() == 1 && symbolic::eq(move_subset[0], symbolic::zero())) {
                    user_node.data() = viewed_container;
                    applied = true;
                    reduced.insert(viewed_container);
                    continue;
                }

                // General case: Arithmetic on pointer, need to update memlet subsets

                // Criterion: Must be computational memlets
                // Criterion: No type casting

                auto& deref_type = move_edge.result_type(builder.subject());
                sdfg::types::Pointer ref_type(static_cast<const types::IType&>(deref_type));

                bool safe = true;
                auto user_graph = user->parent();
                for (auto& oedge : user_graph->out_edges(user_node)) {
                    if (oedge.type() != data_flow::MemletType::Computational) {
                        safe = false;
                        break;
                    }
                    if (oedge.subset().empty()) {
                        safe = false;
                        break;
                    }
                    if (oedge.base_type() != ref_type) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) {
                    continue;
                }
                for (auto& iedge : user_graph->in_edges(user_node)) {
                    if (iedge.type() != data_flow::MemletType::Computational) {
                        safe = false;
                        break;
                    }
                    if (iedge.subset().empty()) {
                        safe = false;
                        break;
                    }
                    if (iedge.base_type() != ref_type) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) {
                    continue;
                }

                // Propagate pointer type

                // Step 1: Replace container
                user_node.data() = viewed_container;

                // Step 2: Update edges
                for (auto& oedge : user_graph->out_edges(user_node)) {
                    // Compute new subset
                    data_flow::Subset new_subset;
                    for (auto dim : move_subset) {
                        new_subset.push_back(dim);
                    }

                    auto old_subset = oedge.subset();

                    // Handle first trailing dimensions
                    auto& trail_dim = old_subset.front();
                    auto& current_dim = new_subset.back();
                    auto new_dim = symbolic::add(current_dim, trail_dim);
                    new_subset.back() = new_dim;
                    old_subset.erase(old_subset.begin());

                    // Add remaining trailing dimensions
                    for (auto dim : old_subset) {
                        new_subset.push_back(dim);
                    }

                    // Build new type
                    if (move_subset.size() == 1) {
                        oedge.set_subset(new_subset);
                    } else {
                        // Case 2: multi-dimensional subset
                        oedge.set_subset(new_subset);
                        oedge.set_base_type(move_edge.base_type());
                    }
                }

                for (auto& iedge : user_graph->in_edges(user_node)) {
                    // Compute new subset
                    data_flow::Subset new_subset;
                    for (auto dim : move_subset) {
                        new_subset.push_back(dim);
                    }

                    auto old_subset = iedge.subset();

                    // Handle first trailing dimensions
                    auto& trail_dim = old_subset.front();
                    auto& current_dim = new_subset.back();
                    auto new_dim = symbolic::add(current_dim, trail_dim);
                    new_subset.back() = new_dim;
                    old_subset.erase(old_subset.begin());

                    // Add remaining trailing dimensions
                    for (auto dim : old_subset) {
                        new_subset.push_back(dim);
                    }

                    // Case 1: 1D subset, "original pointer is shifted"
                    if (move_subset.size() == 1) {
                        iedge.set_subset(new_subset);
                    } else {
                        // Case 2: multi-dimensional subset
                        iedge.set_subset(new_subset);
                        iedge.set_base_type(move_edge.base_type());
                    }
                }

                applied = true;
                reduced.insert(viewed_container);
            }
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg

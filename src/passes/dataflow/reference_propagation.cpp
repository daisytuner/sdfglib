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

        // Criterion: No sub-views (will be eliminated iteratively)
        if (users.views(container).size() > 0) {
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
                // Criterion: Must be an access node
                if (!dynamic_cast<data_flow::AccessNode*>(user->element())) {
                    continue;
                }

                // Criterion: Must be dominated by the move
                if (!users.dominates(*move, *user)) {
                    continue;
                }
                // Criterion: Pointer and view are constant
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

                auto& use_node = static_cast<data_flow::AccessNode&>(*user->element());

                // Criterion: Must be a typed-pointer at this point, no empty subset / constant data
                auto& base_type = move_edge.base_type();
                auto& pointer_type = dynamic_cast<const types::Pointer&>(base_type);

                // Criterion: Must strictly propagate into computational memlets
                // Criterion: Types must match
                auto use_graph = user->parent();

                auto& result_base_type = types::infer_type(builder.subject(), move_edge.base_type(), move_subset);
                sdfg::types::Pointer result_type(static_cast<const types::IType&>(result_base_type));

                bool safe = true;
                for (auto& oedge : use_graph->out_edges(use_node)) {
                    if (oedge.type() != data_flow::MemletType::Computational) {
                        safe = false;
                        break;
                    }
                    if (result_base_type.type_id() != types::TypeID::Pointer && oedge.base_type() != result_type) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) {
                    continue;
                }
                for (auto& iedge : use_graph->in_edges(use_node)) {
                    if (iedge.type() != data_flow::MemletType::Computational) {
                        safe = false;
                        break;
                    }
                    if (result_base_type.type_id() != types::TypeID::Pointer && iedge.base_type() != result_type) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) {
                    continue;
                }

                // Propagate pointer type

                // Step 1: Replace container
                use_node.data() = viewed_container;

                // Step 2: Update edges
                for (auto& oedge : use_graph->out_edges(use_node)) {
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

                for (auto& iedge : use_graph->in_edges(use_node)) {
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

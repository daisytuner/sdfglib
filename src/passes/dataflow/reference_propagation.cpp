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
        if (sdfg.is_external(container)) {
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

            // Criterion: Must not be raw memory address
            if (helpers::is_number(viewed_container) || symbolic::is_nullptr(symbolic::symbol(viewed_container))) {
                continue;
            }

            // Criterion: Must not be reinterpret cast
            auto& move_subset = move_edge.subset();
            if (move_subset.empty()) {
                continue;
            }

            // Replace all uses of the view by the pointer
            for (auto& user : uses) {
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

                // Can only replace access nodes
                if (!dynamic_cast<data_flow::AccessNode*>(user->element())) {
                    continue;
                }
                auto& use_node = static_cast<data_flow::AccessNode&>(*user->element());

                auto use_graph = user->parent();

                // Criterion: Must be a typed-pointer
                auto& base_type = move_edge.base_type();
                if (base_type.type_id() == types::TypeID::Pointer) {
                    auto& element_type = static_cast<const types::Pointer&>(base_type).pointee_type();
                    if (element_type.type_id() == types::TypeID::Scalar) {
                        if (element_type.primitive_type() == types::PrimitiveType::Int8) {
                            continue;
                        }
                    }
                }

                // Criterion: Must be a computational memlet
                bool computational = true;
                for (auto& oedge : use_graph->out_edges(use_node)) {
                    if (oedge.type() != data_flow::MemletType::Computational || oedge.has_range()) {
                        computational = false;
                        break;
                    }
                }
                if (!computational) {
                    continue;
                }
                for (auto& iedge : use_graph->in_edges(use_node)) {
                    if (iedge.type() != data_flow::MemletType::Computational || iedge.has_range()) {
                        computational = false;
                        break;
                    }
                }
                if (!computational) {
                    continue;
                }

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

                    // Pad with zeros until scalar type
                    auto inferred_type = &types::infer_type(builder.subject(), move_edge.base_type(), new_subset);
                    while (inferred_type->type_id() != types::TypeID::Scalar) {
                        new_subset.push_back(symbolic::zero());
                        inferred_type = &types::infer_type(builder.subject(), move_edge.base_type(), new_subset);
                    }

                    oedge.set_subset(new_subset);
                    oedge.set_base_type(move_edge.base_type());
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

                    // Pad with zeros until scalar type
                    auto inferred_type = &types::infer_type(builder.subject(), move_edge.base_type(), new_subset);
                    while (inferred_type->type_id() != types::TypeID::Scalar) {
                        new_subset.push_back(symbolic::zero());
                        inferred_type = &types::infer_type(builder.subject(), move_edge.base_type(), new_subset);
                    }

                    iedge.set_subset(new_subset);
                    iedge.set_base_type(move_edge.base_type());
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

#include "sdfg/passes/dataflow/view_propagation.h"

#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

ViewPropagation::ViewPropagation()
    : Pass() {

      };

std::string ViewPropagation::name() { return "ViewPropagation"; };

bool ViewPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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
        if (!dynamic_cast<const types::Pointer*>(&type)) {
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
            // Location of where the view is created
            auto& access_node = dynamic_cast<data_flow::AccessNode&>(*move->element());
            auto& dataflow = *move->parent();
            auto& move_edge = *dataflow.in_edges(access_node).begin();
            auto& move_subset = move_edge.subset();

            // Retrieve underlying container
            auto& viewed_node = dynamic_cast<const data_flow::AccessNode&>(move_edge.src());
            auto& viewed_container = viewed_node.data();

            // Criterion: Must not be raw memory address
            if (helpers::is_number(viewed_container) || symbolic::is_nullptr(symbolic::symbol(viewed_container))) {
                continue;
            }

            // Criterion: Must not be nested pointer
            if (move_edge.dst_conn() == "void" && !move_subset.empty()) {
                continue;
            }

            // Criterion: Must not be reinterpret cast
            auto viewed_type = &sdfg.type(viewed_container);
            if (move_edge.src_conn() == "void") {
                viewed_type = &types::infer_type(sdfg, *viewed_type, move_subset);
            }
            types::Pointer final_type(*viewed_type);
            if (type != final_type) {
                continue;
            }

            // Replace all uses of the view by the original container
            for (auto& user : uses) {
                // Criterion: Must be read or write
                if (user->use() != analysis::Use::READ && user->use() != analysis::Use::WRITE) {
                    continue;
                }
                // Criterion: Must be dominated by the move
                if (!users.dominates(*move, *user)) {
                    continue;
                }
                // Criterion: Safety - View is not reassigned between the move and the user
                auto uses_between = users.all_uses_between(*move, *user);
                bool unsafe = false;
                for (auto& use : uses_between) {
                    if (use->use() != analysis::Use::MOVE) {
                        continue;
                    }
                    // View is not constant
                    if (use->container() == viewed_container) {
                        unsafe = true;
                        break;
                    }
                    // Another alias of the view
                    if (use->container() == container) {
                        unsafe = true;
                        break;
                    }

                    // Raw memory access
                    auto& use_node = dynamic_cast<data_flow::AccessNode&>(*use->element());
                    auto& use_graph = *use->parent();
                    auto& use_edge = *use_graph.in_edges(use_node).begin();
                    auto& src_node = dynamic_cast<data_flow::AccessNode&>(use_edge.src());
                    if (helpers::is_number(src_node.data())) {
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

                // Step 1: Replace container
                use_node.data() = viewed_container;

                // Step 2: Update edges
                auto use_graph = user->parent();
                for (auto& oedge : use_graph->out_edges(use_node)) {
                    // Compute new subset
                    data_flow::Subset new_subset;

                    // Add leading dimensions from move
                    if (move_edge.src_conn() == "void") {
                        for (size_t i = 0; i < move_subset.size(); i++) {
                            new_subset.push_back(move_subset[i]);
                        }
                    }

                    auto old_subset = oedge.subset();

                    // Handle first trailing dimensions
                    if (new_subset.empty()) {
                        auto& trail_dim = old_subset.front();
                        if (!symbolic::eq(trail_dim, symbolic::zero())) {
                            throw std::runtime_error("View propagation not implemented for non-void source connection");
                        }
                        old_subset.erase(old_subset.begin());
                    } else {
                        auto& trail_dim = old_subset.front();
                        auto& current_dim = new_subset.back();
                        auto new_dim = symbolic::add(current_dim, trail_dim);
                        new_subset.back() = new_dim;
                        old_subset.erase(old_subset.begin());
                    }

                    // Add remaining trailing dimensions
                    for (auto& dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                    oedge.set_subset(new_subset);
                }
                for (auto& iedge : use_graph->in_edges(use_node)) {
                    // Compute new subset
                    data_flow::Subset new_subset;

                    // Add leading dimensions from move
                    if (move_edge.src_conn() == "void") {
                        for (size_t i = 0; i < move_subset.size(); i++) {
                            new_subset.push_back(move_subset[i]);
                        }
                    }

                    auto old_subset = iedge.subset();

                    // Handle first trailing dimensions
                    if (new_subset.empty()) {
                        auto& trail_dim = old_subset.front();
                        if (!symbolic::eq(trail_dim, symbolic::zero())) {
                            throw std::runtime_error("View propagation not implemented for non-void source connection");
                        }
                        old_subset.erase(old_subset.begin());
                    } else {
                        auto& trail_dim = old_subset.front();
                        auto& current_dim = new_subset.back();
                        auto new_dim = symbolic::add(current_dim, trail_dim);
                        new_subset.back() = new_dim;
                        old_subset.erase(old_subset.begin());
                    }

                    // Add remaining trailing dimensions
                    for (auto& dim : old_subset) {
                        new_subset.push_back(dim);
                    }
                    iedge.set_subset(new_subset);
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

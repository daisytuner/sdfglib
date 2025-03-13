#include "sdfg/passes/dataflow/view_propagation.h"

namespace sdfg {
namespace passes {

ViewPropagation::ViewPropagation()
    : Pass(){

      };

std::string ViewPropagation::name() { return "ViewPropagation"; };

bool ViewPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                               analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    // Replaces all views by their original pointers
    auto& users = analysis_manager.get<analysis::Users>();
    std::unordered_set<std::string> reduced;
    for (auto& container : sdfg.containers()) {
        if (reduced.find(container) != reduced.end()) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Pointer*>(&type)) {
            continue;
        }
        if (sdfg.is_external(container)) {
            continue;
        }

        // Collect all moves of the pointer
        auto moves = users.moves(container);
        if (moves.empty() || !users.views(container).empty()) {
            continue;
        }
        auto uses = users.uses(container);
        for (auto& move : moves) {
            // Location of where the view is created
            auto& access_node = dynamic_cast<data_flow::AccessNode&>(*move->element());
            auto& graph = *move->parent();
            auto& edge = *graph.in_edges(access_node).begin();

            // The original pointer
            auto& viewed_node = dynamic_cast<const data_flow::AccessNode&>(edge.src());
            auto& viewed_container = viewed_node.data();
            auto& viewed_subset = edge.subset();
            if (edge.src_conn() != "void") {
                continue;
            }
            if (symbolic::is_pointer(symbolic::symbol(viewed_container))) {
                continue;
            }
            types::Pointer viewed_type(
                types::infer_type(sdfg, sdfg.type(viewed_container), viewed_subset));
            if (viewed_type != type) {
                continue;
            }

            // Iterate over all uses of the viewing container
            // Replace the view by the original pointer if possible
            for (auto& user : uses) {
                if (user->use() == analysis::Use::MOVE || user->use() == analysis::Use::VIEW) {
                    continue;
                }
                // Criterion: The assignment of the view must dominate the use of the view
                if (!users.dominates(*move, *user)) {
                    continue;
                }
                // Criterion: No pointer operations between the assignment and the use
                auto uses_between = users.all_uses_between(*move, *user);
                bool moving_pointers = false;
                for (auto& use : uses_between) {
                    if (use->use() != analysis::Use::MOVE) {
                        continue;
                    }
                    // Viewed container is not constant
                    if (use->container() == viewed_container) {
                        moving_pointers = true;
                        break;
                    }
                    // Moved container is not constant
                    if (use->container() == container) {
                        moving_pointers = true;
                        break;
                    }

                    // Unsafe pointer operations
                    auto& move_node = dynamic_cast<data_flow::AccessNode&>(*use->element());
                    auto& move_graph = *use->parent();
                    auto& move_edge = *move_graph.in_edges(move_node).begin();
                    auto& view_node = dynamic_cast<data_flow::AccessNode&>(move_edge.src());
                    if (move_edge.dst_conn() == "void" ||
                        symbolic::is_pointer(symbolic::symbol(view_node.data()))) {
                        moving_pointers = true;
                        break;
                    }
                }
                if (moving_pointers) {
                    continue;
                }

                // Replace the view by the original pointer
                if (auto use_node = dynamic_cast<data_flow::AccessNode*>(user->element())) {
                    auto use_graph = user->parent();
                    use_node->data() = viewed_container;

                    // Update subsets
                    for (auto& oedge : use_graph->out_edges(*use_node)) {
                        auto& old_subset = oedge.subset();

                        // Add leading dimensions
                        data_flow::Subset new_subset(viewed_subset.begin(),
                                                     viewed_subset.end() - 1);

                        // Accumulate overlapping dimension
                        auto& dim = old_subset[0];
                        auto& alias_dim = viewed_subset.back();
                        auto new_dim = symbolic::add(alias_dim, dim);
                        new_subset.push_back(new_dim);

                        // Add trailing dimensions
                        for (size_t i = 1; i < old_subset.size(); i++) {
                            auto& old_dim = old_subset[i];
                            new_subset.push_back(old_dim);
                        }

                        oedge.subset() = new_subset;
                    }
                    for (auto& iedge : use_graph->in_edges(*use_node)) {
                        auto& old_subset = iedge.subset();

                        // Add leading dimensions
                        data_flow::Subset new_subset(viewed_subset.begin(),
                                                     viewed_subset.end() - 1);

                        // Accumulate overlapping dimension
                        auto& dim = old_subset[0];
                        auto& alias_dim = viewed_subset.back();
                        auto new_dim = symbolic::add(alias_dim, dim);
                        new_subset.push_back(new_dim);

                        // Add trailing dimensions
                        for (size_t i = 1; i < old_subset.size(); i++) {
                            auto& old_dim = old_subset[i];
                            new_subset.push_back(old_dim);
                        }

                        iedge.subset() = new_subset;
                    }

                    applied = true;
                    reduced.insert(viewed_container);
                }
            }
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

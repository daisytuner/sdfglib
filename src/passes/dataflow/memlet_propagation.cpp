#include "sdfg/passes/dataflow/memlet_propagation.h"

namespace sdfg {
namespace passes {

ForwardMemletPropagation::ForwardMemletPropagation()
    : Pass() {

      };

std::string ForwardMemletPropagation::name() { return "ForwardMemletPropagation"; };

bool ForwardMemletPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                        analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<std::string> visited;
    for (auto& container : sdfg.containers()) {
        if (visited.find(container) != visited.end()) {
            continue;
        }
        if (!sdfg.is_transient(container)) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }

        // Criterion: Scalar written once
        if (!users.views(container).empty() || !users.moves(container).empty()) {
            continue;
        }
        if (users.writes(container).size() != 1) {
            continue;
        }

        // Criterion: Written to by access node
        auto write = users.writes(container).at(0);
        if (!dynamic_cast<data_flow::AccessNode*>(write->element())) {
            continue;
        }
        auto& access_node = static_cast<data_flow::AccessNode&>(*write->element());
        auto& graph = *write->parent();
        auto& block = static_cast<structured_control_flow::Block&>(*graph.get_parent());

        // Criterion: Access node is connected to an assignment tasklet
        const data_flow::Tasklet* tasklet = nullptr;
        if (graph.in_degree(access_node) == 1) {
            auto& edge = *graph.in_edges(access_node).begin();
            auto src = dynamic_cast<const data_flow::Tasklet*>(&edge.src());
            if (graph.in_degree(*src) == 1 && !src->is_conditional()) {
                if (src->code() == data_flow::TaskletCode::assign) {
                    tasklet = src;
                }
            }
        }
        if (!tasklet) {
            continue;
        }

        // Retrieve assigning data
        auto& edge = *graph.in_edges(*tasklet).begin();
        auto& src = static_cast<const data_flow::AccessNode&>(edge.src());
        std::string assigning_data = src.data();
        if (symbolic::is_nv(symbolic::symbol(assigning_data))) {
            continue;
        }
        data_flow::Subset assigning_subset = edge.subset();
        const data_flow::AccessNode* assigning_node = &src;

        // Criterion: Not casting types
        auto& assigning_type = types::infer_type(sdfg, sdfg.type(assigning_data), assigning_subset);
        if (assigning_type.primitive_type() !=
            tasklet->input_type(edge.dst_conn()).primitive_type()) {
            continue;
        }
        if (type.primitive_type() != tasklet->outputs().begin()->second.primitive_type()) {
            continue;
        }

        // Risky symbols
        std::unordered_set<std::string> risky_symbols = {assigning_data};
        for (auto& dim : assigning_subset) {
            for (auto& atom : symbolic::atoms(dim)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom)->get_name();
                risky_symbols.insert(sym);
            }
        }

        // Propagate assigning data into container uses
        auto reads = users.reads(container);
        size_t propagated = reads.size();
        for (auto read : reads) {
            auto read_node = dynamic_cast<data_flow::AccessNode*>(read->element());
            if (!read_node) {
                continue;
            }
            auto read_graph = read->parent();
            if (read_node == &access_node) {
                propagated--;
                continue;
            }
            // Criterion: Data races
            if (!users.dominates(*write, *read)) {
                continue;
            }
            auto& uses_between = users.all_uses_between(*write, *read);
            bool assigning_data_is_written = false;
            for (auto& user : uses_between) {
                // Unsafe pointer operations
                if (user->use() == analysis::Use::MOVE) {
                    // Viewed container is not constant
                    if (user->container() == assigning_data) {
                        assigning_data_is_written = true;
                        break;
                    }

                    // Unsafe pointer operations
                    auto& move_node = dynamic_cast<data_flow::AccessNode&>(*user->element());
                    auto& move_graph = *user->parent();
                    auto& move_edge = *move_graph.in_edges(move_node).begin();
                    auto& view_node = dynamic_cast<data_flow::AccessNode&>(move_edge.src());
                    if (move_edge.dst_conn() == "void" ||
                        symbolic::is_pointer(symbolic::symbol(view_node.data()))) {
                        assigning_data_is_written = true;
                        break;
                    }
                } else if (user->use() == analysis::Use::WRITE) {
                    if (risky_symbols.find(user->container()) != risky_symbols.end()) {
                        assigning_data_is_written = true;
                        break;
                    }
                }
            }
            if (assigning_data_is_written) {
                continue;
            }

            // Casting
            bool unsafe_cast = false;
            for (auto& oedge : read_graph->out_edges(*read_node)) {
                auto& dst = dynamic_cast<const data_flow::Tasklet&>(oedge.dst());
                if (dst.input_type(oedge.dst_conn()).primitive_type() != type.primitive_type()) {
                    unsafe_cast = true;
                    break;
                }
            }
            if (unsafe_cast) {
                continue;
            }

            // Propagate
            read_node->data() = assigning_data;
            for (auto& oedge : read_graph->out_edges(*read_node)) {
                oedge.subset() = assigning_subset;
            }

            propagated--;
            applied = true;
        }

        if (propagated != reads.size()) {
            visited.insert(container);
            visited.insert(assigning_data);
        }

        // Remove tasklet and access node if all reads were propagated
        if (propagated == 0 && false) {
            auto& tasklet_in_edge = *graph.in_edges(*tasklet).begin();
            auto& tasklet_out_edge = *graph.out_edges(*tasklet).begin();
            if (graph.out_degree(access_node) > 0) {
                access_node.data() = assigning_data;
                for (auto& oedge : graph.out_edges(access_node)) {
                    oedge.subset() = assigning_subset;
                }
                std::vector<data_flow::Memlet*> assigning_node_in_edges;
                for (auto& iedge : graph.in_edges(*assigning_node)) {
                    assigning_node_in_edges.push_back(&iedge);
                }
                for (auto& iedge : assigning_node_in_edges) {
                    builder.add_memlet(block, iedge->src(), iedge->src_conn(), access_node,
                                       iedge->dst_conn(), iedge->subset());
                }
                for (auto& iedge : assigning_node_in_edges) {
                    builder.remove_memlet(block, *iedge);
                }
            }
            builder.remove_memlet(block, tasklet_in_edge);
            builder.remove_memlet(block, tasklet_out_edge);
            builder.remove_node(block, *tasklet);
            builder.remove_node(block, *assigning_node);
            if (graph.out_degree(access_node) == 0) {
                builder.remove_node(block, access_node);
            }
            applied = true;
        }
    }

    return applied;
};

BackwardMemletPropagation::BackwardMemletPropagation()
    : Pass() {

      };

std::string BackwardMemletPropagation::name() { return "BackwardMemletPropagation"; };

bool BackwardMemletPropagation::run_pass(builder::StructuredSDFGBuilder& builder,
                                         analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<std::string> visited;
    for (auto& container : sdfg.containers()) {
        if (visited.find(container) != visited.end()) {
            continue;
        }
        if (!sdfg.is_transient(container)) {
            continue;
        }
        auto& type = sdfg.type(container);
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }

        // Criterion: Scalar written once
        if (!users.views(container).empty() || !users.moves(container).empty()) {
            continue;
        }
        if (users.writes(container).size() != 1 || users.reads(container).size() != 1) {
            continue;
        }

        // Criterion: Read by access node
        auto read = users.reads(container).at(0);
        if (!dynamic_cast<data_flow::AccessNode*>(read->element())) {
            continue;
        }
        auto write = users.writes(container).at(0);
        if (!dynamic_cast<data_flow::AccessNode*>(write->element())) {
            continue;
        }

        auto& read_access_node = static_cast<data_flow::AccessNode&>(*read->element());
        auto& read_graph = *read->parent();
        auto& read_block = static_cast<structured_control_flow::Block&>(*read_graph.get_parent());

        // Criterion: Access node is connected to an assignment tasklet
        const data_flow::Tasklet* tasklet = nullptr;
        if (read_graph.out_degree(read_access_node) == 1) {
            auto& edge = *read_graph.out_edges(read_access_node).begin();
            auto dst = dynamic_cast<const data_flow::Tasklet*>(&edge.dst());
            if (read_graph.in_degree(*dst) == 1 && !dst->is_conditional()) {
                if (dst->code() == data_flow::TaskletCode::assign) {
                    tasklet = dst;
                }
            }
        }
        if (!tasklet) {
            continue;
        }

        // Retrieve assigning data
        auto& edge = *read_graph.out_edges(*tasklet).begin();
        auto& dst = static_cast<const data_flow::AccessNode&>(edge.dst());
        std::string assigning_data = dst.data();
        data_flow::Subset assigning_subset = edge.subset();
        const data_flow::AccessNode* assigning_node = &dst;
        if (read_graph.out_degree(*assigning_node) != 0) {
            continue;
        }

        // Criterion: Not casting types
        auto& assigning_type = types::infer_type(sdfg, sdfg.type(assigning_data), assigning_subset);
        if (assigning_type.primitive_type() !=
            tasklet->outputs().begin()->second.primitive_type()) {
            continue;
        }
        if (type.primitive_type() != tasklet->inputs().begin()->second.primitive_type()) {
            continue;
        }

        // Risky symbols
        std::unordered_set<std::string> risky_symbols = {assigning_data};
        for (auto& dim : assigning_subset) {
            for (auto& atom : symbolic::atoms(dim)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom)->get_name();
                risky_symbols.insert(sym);
            }
        }

        // Propagate assigning data into container write

        // Criterion: Written to by access node
        auto& write_access_node = static_cast<data_flow::AccessNode&>(*write->element());
        auto& write_graph = *write->parent();

        // Criterion: Data races
        if (!users.dominates(*write, *read) || !users.post_dominates(*read, *write)) {
            continue;
        }
        auto& uses_between = users.all_uses_between(*write, *read);
        bool race_condition = false;
        for (auto& user : uses_between) {
            // Unsafe pointer operations
            if (user->use() == analysis::Use::MOVE) {
                // Viewed container is not constant
                if (user->container() == assigning_data) {
                    race_condition = true;
                    break;
                }

                // Unsafe pointer operations
                auto& move_node = dynamic_cast<data_flow::AccessNode&>(*user->element());
                auto& move_graph = *user->parent();
                auto& move_edge = *move_graph.in_edges(move_node).begin();
                auto& view_node = dynamic_cast<data_flow::AccessNode&>(move_edge.src());
                if (move_edge.dst_conn() == "void" ||
                    symbolic::is_pointer(symbolic::symbol(view_node.data()))) {
                    race_condition = true;
                    break;
                }
            } else if (user->use() == analysis::Use::WRITE) {
                if (risky_symbols.find(user->container()) != risky_symbols.end()) {
                    race_condition = true;
                    break;
                }
            } else if (user->use() == analysis::Use::READ) {
                if (user->container() == assigning_data) {
                    race_condition = true;
                    break;
                }
            }
        }
        if (race_condition) {
            continue;
        }

        // Propagate
        write_access_node.data() = assigning_data;
        for (auto& iedge : write_graph.in_edges(write_access_node)) {
            iedge.subset() = assigning_subset;
        }
        for (auto& oedge : write_graph.out_edges(write_access_node)) {
            oedge.subset() = assigning_subset;
        }
        applied = true;

        visited.insert(container);
        visited.insert(assigning_data);

        // Remove tasklet and access node if all reads were propagated
        auto& tasklet_in_edge = *read_graph.in_edges(*tasklet).begin();
        auto& tasklet_out_edge = *read_graph.out_edges(*tasklet).begin();
        builder.remove_memlet(read_block, tasklet_in_edge);
        builder.remove_memlet(read_block, tasklet_out_edge);
        builder.remove_node(read_block, *tasklet);
        builder.remove_node(read_block, *assigning_node);
        if (read_graph.in_degree(read_access_node) == 0) {
            builder.remove_node(read_block, read_access_node);
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

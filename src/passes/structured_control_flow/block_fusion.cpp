#include "sdfg/passes/structured_control_flow/block_fusion.h"

namespace sdfg {
namespace passes {

BlockFusion::BlockFusion(builder::StructuredSDFGBuilder& builder,
                         analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool BlockFusion::can_be_applied(data_flow::DataFlowGraph& first_graph,
                                 symbolic::Assignments& first_assignments,
                                 data_flow::DataFlowGraph& second_graph,
                                 symbolic::Assignments& second_assignments) {
    // Criterion: no pointer ref fusions
    for (auto& edge : first_graph.edges()) {
        if (edge.src_conn() == "refs" || edge.dst_conn() == "refs") {
            return false;
        }
    }
    for (auto& edge : second_graph.edges()) {
        if (edge.src_conn() == "refs" || edge.dst_conn() == "refs") {
            return false;
        }
    }

    // Criterion: No conditional tasklets
    for (auto& tasklet : first_graph.tasklets()) {
        if (tasklet->is_conditional()) {
            return false;
        }
    }
    for (auto& tasklet : second_graph.tasklets()) {
        if (tasklet->is_conditional()) {
            return false;
        }
    }

    // Criterion: No data races cause by transition
    if (!first_assignments.empty()) {
        return false;
    }

    // Numerical stability: Unique order of nodes
    auto pdoms = first_graph.post_dominators();
    bool has_connector = false;
    for (auto& node : second_graph.sources()) {
        if (!dynamic_cast<const data_flow::AccessNode*>(node)) {
            return false;
        }
        auto access_node = static_cast<const data_flow::AccessNode*>(node);
        auto data = access_node->data();

        // Connects to first block
        if (pdoms.find(data) == pdoms.end()) {
            continue;
        }
        has_connector = true;
        // Is unique successor in first block
        if (first_graph.out_degree(*pdoms.at(data)) > 0) {
            return false;
        }
    }
    if (!has_connector) {
        return false;
    }

    return true;
};

void BlockFusion::apply(structured_control_flow::Block& first_block,
                        symbolic::Assignments& first_assignments,
                        structured_control_flow::Block& second_block,
                        symbolic::Assignments& second_assignments) {
    data_flow::DataFlowGraph& first_graph = first_block.dataflow();
    data_flow::DataFlowGraph& second_graph = second_block.dataflow();

    // Update symbols
    for (auto& entry : second_assignments) {
        first_assignments[entry.first] = entry.second;
    }

    // Collect nodes to connect to,
    // i.e., last access node for each container
    auto pdoms = first_graph.post_dominators();

    // Collect nodes which need to be connected,
    // i.e., sources of the second graph
    std::unordered_map<data_flow::DataFlowNode*, std::unordered_set<data_flow::DataFlowNode*>>
        connectors;
    for (auto& node : second_graph.sources()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (pdoms.find(access_node->data()) != pdoms.end()) {
                connectors.insert({node, {pdoms[access_node->data()]}});
            }
        }
    }

    // Copy nodes from second to first
    std::unordered_map<data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
    for (auto& node : second_graph.nodes()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (connectors.find(access_node) != connectors.end()) {
                if (connectors[access_node].size() != 1) {
                    throw InvalidSDFGException("BlockFusion: Expected exactly one connector");
                }
                // Connect by replacement
                node_mapping[access_node] = *connectors[access_node].begin();
            } else {
                // Add new
                node_mapping[access_node] = &builder_.add_access(first_block, access_node->data());
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            node_mapping[tasklet] = &builder_.add_tasklet(first_block, tasklet->code(),
                                                          tasklet->output(), tasklet->inputs());
        }
    }

    // Connect new nodes according to edges of second graph
    for (auto& edge : second_graph.edges()) {
        auto& src_node = edge.src();
        auto& dst_node = edge.dst();

        builder_.add_memlet(first_block, *node_mapping[&src_node], edge.src_conn(),
                            *node_mapping[&dst_node], edge.dst_conn(), edge.subset());
    }
};

bool BlockFusion::accept(structured_control_flow::Sequence& parent,
                         structured_control_flow::Sequence& node) {
    bool applied = false;

    if (node.size() == 0) {
        return applied;
    }

    // Traverse node to find pairs of blocks
    size_t i = 0;
    while (i < (node.size() - 1)) {
        auto current_entry = node.at(i);
        if (dynamic_cast<structured_control_flow::Block*>(&current_entry.first) == nullptr) {
            i++;
            continue;
        }
        auto current_block = dynamic_cast<structured_control_flow::Block*>(&current_entry.first);

        auto next_entry = node.at(i + 1);
        if (dynamic_cast<structured_control_flow::Block*>(&next_entry.first) == nullptr) {
            i++;
            continue;
        }
        auto next_block = dynamic_cast<structured_control_flow::Block*>(&next_entry.first);

        if (this->can_be_applied(current_block->dataflow(), current_entry.second.assignments(),
                                 next_block->dataflow(), next_entry.second.assignments())) {
            this->apply(*current_block, current_entry.second.assignments(), *next_block,
                        next_entry.second.assignments());
            builder_.remove_child(node, i + 1);
            applied = true;
        } else {
            i++;
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

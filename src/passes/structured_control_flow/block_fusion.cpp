#include "sdfg/passes/structured_control_flow/block_fusion.h"

namespace sdfg {
namespace passes {

BlockFusion::BlockFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool BlockFusion::can_be_applied(
    data_flow::DataFlowGraph& first_graph,
    control_flow::Assignments& first_assignments,
    data_flow::DataFlowGraph& second_graph,
    control_flow::Assignments& second_assignments
) {
    // Criterion: No side-effect nodes
    for (auto& node : first_graph.nodes()) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            if (lib_node->side_effect()) {
                return false;
            }
        } else if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            if (tasklet->is_conditional()) {
                return false;
            }
        }
    }
    for (auto& node : second_graph.nodes()) {
        if (auto lib_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
            if (lib_node->side_effect()) {
                return false;
            }
        } else if (auto tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            if (tasklet->is_conditional()) {
                return false;
            }
        }
    }
    // Criterion: No data races cause by transition
    if (!first_assignments.empty()) {
        return false;
    }

    // Numerical stability: Unique order of nodes
    auto pdoms = first_graph.post_dominators();
    std::unordered_map<std::string, const data_flow::AccessNode*> connectors;
    for (auto& node : second_graph.topological_sort()) {
        if (!dynamic_cast<const data_flow::AccessNode*>(node)) {
            continue;
        }
        auto access_node = static_cast<const data_flow::AccessNode*>(node);

        // Already connected
        if (connectors.find(access_node->data()) != connectors.end()) {
            continue;
        }
        // Write-after-write
        if (second_graph.in_degree(*access_node) > 0) {
            return false;
        }

        if (pdoms.find(access_node->data()) == pdoms.end()) {
            continue;
        }
        connectors[access_node->data()] = pdoms.at(access_node->data());
    }

    return true;
};

void BlockFusion::apply(
    structured_control_flow::Block& first_block,
    control_flow::Assignments& first_assignments,
    structured_control_flow::Block& second_block,
    control_flow::Assignments& second_assignments
) {
    data_flow::DataFlowGraph& first_graph = first_block.dataflow();
    data_flow::DataFlowGraph& second_graph = second_block.dataflow();

    // Update symbols
    for (auto& entry : second_assignments) {
        first_assignments[entry.first] = entry.second;
    }

    // Collect nodes to connect to
    auto pdoms = first_graph.post_dominators();
    std::unordered_set<std::string> already_connected;
    std::unordered_map<data_flow::AccessNode*, data_flow::AccessNode*> connectors;
    for (auto& node : second_graph.topological_sort()) {
        if (!dynamic_cast<data_flow::AccessNode*>(node)) {
            continue;
        }
        auto access_node = static_cast<data_flow::AccessNode*>(node);

        // Already connected
        if (already_connected.find(access_node->data()) != already_connected.end()) {
            continue;
        }
        // Write-after-write
        if (second_graph.in_degree(*access_node) > 0) {
            throw InvalidSDFGException("BlockFusion: Write-after-write");
        }

        if (pdoms.find(access_node->data()) == pdoms.end()) {
            continue;
        }
        connectors[access_node] = pdoms.at(access_node->data());
        already_connected.insert(access_node->data());
    }

    // Copy nodes from second to first
    std::unordered_map<data_flow::DataFlowNode*, data_flow::DataFlowNode*> node_mapping;
    for (auto& node : second_graph.nodes()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (connectors.find(access_node) != connectors.end()) {
                // Connect by replacement
                node_mapping[access_node] = connectors[access_node];
            } else {
                // Add new
                node_mapping[access_node] = &builder_.add_access(first_block, access_node->data());
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            node_mapping[tasklet] =
                &builder_.add_tasklet(first_block, tasklet->code(), tasklet->output(), tasklet->inputs());
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            node_mapping[library_node] = &builder_.copy_library_node(first_block, *library_node);
        } else {
            throw InvalidSDFGException("BlockFusion: Unknown node type");
        }
    }

    // Connect new nodes according to edges of second graph
    for (auto& edge : second_graph.edges()) {
        auto& src_node = edge.src();
        auto& dst_node = edge.dst();

        builder_.add_memlet(
            first_block,
            *node_mapping[&src_node],
            edge.src_conn(),
            *node_mapping[&dst_node],
            edge.dst_conn(),
            edge.subset()
        );
    }
};

bool BlockFusion::accept(structured_control_flow::Sequence& parent, structured_control_flow::Sequence& node) {
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

        if (this->can_be_applied(
                current_block->dataflow(),
                current_entry.second.assignments(),
                next_block->dataflow(),
                next_entry.second.assignments()
            )) {
            this->apply(*current_block, current_entry.second.assignments(), *next_block, next_entry.second.assignments());
            builder_.remove_child(node, i + 1);
            applied = true;
        } else {
            i++;
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg

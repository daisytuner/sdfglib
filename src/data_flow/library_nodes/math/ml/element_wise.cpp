#include "sdfg/data_flow/library_nodes/math/ml/relu.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ElementWiseUnaryNode::ElementWiseUnaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::unordered_map<std::string, std::string>& attributes
)
    : MathNode(element_id, debug_info, vertex, parent, code, {"Y"}, {"X"}, data_flow::ImplementationType_NONE),
      attributes_(attributes) {}

void ElementWiseUnaryNode::validate(const Function& function) const {
    // TODO: Implement
}

bool ElementWiseUnaryNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    auto& begin_subsets_out = oedge.begin_subset();
    auto& end_subsets_out = oedge.end_subset();
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    for (size_t i = 0; i < begin_subsets_out.size(); i++) {
        auto& dim_begin = begin_subsets_out[i];
        auto& dim_end = end_subsets_out[i];

        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, symbolic::add(dim_end, symbolic::one()));
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        new_subset.push_back(indvar);
    }

    bool success = this->expand_operation(
        builder,
        analysis_manager,
        *last_scope,
        input_node.data(),
        output_node.data(),
        iedge.base_type(),
        oedge.base_type(),
        new_subset
    );
    if (!success) {
        return false;
    }

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

ElementWiseBinaryNode::ElementWiseBinaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::unordered_map<std::string, std::string>& attributes
)
    : MathNode(element_id, debug_info, vertex, parent, code, {"C"}, {"A", "B"}, data_flow::ImplementationType_NONE),
      attributes_(attributes) {}

void ElementWiseBinaryNode::validate(const Function& function) const {
    // TODO: Implement
}

bool ElementWiseBinaryNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input_a = this->inputs_.at(0);
    auto& input_b = this->inputs_.at(1);
    auto& output = this->outputs_.at(0);

    auto iedge_a = &(*dataflow.in_edges(*this).begin());
    auto iedge_b = &(*(++dataflow.in_edges(*this).begin()));
    if (iedge_a->dst_conn() != "A") {
        std::swap(iedge_a, iedge_b);
    }
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node_a = static_cast<data_flow::AccessNode&>(iedge_a->src());
    auto& input_node_b = static_cast<data_flow::AccessNode&>(iedge_b->src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node_a) != 0 || dataflow.in_degree(input_node_b) != 0 ||
        dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    auto& begin_subsets_out = oedge.begin_subset();
    auto& end_subsets_out = oedge.end_subset();
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    for (size_t i = 0; i < begin_subsets_out.size(); i++) {
        auto& dim_begin = begin_subsets_out[i];
        auto& dim_end = end_subsets_out[i];

        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, symbolic::add(dim_end, symbolic::one()));
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        new_subset.push_back(indvar);
    }

    bool success = this->expand_operation(
        builder,
        analysis_manager,
        *last_scope,
        input_node_a.data(),
        input_node_b.data(),
        output_node.data(),
        iedge_a->base_type(),
        iedge_b->base_type(),
        oedge.base_type(),
        new_subset
    );
    if (!success) {
        return false;
    }

    // Clean up block
    builder.remove_memlet(block, *iedge_a);
    builder.remove_memlet(block, *iedge_b);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node_a);
    builder.remove_node(block, input_node_b);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

} // namespace ml
} // namespace math
} // namespace sdfg

#include "sdfg/data_flow/library_nodes/math/ml/dropout.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

DropoutNode::DropoutNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Dropout,
          {"output", "mask"},
          {"data"},
          data_flow::ImplementationType_NONE
      ) {}

void DropoutNode::validate(const Function& function) const {
    // TODO: Implement
}

bool DropoutNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 2) {
        return false;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));

    auto& input = this->inputs_.at(0);
    auto& output_data = this->outputs_.at(0);
    auto& output_mask = this->outputs_.at(1);

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto oedge_output = &(*dataflow.out_edges(*this).begin());
    auto oedge_mask = &(*(++dataflow.out_edges(*this).begin()));
    if (oedge_output->src_conn() != "output") {
        std::swap(oedge_output, oedge_mask);
    }

    // Checks if legal
    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node_output = static_cast<data_flow::AccessNode&>(oedge_output->dst());
    auto& output_node_mask = static_cast<data_flow::AccessNode&>(oedge_mask->dst());
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node_output) != 0 ||
        dataflow.out_degree(output_node_mask) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;

    // Add maps
    auto& begin_subsets_out = oedge_output->begin_subset();
    auto& end_subsets_out = oedge_output->end_subset();
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
            structured_control_flow::ScheduleType_Sequential,
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        new_subset.push_back(indvar);
    }

    // output = data, mask = 1
    {
        auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());
        auto& input_node_new = builder.add_access(code_block, input_node.data(), input_node.debug_info());
        auto& output_node_output_new =
            builder.add_access(code_block, output_node_output.data(), output_node_output.debug_info());
        auto& output_node_mask_new =
            builder.add_access(code_block, output_node_mask.data(), output_node_mask.debug_info());

        auto& tasklet_output = builder.add_tasklet(code_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
        builder.add_computational_memlet(
            code_block, input_node_new, tasklet_output, "_in", new_subset, iedge.base_type(), block.debug_info()
        );
        builder.add_computational_memlet(
            code_block,
            tasklet_output,
            "_out",
            output_node_output_new,
            new_subset,
            oedge_output->base_type(),
            block.debug_info()
        );

        auto& tasklet_mask = builder.add_tasklet(code_block, data_flow::assign, "_out", {"1"}, block.debug_info());
        builder.add_computational_memlet(
            code_block,
            tasklet_mask,
            "_out",
            output_node_mask_new,
            new_subset,
            oedge_mask->base_type(),
            block.debug_info()
        );
    }

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, *oedge_output);
    builder.remove_memlet(block, *oedge_mask);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node_output);
    builder.remove_node(block, output_node_mask);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> DropoutNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new DropoutNode(element_id, this->debug_info(), vertex, parent));
}

} // namespace ml
} // namespace math
} // namespace sdfg

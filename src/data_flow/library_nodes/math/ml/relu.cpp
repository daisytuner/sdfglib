#include "sdfg/data_flow/library_nodes/math/ml/relu.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

ReLUNode::ReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& output,
    const std::string& input
)
    : MathNode(element_id, debug_info, vertex, parent, LibraryNodeType_ReLU, {output}, {input}) {}

bool ReLUNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }
    auto& scope_analyisis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analyisis.parent_scope(&block));

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    auto& type = sdfg.type(input);
    types::Scalar scalar_type(type.primitive_type());

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node = iedge.src();
    auto& output_node = oedge.dst();
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info()).first;

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
            structured_control_flow::ScheduleType_Sequential,
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        new_subset.push_back(indvar);
    }

    // Add code
    auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());
    auto& input_node_new = builder.add_access(code_block, input, input_node.debug_info());
    auto& output_node_new = builder.add_access(code_block, output, output_node.debug_info());
    auto& tasklet = builder.add_tasklet(
        code_block,
        data_flow::TaskletCode::max,
        {"_out", scalar_type},
        {{"0", scalar_type}, {"_in", scalar_type}},
        block.debug_info()
    );
    builder.add_memlet(code_block, input_node_new, "void", tasklet, "_in", new_subset, block.debug_info());
    builder.add_memlet(code_block, tasklet, "_out", output_node_new, "void", new_subset, block.debug_info());

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, block);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new ReLUNode(element_id, this->debug_info(), vertex, parent, this->outputs_.at(0), this->inputs_.at(0))
    );
}

nlohmann::json ReLUNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ReLUNode& relu_node = static_cast<const ReLUNode&>(library_node);
    nlohmann::json j;

    j["code"] = relu_node.code().value();
    j["outputs"] = relu_node.outputs();
    j["inputs"] = relu_node.inputs();

    return j;
}

data_flow::LibraryNode& ReLUNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_ReLU.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto outputs = j.at("outputs").get<std::vector<std::string>>();
    auto inputs = j.at("inputs").get<std::vector<std::string>>();

    return builder.add_library_node<ReLUNode>(parent, debug_info, outputs.at(0), inputs.at(0));
}

ReLUNodeDispatcher::ReLUNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ReLUNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void ReLUNodeDispatcher::dispatch(codegen::PrettyPrinter& stream) {
    throw std::runtime_error("ReLUNode not implemented");
}


} // namespace ml
} // namespace math
} // namespace sdfg

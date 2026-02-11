#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/cast_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

CastNode::CastNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Cast,
          {"_out"},
          {"_in1"},
          data_flow::ImplementationType_NONE
      ) {}

bool CastNode::supports_integer_types() const { return true; }

symbolic::SymbolSet CastNode::symbols() const { return {}; }

void CastNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

bool CastNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    auto& shape = static_cast<const types::Tensor&>(oedge.base_type()).shape();
    for (size_t i = 0; i < shape.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, shape.at(i));
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

        loop_vars.push_back(indvar);
    }

    auto& code_block = builder.add_block(*last_scope);
    auto& input_node_new = builder.add_access(code_block, input_node.data());
    auto& output_node_new = builder.add_access(code_block, output_node.data());

    auto& tasklet =
        builder.add_tasklet(code_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, code_block.debug_info());

    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", loop_vars, iedge.base_type());
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, loop_vars, oedge.base_type());

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

void CastNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("CastNode must have exactly one input memlet");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("CastNode must have exactly one output memlet");
    }

    auto& iedge = *graph.in_edges(*this).begin();
    auto& oedge = *graph.out_edges(*this).begin();
    if (iedge.base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException("CastNode input must be a tensor");
    }
    if (oedge.base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException("CastNode output must be a tensor");
    }
}

std::unique_ptr<data_flow::DataFlowNode> CastNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new CastNode(element_id, this->debug_info(), vertex, parent));
}

nlohmann::json CastNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CastNode& cast_node = static_cast<const CastNode&>(library_node);
    nlohmann::json j;

    j["code"] = cast_node.code().value();

    return j;
}

data_flow::LibraryNode& CastNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return static_cast<CastNode&>(builder.add_library_node<CastNode>(parent, debug_info));
}

} // namespace tensor
} // namespace math
} // namespace sdfg

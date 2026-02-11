#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/elementwise_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise/elementwise_node.h"

namespace sdfg {
namespace math {
namespace tensor {

ElementwiseNode::ElementwiseNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::TaskletCode& code
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Elementwise, {}, {}, data_flow::ImplementationType_NONE
      ),
      tasklet_code_(code) {
    for (size_t i = 1; i <= data_flow::arity(code); i++) {
        this->inputs_.push_back("_in" + std::to_string(i));
    }
    this->outputs_.push_back("_out");
}

ElementwiseNode::ElementwiseNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const math::cmath::CMathFunction& function,
    types::PrimitiveType precision
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Elementwise, {}, {}, data_flow::ImplementationType_NONE
      ),
      cmath_function_(function), precision_(precision) {
    for (size_t i = 1; i <= math::cmath::cmath_function_to_arity(function); i++) {
        this->inputs_.push_back("_in" + std::to_string(i));
    }
    this->outputs_.push_back("_out");
}

symbolic::SymbolSet ElementwiseNode::symbols() const { return {}; }

void ElementwiseNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

bool ElementwiseNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    std::cout << "Expanding ElementwiseNode with ID " << this->element_id() << std::endl;
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& output = this->outputs_.at(0);
    auto& oedge = *dataflow.out_edges(*this).begin();
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.out_degree(output_node) != 0) {
        return false;
    }
    std::cout << "Output node: " << output_node.data() << std::endl;
    std::unordered_map<std::string, const data_flow::Memlet*> input_memlets;
    for (auto& iedge : dataflow.in_edges(*this)) {
        auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
        if (dataflow.in_degree(input_node) != 0) {
            return false;
        }
        input_memlets.insert({iedge.dst_conn(), &iedge});
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    auto& shape = dynamic_cast<const types::Tensor&>(oedge.base_type()).shape();
    for (size_t i = 0; i < shape.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

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

    std::cout << "Created maps" << std::endl;
    if (this->cmath_function_.has_value()) {
        auto& code_block = builder.add_block(*last_scope);

        auto& cmath_node = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), this->cmath_function_.value(), this->precision_.value()
        );

        auto& output_node_new = builder.add_access(code_block, output_node.data());
        builder.add_computational_memlet(code_block, cmath_node, "_out", output_node_new, loop_vars, oedge.base_type());

        for (auto& entry : input_memlets) {
            if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&entry.second->src())) {
                auto& const_node_new = builder.add_constant(code_block, const_node->data(), const_node->type());
                builder
                    .add_computational_memlet(code_block, const_node_new, cmath_node, entry.first, {}, const_node->type());
                continue;
            }
            auto& src = dynamic_cast<const data_flow::AccessNode&>(entry.second->src());
            auto& input_node_new = builder.add_access(code_block, src.data());
            builder.add_computational_memlet(
                code_block, input_node_new, cmath_node, entry.first, loop_vars, entry.second->base_type()
            );
        }
    } else {
        auto& code_block = builder.add_block(*last_scope);

        auto& tasklet = builder.add_tasklet(
            code_block, this->tasklet_code_.value(), this->outputs_.at(0), this->inputs_, this->debug_info()
        );

        auto& output_node_new = builder.add_access(code_block, output_node.data());
        builder.add_computational_memlet(
            code_block, tasklet, this->outputs_.at(0), output_node_new, loop_vars, oedge.base_type()
        );

        for (auto& entry : input_memlets) {
            auto& src = static_cast<const data_flow::AccessNode&>(entry.second->src());
            auto& input_node_new = builder.add_access(code_block, src.data());
            builder.add_computational_memlet(
                code_block, input_node_new, tasklet, entry.first, loop_vars, entry.second->base_type()
            );
        }
    }

    // Clean up block
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, output_node);

    std::unordered_set<const data_flow::DataFlowNode*> nodes_to_remove;
    for (auto& entry : input_memlets) {
        builder.remove_memlet(block, *entry.second);
        if (nodes_to_remove.find(&entry.second->src()) == nodes_to_remove.end()) {
            nodes_to_remove.insert(&entry.second->src());
        }
    }
    for (auto* node : nodes_to_remove) {
        builder.remove_node(block, *node);
    }

    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ElementwiseNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    if (this->cmath_function_) {
        return std::unique_ptr<data_flow::DataFlowNode>(new ElementwiseNode(
            element_id, this->debug_info(), vertex, parent, this->cmath_function_.value(), this->precision_.value()
        ));
    } else if (this->tasklet_code_) {
        return std::unique_ptr<data_flow::DataFlowNode>(
            new ElementwiseNode(element_id, this->debug_info(), vertex, parent, this->tasklet_code_.value())
        );
    } else {
        throw std::runtime_error("ElementwiseNode must have either a tasklet code or a CMath function.");
    }
}

nlohmann::json ElementwiseNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ElementwiseNode& elem_node = static_cast<const ElementwiseNode&>(library_node);
    nlohmann::json j;

    j["code"] = elem_node.code().value();
    if (elem_node.cmath_function()) {
        j["cmath_function"] = elem_node.cmath_function().value();
        j["precision"] = elem_node.precision().value();
    } else if (elem_node.tasklet_code()) {
        j["tasklet_code"] = elem_node.tasklet_code().value();
    } else {
        throw std::runtime_error("ElementwiseNode must have either a tasklet code or a CMath function.");
    }

    return j;
}

data_flow::LibraryNode& ElementwiseNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto code = j["code"].get<std::string>();
    if (j.contains("cmath_function") && j.contains("precision")) {
        auto stem = j["function_stem"].get<std::string>();
        auto function = math::cmath::string_to_cmath_function(stem);
        auto prim_type = static_cast<types::PrimitiveType>(j["primitive_type"].get<int>());
        return static_cast<
            ElementwiseNode&>(builder.add_library_node<ElementwiseNode>(parent, debug_info, function, prim_type));
    } else if (j.contains("tasklet_code")) {
        return static_cast<ElementwiseNode&>(builder.add_library_node<ElementwiseNode>(parent, debug_info, j.at("code"))
        );
    } else {
        throw std::runtime_error(
            "JSON for ElementwiseNode must contain either 'cmath_function' and 'precision' or 'tasklet_code'."
        );
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg

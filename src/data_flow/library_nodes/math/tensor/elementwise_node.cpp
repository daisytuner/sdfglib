#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

ElementWiseUnaryNode::ElementWiseUnaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape
)
    : MathNode(element_id, debug_info, vertex, parent, code, {"Y"}, {"X"}, data_flow::ImplementationType_NONE),
      shape_(shape) {}

symbolic::SymbolSet ElementWiseUnaryNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ElementWiseUnaryNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ElementWiseUnaryNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 1 || graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("ElementWiseUnaryNode: Node must have exactly one input and one output");
    }

    auto& iedge = *graph.in_edges(*this).begin();
    if (iedge.base_type().type_id() != types::TypeID::Scalar && iedge.base_type().type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException(
            "ElementWiseUnaryNode: Input memlet must be of scalar or pointer type. Found type: " +
            iedge.base_type().print()
        );
    }
    if (iedge.base_type().type_id() == types::TypeID::Pointer) {
        auto& ptr_type = static_cast<const types::Pointer&>(iedge.base_type());
        if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ElementWiseUnaryNode: Input memlet pointer be flat. Found type: " + ptr_type.pointee_type().print()
            );
        }
        if (!iedge.subset().empty()) {
            throw InvalidSDFGException("ElementWiseUnaryNode: Input memlet pointer must not be dereferenced.");
        }
    }

    auto& oedge = *graph.out_edges(*this).begin();
    if (oedge.base_type().type_id() != types::TypeID::Scalar && oedge.base_type().type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException(
            "ElementWiseUnaryNode: Output memlet must be of scalar or pointer type. Found type: " +
            oedge.base_type().print()
        );
    }
    if (oedge.base_type().type_id() == types::TypeID::Pointer) {
        auto& ptr_type = static_cast<const types::Pointer&>(oedge.base_type());
        if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ElementWiseUnaryNode: Output memlet pointer be flat. Found type: " + ptr_type.pointee_type().print()
            );
        }
        if (!oedge.subset().empty()) {
            throw InvalidSDFGException("ElementWiseUnaryNode: Output memlet pointer must not be dereferenced.");
        }
    }
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
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    for (size_t i = 0; i < this->shape_.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, this->shape_[i]);
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

    // Linearize subset
    symbolic::Expression linear_index = symbolic::zero();
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        linear_index = symbolic::add(symbolic::mul(linear_index, this->shape_[i]), loop_vars[i]);
    }
    if (!this->shape_.empty()) {
        new_subset.push_back(linear_index);
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
    const std::vector<symbolic::Expression>& shape
)
    : MathNode(element_id, debug_info, vertex, parent, code, {"C"}, {"A", "B"}, data_flow::ImplementationType_NONE),
      shape_(shape) {}

symbolic::SymbolSet ElementWiseBinaryNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ElementWiseBinaryNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ElementWiseBinaryNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 2 || graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("ElementWiseBinaryNode: Node must have exactly two inputs and one output");
    }

    auto iedge_a = &(*graph.in_edges(*this).begin());
    auto iedge_b = &(*(++graph.in_edges(*this).begin()));
    if (iedge_a->dst_conn() != "A") {
        std::swap(iedge_a, iedge_b);
    }

    if (iedge_a->base_type().type_id() != types::TypeID::Scalar &&
        iedge_a->base_type().type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException(
            "ElementWiseBinaryNode: Input A memlet must be of scalar or pointer type. Found type: " +
            iedge_a->base_type().print()
        );
    }
    if (iedge_a->base_type().type_id() == types::TypeID::Pointer) {
        auto& ptr_type = static_cast<const types::Pointer&>(iedge_a->base_type());
        if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ElementWiseBinaryNode: Input A memlet pointer be flat. Found type: " + ptr_type.pointee_type().print()
            );
        }
        if (!iedge_a->subset().empty()) {
            throw InvalidSDFGException("ElementWiseBinaryNode: Input A memlet pointer must not be dereferenced.");
        }
    }
    if (iedge_b->base_type().type_id() != types::TypeID::Scalar &&
        iedge_b->base_type().type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException(
            "ElementWiseBinaryNode: Input B memlet must be of scalar or pointer type. Found type: " +
            iedge_b->base_type().print()
        );
    }
    if (iedge_b->base_type().type_id() == types::TypeID::Pointer) {
        auto& ptr_type = static_cast<const types::Pointer&>(iedge_b->base_type());
        if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ElementWiseBinaryNode: Input B memlet pointer be flat. Found type: " + ptr_type.pointee_type().print()
            );
        }
        if (!iedge_b->subset().empty()) {
            throw InvalidSDFGException("ElementWiseBinaryNode: Input B memlet pointer must not be dereferenced.");
        }
    }

    auto& oedge = *graph.out_edges(*this).begin();
    if (oedge.base_type().type_id() != types::TypeID::Scalar && oedge.base_type().type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException(
            "ElementWiseBinaryNode: Output memlet must be of scalar or pointer type. Found type: " +
            oedge.base_type().print()
        );
    }
    if (oedge.base_type().type_id() == types::TypeID::Pointer) {
        auto& ptr_type = static_cast<const types::Pointer&>(oedge.base_type());
        if (ptr_type.pointee_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ElementWiseBinaryNode: Output memlet pointer be flat. Found type: " + ptr_type.pointee_type().print()
            );
        }
        if (!oedge.subset().empty()) {
            throw InvalidSDFGException("ElementWiseBinaryNode: Output memlet pointer must not be dereferenced.");
        }
    }
}

bool ElementWiseBinaryNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 1) {
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
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    for (size_t i = 0; i < this->shape_.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, this->shape_[i]);
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

    // Linearize subset
    symbolic::Expression linear_index = symbolic::zero();
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        linear_index = symbolic::add(symbolic::mul(linear_index, this->shape_[i]), loop_vars[i]);
    }
    if (!this->shape_.empty()) {
        new_subset.push_back(linear_index);
    }

    // Add tasklet block
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

} // namespace tensor
} // namespace math
} // namespace sdfg

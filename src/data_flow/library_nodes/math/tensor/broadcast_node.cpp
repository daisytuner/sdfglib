#include "sdfg/data_flow/library_nodes/math/tensor/broadcast_node.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"

namespace sdfg {
namespace math {
namespace tensor {

BroadcastNode::BroadcastNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& input_shape,
    const std::vector<symbolic::Expression>& output_shape
)
    : MathNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Broadcast,
          {"Y"},
          {"X"},
          data_flow::ImplementationType_NONE
      ),
      input_shape_(input_shape), output_shape_(output_shape) {}

symbolic::SymbolSet BroadcastNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : input_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : output_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void BroadcastNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : output_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void BroadcastNode::validate(const Function& function) const {}

bool BroadcastNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));

    auto& in_edge = *dataflow.in_edges(*this).begin();
    auto& out_edge = *dataflow.out_edges(*this).begin();
    auto& in_node = static_cast<data_flow::AccessNode&>(in_edge.src());
    auto& out_node = static_cast<data_flow::AccessNode&>(out_edge.dst());

    symbolic::SymbolVec loop_vars;
    structured_control_flow::Sequence* inner_scope = nullptr;

    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));

        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, output_shape_[i]);
        auto init = symbolic::zero();
        auto update = symbolic::add(sym_var, symbolic::one());

        if (i == 0) {
            auto& loop = builder.add_map_before(
                parent,
                block,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                {},
                this->debug_info()
            );
            inner_scope = &loop.root();
        } else {
            auto& loop = builder.add_map(
                *inner_scope,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                {},
                this->debug_info()
            );
            inner_scope = &loop.root();
        }
        loop_vars.push_back(sym_var);
    }

    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());

    auto& in_acc = builder.add_access(tasklet_block, in_node.data());
    auto& out_acc = builder.add_access(tasklet_block, out_node.data());

    auto& tasklet =
        builder.add_tasklet(tasklet_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());

    symbolic::Expression input_linear_index = symbolic::zero();
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        if (!symbolic::eq(input_shape_[i], symbolic::one())) {
            input_linear_index = symbolic::add(symbolic::mul(input_linear_index, input_shape_[i]), loop_vars[i]);
        }
    }

    symbolic::Expression output_linear_index = symbolic::zero();
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        output_linear_index = symbolic::add(symbolic::mul(output_linear_index, output_shape_[i]), loop_vars[i]);
    }

    data_flow::Subset input_subset;
    if (in_edge.base_type().type_id() != types::TypeID::Scalar) {
        input_subset = {input_linear_index};
    }
    data_flow::Subset output_subset;
    if (out_edge.base_type().type_id() != types::TypeID::Scalar) {
        output_subset = {output_linear_index};
    }
    builder.add_computational_memlet(
        tasklet_block, in_acc, tasklet, "_in", input_subset, in_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, tasklet, "_out", out_acc, output_subset, out_edge.base_type(), this->debug_info()
    );

    builder.remove_memlet(block, in_edge);
    builder.remove_memlet(block, out_edge);
    builder.remove_node(block, in_node);
    builder.remove_node(block, out_node);
    builder.remove_node(block, *this);

    int index = parent.index(block);
    builder.remove_child(parent, index);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> BroadcastNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new BroadcastNode(element_id, this->debug_info(), vertex, parent, input_shape_, output_shape_)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg

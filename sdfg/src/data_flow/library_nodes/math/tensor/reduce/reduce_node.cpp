#include "sdfg/data_flow/library_nodes/math/tensor/reduce/reduce_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"

#include <algorithm>

namespace sdfg {
namespace math {
namespace tensor {

ReduceNode::ReduceNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : TensorNode(element_id, debug_info, vertex, parent, code, {"Y"}, {"X"}, data_flow::ImplementationType_NONE),
      shape_(shape), axes_(axes), keepdims_(keepdims) {}

symbolic::SymbolSet ReduceNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ReduceNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

bool ReduceNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());

    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Calculate output shape
    std::vector<symbolic::Expression> output_shape;
    std::vector<int64_t> sorted_axes = axes_;
    std::sort(sorted_axes.begin(), sorted_axes.end());

    for (size_t i = 0; i < shape_.size(); ++i) {
        bool is_axis = false;
        for (auto axis : sorted_axes) {
            if (axis == (int64_t) i) {
                is_axis = true;
                break;
            }
        }

        if (is_axis) {
            if (keepdims_) {
                output_shape.push_back(symbolic::one());
            }
        } else {
            output_shape.push_back(shape_[i]);
        }
    }

    sdfg::types::Scalar element_type(oedge.base_type().primitive_type());

    // Add new sequence
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // 1. Initialization Loop
    {
        symbolic::Expression init_expr = symbolic::zero();
        structured_control_flow::Sequence* last_scope = &new_sequence;
        structured_control_flow::Map* last_map = nullptr;

        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::string indvar_str = builder.find_new_name("_i_init");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, output_shape[i]);

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
            init_expr = symbolic::add(symbolic::mul(init_expr, output_shape[i]), indvar);
        }
        data_flow::Subset init_subset;
        if (!output_shape.empty()) {
            init_subset.push_back(init_expr);
        }

        // Add initialization tasklet
        auto& init_block = builder.add_block(*last_scope, {}, block.debug_info());
        auto& init_tasklet =
            builder.add_tasklet(init_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, block.debug_info());

        auto& const_node = builder.add_constant(init_block, this->identity(), element_type, block.debug_info());
        auto& out_access = builder.add_access(init_block, output_node.data(), block.debug_info());

        builder
            .add_computational_memlet(init_block, const_node, init_tasklet, "_in", {}, element_type, block.debug_info());
        builder.add_computational_memlet(
            init_block, init_tasklet, "_out", out_access, init_subset, oedge.base_type(), block.debug_info()
        );
    }

    // 2. Reduction Loop
    {
        data_flow::Subset input_subset;
        data_flow::Subset output_subset;

        structured_control_flow::Sequence* last_scope = &new_sequence;
        structured_control_flow::StructuredLoop* last_loop = nullptr;

        std::map<size_t, symbolic::Expression> loop_vars_map;
        std::vector<size_t> outer_dims;
        std::vector<size_t> inner_dims;

        // Partition dimensions
        for (size_t i = 0; i < shape_.size(); ++i) {
            bool is_axis = false;
            for (auto axis : sorted_axes) {
                if (axis == (int64_t) i) {
                    is_axis = true;
                    break;
                }
            }
            if (is_axis) {
                inner_dims.push_back(i);
            } else {
                outer_dims.push_back(i);
            }
        }

        // Generate outer parallel loops (Maps)
        for (size_t dim_idx : outer_dims) {
            std::string indvar_str = builder.find_new_name("_i");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, shape_[dim_idx]);

            auto& map = builder.add_map(
                *last_scope,
                indvar,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                {},
                block.debug_info()
            );
            last_scope = &map.root();
            loop_vars_map[dim_idx] = indvar;
        }

        // Generate inner sequential loops (Fors)
        for (size_t dim_idx : inner_dims) {
            std::string indvar_str = builder.find_new_name("_k");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, shape_[dim_idx]);

            last_loop = &builder.add_for(*last_scope, indvar, condition, init, update, {}, block.debug_info());
            last_scope = &last_loop->root();
            loop_vars_map[dim_idx] = indvar;
        }

        // Linearize input
        symbolic::Expression input_linear_index = symbolic::zero();
        for (size_t i = 0; i < shape_.size(); ++i) {
            input_linear_index = symbolic::add(symbolic::mul(input_linear_index, shape_[i]), loop_vars_map[i]);
        }
        if (!shape_.empty()) {
            input_subset.push_back(input_linear_index);
        }

        // Construct output indices
        std::vector<symbolic::Expression> output_indices;
        for (size_t i = 0; i < shape_.size(); ++i) {
            bool is_axis = false;
            for (auto axis : sorted_axes) {
                if (axis == (int64_t) i) {
                    is_axis = true;
                    break;
                }
            }

            if (is_axis) {
                if (keepdims_) {
                    output_indices.push_back(symbolic::zero());
                }
            } else {
                output_indices.push_back(loop_vars_map[i]);
            }
        }

        // Linearize output
        symbolic::Expression output_linear_index = symbolic::zero();
        for (size_t i = 0; i < output_shape.size(); ++i) {
            output_linear_index = symbolic::add(symbolic::mul(output_linear_index, output_shape[i]), output_indices[i]);
        }
        if (!output_shape.empty()) {
            output_subset.push_back(output_linear_index);
        }

        this->expand_reduction(
            builder,
            analysis_manager,
            *last_scope,
            input_node.data(),
            output_node.data(),
            iedge.base_type(),
            oedge.base_type(),
            input_subset,
            output_subset
        );
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

} // namespace tensor
} // namespace math
} // namespace sdfg

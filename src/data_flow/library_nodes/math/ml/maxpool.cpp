#include "sdfg/data_flow/library_nodes/math/ml/maxpool.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace ml {

MaxPoolNode::MaxPoolNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& output,
    const std::string& input,
    const std::vector<int>& dilations,
    const std::vector<int>& kernel_shape,
    const std::vector<int>& pads,
    const std::vector<int>& strides
)
    : MathNode(element_id, debug_info, vertex, parent, LibraryNodeType_MaxPool, {output}, {input}),
      dilations_(dilations), kernel_shape_(kernel_shape), pads_(pads), strides_(strides) {}

bool MaxPoolNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    // must have exactly one in‐ and one out‐edge
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) return false;

    auto& scope_analyisis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analyisis.parent_scope(&block));

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    // scalar element type
    auto& type = sdfg.type(input);
    types::Scalar scalar_type(type.primitive_type());

    // grab the single in‐ and out‐edges
    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // the access nodes feeding/consuming this library node must be “loose”
    auto& input_node = iedge.src();
    auto& output_node = oedge.dst();
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) return false;

    // splice in a new Sequence right after `block`
    auto& seq = builder.add_sequence_before(parent, block, block.debug_info()).first;

    // Get input and output subset bounds:
    auto& in_begin = iedge.begin_subset();
    auto& in_end = iedge.end_subset();
    auto& out_begin = oedge.begin_subset();
    auto& out_end = oedge.end_subset();

    // 1) Loop over input dims with stride steps:
    Sequence* scope = &seq;
    symbolic::SymbolVec in_vars;
    for (size_t d = 0; d < in_begin.size(); ++d) {
        // ivar_d runs from in_begin[d] up to in_end[d], stepping by `strides_[d]`
        std::string ivar_name = builder.find_new_name("_i");
        builder.add_container(ivar_name, types::Scalar(types::PrimitiveType::UInt64));
        auto iv = symbolic::symbol(ivar_name);
        auto init = in_begin[d];
        auto cond = symbolic::Lt(iv, symbolic::add(in_end[d], symbolic::one()));
        auto update = symbolic::add(iv, symbolic::integer(strides_[d]));
        auto& map = builder.add_map(*scope, iv, cond, init, update, ScheduleType_Sequential, {}, block.debug_info());
        scope = &map.root();
        in_vars.push_back(iv);
    }
}


std::unique_ptr<data_flow::DataFlowNode> MaxPoolNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MaxPoolNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->outputs_.at(0),
        this->inputs_.at(0),
        this->dilations_,
        this->kernel_shape_,
        this->pads_,
        this->strides_
    ));
}

} // namespace ml
} // namespace math
} // namespace sdfg

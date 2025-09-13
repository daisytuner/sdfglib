#include "sdfg/data_flow/library_nodes/math/ml/maxpool.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace ml {

/*************** Constructor ***************/
MaxPoolNode::MaxPoolNode(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    std::vector<symbolic::Expression> shape,
    std::vector<size_t> kernel_shape,
    std::vector<size_t> pads,
    std::vector<size_t> strides
)
    : MathNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_MaxPool, {"Y"}, {"X"}, data_flow::ImplementationType_NONE
      ),
      shape_(shape), kernel_shape_(std::move(kernel_shape)), pads_(std::move(pads)), strides_(std::move(strides)) {}

/*************** Accessors ***************/
const std::vector<symbolic::Expression> &MaxPoolNode::shape() const { return shape_; }
std::vector<size_t> MaxPoolNode::kernel_shape() const { return kernel_shape_; }
std::vector<size_t> MaxPoolNode::pads() const { return pads_; }
std::vector<size_t> MaxPoolNode::strides() const { return strides_; }

symbolic::SymbolSet MaxPoolNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto &dim : shape_) {
        for (auto &atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void MaxPoolNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto &dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void MaxPoolNode::validate(const Function &) const {}

/*************** Expand ***************/
bool MaxPoolNode::expand(builder::StructuredSDFGBuilder &builder, analysis::AnalysisManager &analysis_manager) {
    auto &dataflow = this->get_parent();
    auto &block = static_cast<structured_control_flow::Block &>(*dataflow.get_parent());

    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto &parent = static_cast<structured_control_flow::Sequence &>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto &transition = parent.at(index).second;

    // Locate edges
    const data_flow::Memlet *iedge_X = nullptr;
    const data_flow::Memlet *oedge_Y = nullptr;
    for (const auto &edge : dataflow.in_edges(*this)) {
        if (edge.dst_conn() == "X") {
            iedge_X = &edge;
        }
    }
    for (const auto &edge : dataflow.out_edges(*this)) {
        if (edge.src_conn() == "Y") {
            oedge_Y = &edge;
        }
    }
    if (!iedge_X || !oedge_Y) return false;

    std::string X_name = static_cast<const data_flow::AccessNode &>(iedge_X->src()).data();
    std::string Y_name = static_cast<const data_flow::AccessNode &>(oedge_Y->dst()).data();

    // Create new sequence before
    auto &new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());
    structured_control_flow::Sequence *last_scope = &new_sequence;

    // Create maps over output subset dims (parallel dims)
    data_flow::Subset out_subset;
    std::vector<symbolic::Expression> out_syms;
    structured_control_flow::Map *last_map = nullptr;
    for (size_t d = 0; d < this->shape_.size(); ++d) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));
        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto cond = symbolic::Lt(indvar, this->shape_[d]);
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            cond,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();
        out_subset.push_back(indvar);
        out_syms.push_back(indvar);
    }

    // Kernel reduction loops
    structured_control_flow::For *last_for = nullptr;
    std::vector<symbolic::Expression> kernel_syms;
    for (size_t d = 0; d < kernel_shape_.size(); ++d) {
        std::string indvar_str = builder.find_new_name("_k");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));
        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::integer(0);
        auto update = symbolic::add(indvar, symbolic::one());
        auto cond = symbolic::Lt(indvar, symbolic::integer(static_cast<int64_t>(kernel_shape_[d])));
        last_for = &builder.add_for(*last_scope, indvar, cond, init, update, {}, block.debug_info());
        last_scope = &last_for->root();
        kernel_syms.push_back(indvar);
    }

    // Build subsets
    // infer: X dims N,C,H,W => assume same dims as Y except spatial dims with stride/pad

    auto int_expr = [](size_t v) { return symbolic::integer(static_cast<int64_t>(v)); };
    auto get_stride = [&](size_t idx) -> symbolic::Expression {
        return idx < strides_.size() ? int_expr(strides_[idx]) : symbolic::one();
    };
    auto get_pad = [&](size_t idx) -> symbolic::Expression {
        return idx < pads_.size() ? int_expr(pads_[idx]) : symbolic::zero();
    };

    // Create innermost block
    auto &code_block = builder.add_block(*last_scope, {}, block.debug_info());

    // Access nodes
    const DebugInfo dbg = block.debug_info();
    auto &X_acc = builder.add_access(code_block, X_name, dbg);
    auto &Y_acc_in = builder.add_access(code_block, Y_name, dbg);
    auto &Y_acc_out = builder.add_access(code_block, Y_name, dbg);

    // Build X subset using output coords * stride - pad + kernel_idx
    data_flow::Subset subset_X;
    // Assume dims: N, C, spatial...
    subset_X.push_back(out_syms[0]); // N
    subset_X.push_back(out_syms[1]); // C
    for (size_t d = 0; d < kernel_syms.size(); ++d) {
        auto expr =
            symbolic::sub(symbolic::add(symbolic::mul(get_stride(d), out_syms[2 + d]), kernel_syms[d]), get_pad(d));
        subset_X.push_back(expr);
    }

    // Y subset is just out_subset
    data_flow::Subset subset_Y = out_subset;

    // Tasklet max
    auto &tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::max, "_out", {"_in1", "_in2"}, dbg);

    builder.add_computational_memlet(code_block, Y_acc_in, tasklet, "_in1", subset_Y, oedge_Y->base_type(), dbg);
    builder.add_computational_memlet(code_block, X_acc, tasklet, "_in2", subset_X, iedge_X->base_type(), dbg);
    builder.add_computational_memlet(code_block, tasklet, "_out", Y_acc_out, subset_Y, oedge_Y->base_type(), dbg);

    // Cleanup old block
    builder.remove_memlet(block, *iedge_X);
    builder.remove_memlet(block, *oedge_Y);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

/*************** Clone ***************/
std::unique_ptr<data_flow::DataFlowNode> MaxPoolNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MaxPoolNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, this->kernel_shape_, this->pads_, this->strides_
    ));
}

/*************** Serializer ***************/
nlohmann::json MaxPoolNodeSerializer::serialize(const data_flow::LibraryNode &library_node) {
    const MaxPoolNode &node = static_cast<const MaxPoolNode &>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["kernel_shape"] = node.kernel_shape();
    j["pads"] = node.pads();
    j["strides"] = node.strides();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto &dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode &MaxPoolNodeSerializer::deserialize(
    const nlohmann::json &j, builder::StructuredSDFGBuilder &builder, structured_control_flow::Block &parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_MaxPool.value()) {
        throw std::runtime_error("Invalid library node code");
    }
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto kernel_shape = j["kernel_shape"].get<std::vector<size_t>>();
    auto pads = j["pads"].get<std::vector<size_t>>();
    auto strides = j["strides"].get<std::vector<size_t>>();

    std::vector<symbolic::Expression> shape;
    for (const auto &dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    return builder.add_library_node<MaxPoolNode>(parent, debug_info, shape, kernel_shape, pads, strides);
}

} // namespace ml
} // namespace math
} // namespace sdfg

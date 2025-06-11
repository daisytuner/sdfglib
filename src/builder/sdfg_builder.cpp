#include "sdfg/builder/sdfg_builder.h"

#include "sdfg/types/utils.h"

namespace sdfg {
namespace builder {

Function& SDFGBuilder::function() const { return static_cast<Function&>(*this->sdfg_); };

SDFGBuilder::SDFGBuilder(std::unique_ptr<SDFG>& sdfg)
    : FunctionBuilder(), sdfg_(std::move(sdfg)) {

      };

SDFGBuilder::SDFGBuilder(const std::string& name, FunctionType type)
    : FunctionBuilder(), sdfg_(new SDFG(name, type)) {

      };

SDFG& SDFGBuilder::subject() const { return *this->sdfg_; };

std::unique_ptr<SDFG> SDFGBuilder::move() { return std::move(this->sdfg_); };

/***** Section: Control-Flow Graph *****/

control_flow::State& SDFGBuilder::add_state(bool is_start_state, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(this->sdfg_->graph_);
    auto res = this->sdfg_->states_.insert(
        {vertex,
         std::unique_ptr<control_flow::State>(new control_flow::State(debug_info, vertex))});

    assert(res.second);
    (*res.first).second->dataflow_->parent_ = (*res.first).second.get();

    if (is_start_state) {
        this->sdfg_->start_state_ = (*res.first).second.get();
    }

    return *(*res.first).second;
};

control_flow::State& SDFGBuilder::add_state_before(const control_flow::State& state,
                                                   bool is_start_state,
                                                   const DebugInfo& debug_info) {
    auto& new_state = this->add_state(false, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->in_edges(state)) to_redirect.push_back(&e);

    // Redirect control-flow
    for (auto edge : to_redirect) {
        this->add_edge(edge->src(), new_state, edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    this->add_edge(new_state, state);

    if (is_start_state) {
        this->sdfg_->start_state_ = &new_state;
    }

    return new_state;
};

control_flow::State& SDFGBuilder::add_state_after(const control_flow::State& state,
                                                  bool connect_states,
                                                  const DebugInfo& debug_info) {
    auto& new_state = this->add_state(false, debug_info);

    std::vector<const control_flow::InterstateEdge*> to_redirect;
    for (auto& e : this->sdfg_->out_edges(state)) to_redirect.push_back(&e);

    // Redirect control-flow
    for (auto& edge : to_redirect) {
        this->add_edge(new_state, edge->dst(), edge->condition());

        auto desc = edge->edge();
        this->sdfg_->edges_.erase(desc);
        boost::remove_edge(desc, this->sdfg_->graph_);
    }
    if (connect_states) {
        this->add_edge(state, new_state);
    }

    return new_state;
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(const control_flow::State& src,
                                                    const control_flow::State& dst,
                                                    const DebugInfo& debug_info) {
    return this->add_edge(src, dst, symbolic::Assignments{}, SymEngine::boolTrue, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(const control_flow::State& src,
                                                    const control_flow::State& dst,
                                                    const symbolic::Condition condition,
                                                    const DebugInfo& debug_info) {
    return this->add_edge(src, dst, symbolic::Assignments{}, condition, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(const control_flow::State& src,
                                                    const control_flow::State& dst,
                                                    const symbolic::Assignments& assignments,
                                                    const DebugInfo& debug_info) {
    return this->add_edge(src, dst, assignments, SymEngine::boolTrue, debug_info);
};

control_flow::InterstateEdge& SDFGBuilder::add_edge(const control_flow::State& src,
                                                    const control_flow::State& dst,
                                                    const symbolic::Assignments& assignments,
                                                    const symbolic::Condition condition,
                                                    const DebugInfo& debug_info) {
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, this->sdfg_->graph_);
    assert(edge.second);

    auto res = this->sdfg_->edges_.insert(
        {edge.first, std::unique_ptr<control_flow::InterstateEdge>(new control_flow::InterstateEdge(
                         debug_info, edge.first, src, dst, condition, assignments))});

    assert(res.second);

    return *(*res.first).second;
};

void SDFGBuilder::remove_edge(const control_flow::InterstateEdge& edge) {
    auto desc = edge.edge();
    this->sdfg_->edges_.erase(desc);

    boost::remove_edge(desc, this->sdfg_->graph_);
};

std::tuple<control_flow::State&, control_flow::State&, control_flow::State&> SDFGBuilder::add_loop(
    const control_flow::State& state, sdfg::symbolic::Symbol iterator,
    sdfg::symbolic::Expression init, sdfg::symbolic::Condition cond,
    sdfg::symbolic::Expression update, const DebugInfo& debug_info) {
    // Init: iterator = init
    auto& init_state = this->add_state_after(state, true, debug_info);
    const graph::Edge init_edge_desc = (*this->sdfg_->in_edges(init_state).begin()).edge_;
    auto& init_edge = this->sdfg_->edges_[init_edge_desc];
    init_edge->assignments_.insert({iterator, init});

    // Final state
    auto& final_state = this->add_state_after(init_state, false, debug_info);

    // Init -> early_exit -> final
    auto& early_exit_state = this->add_state(false, debug_info);
    this->add_edge(init_state, early_exit_state, symbolic::Not(cond));
    this->add_edge(early_exit_state, final_state);

    // Init -> header -> body
    auto& header_state = this->add_state(false, debug_info);
    this->add_edge(init_state, header_state, cond);

    auto& body_state = this->add_state(false, debug_info);
    this->add_edge(header_state, body_state);

    auto& update_state = this->add_state(false, debug_info);
    this->add_edge(body_state, update_state, {{iterator, update}});

    // Back edge and exit edge
    this->add_edge(update_state, header_state, cond);
    this->add_edge(update_state, final_state, symbolic::Not(cond));

    return {init_state, body_state, final_state};
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& SDFGBuilder::add_access(control_flow::State& state, const std::string& data,
                                               const DebugInfo& debug_info) {
    // Check: Data exists
    if (!this->subject().exists(data)) {
        throw InvalidSDFGException("Data does not exist in SDFG: " + data);
    }

    auto& dataflow = state.dataflow();
    auto vertex = boost::add_vertex(dataflow.graph_);
    auto res = dataflow.nodes_.insert(
        {vertex, std::unique_ptr<data_flow::AccessNode>(
                     new data_flow::AccessNode(debug_info, vertex, dataflow, data))});

    return dynamic_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::Tasklet& SDFGBuilder::add_tasklet(
    control_flow::State& state, const data_flow::TaskletCode code,
    const std::pair<std::string, sdfg::types::Scalar>& output,
    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
    const DebugInfo& debug_info) {
    // Check: Duplicate inputs
    std::unordered_set<std::string> input_names;
    for (auto& input : inputs) {
        if (!input.first.starts_with("_in")) {
            continue;
        }
        if (input_names.find(input.first) != input_names.end()) {
            throw InvalidSDFGException("Input " + input.first + " already exists in SDFG");
        }
        input_names.insert(input.first);
    }

    auto& dataflow = state.dataflow();
    auto vertex = boost::add_vertex(dataflow.graph_);
    auto res = dataflow.nodes_.insert(
        {vertex, std::unique_ptr<data_flow::Tasklet>(new data_flow::Tasklet(
                     debug_info, vertex, dataflow, code, output, inputs, symbolic::__true__()))});

    return dynamic_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& SDFGBuilder::add_memlet(control_flow::State& state, data_flow::DataFlowNode& src,
                                           const std::string& src_conn,
                                           data_flow::DataFlowNode& dst,
                                           const std::string& dst_conn,
                                           const data_flow::Subset& subset,
                                           const DebugInfo& debug_info) {
    auto& function_ = this->function();

    // Check - Case 1: Access Node -> Access Node
    // - src_conn or dst_conn must be refs. The other must be void.
    // - The side of the memlet that is void, is dereferenced.
    // - The dst type must always be a pointer after potential dereferencing.
    // - The src type can be any type after dereferecing (&dereferenced_src_type).
    if (dynamic_cast<data_flow::AccessNode*>(&src) && dynamic_cast<data_flow::AccessNode*>(&dst)) {
        auto& src_node = dynamic_cast<data_flow::AccessNode&>(src);
        auto& dst_node = dynamic_cast<data_flow::AccessNode&>(dst);
        if (src_conn == "refs") {
            if (dst_conn != "void") {
                throw InvalidSDFGException("Invalid dst connector: " + dst_conn);
            }

            auto& dst_type = types::infer_type(function_, function_.type(dst_node.data()), subset);
            if (!dynamic_cast<const types::Pointer*>(&dst_type)) {
                throw InvalidSDFGException("dst type must be a pointer");
            }

            auto& src_type = function_.type(src_node.data());
            if (!dynamic_cast<const types::Pointer*>(&src_type)) {
                throw InvalidSDFGException("src type must be a pointer");
            }
        } else if (src_conn == "void") {
            if (dst_conn != "refs") {
                throw InvalidSDFGException("Invalid dst connector: " + dst_conn);
            }

            if (symbolic::is_pointer(symbolic::symbol(src_node.data()))) {
                throw InvalidSDFGException("src_conn is void: src cannot be a raw pointer");
            }

            // Trivially correct but checks inference
            auto& src_type = types::infer_type(function_, function_.type(src_node.data()), subset);
            types::Pointer ref_type(src_type);
            if (!dynamic_cast<const types::Pointer*>(&ref_type)) {
                throw InvalidSDFGException("src type must be a pointer");
            }

            auto& dst_type = function_.type(dst_node.data());
            if (!dynamic_cast<const types::Pointer*>(&dst_type)) {
                throw InvalidSDFGException("dst type must be a pointer");
            }
        } else {
            throw InvalidSDFGException("Invalid src connector: " + src_conn);
        }
    } else if (dynamic_cast<data_flow::AccessNode*>(&src) &&
               dynamic_cast<data_flow::Tasklet*>(&dst)) {
        auto& src_node = dynamic_cast<data_flow::AccessNode&>(src);
        auto& dst_node = dynamic_cast<data_flow::Tasklet&>(dst);
        if (src_conn != "void") {
            throw InvalidSDFGException("src_conn must be void. Found: " + src_conn);
        }
        bool found = false;
        for (auto& input : dst_node.inputs()) {
            if (input.first == dst_conn) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw InvalidSDFGException("dst_conn not found in tasklet: " + dst_conn);
        }
        auto& element_type = types::infer_type(function_, function_.type(src_node.data()), subset);
        if (!dynamic_cast<const types::Scalar*>(&element_type)) {
            throw InvalidSDFGException("Tasklets inputs must be scalars");
        }
    } else if (dynamic_cast<data_flow::Tasklet*>(&src) &&
               dynamic_cast<data_flow::AccessNode*>(&dst)) {
        auto& src_node = dynamic_cast<data_flow::Tasklet&>(src);
        auto& dst_node = dynamic_cast<data_flow::AccessNode&>(dst);
        if (src_conn != src_node.output().first) {
            throw InvalidSDFGException("src_conn must match tasklet output name");
        }
        if (dst_conn != "void") {
            throw InvalidSDFGException("dst_conn must be void. Found: " + dst_conn);
        }

        auto& element_type = types::infer_type(function_, function_.type(dst_node.data()), subset);
        if (!dynamic_cast<const types::Scalar*>(&element_type)) {
            throw InvalidSDFGException("Tasklet output must be a scalar");
        }
    } else if (dynamic_cast<data_flow::AccessNode*>(&src) &&
               dynamic_cast<data_flow::LibraryNode*>(&dst)) {
        auto& dst_node = dynamic_cast<data_flow::LibraryNode&>(dst);
        if (src_conn != "void") {
            throw InvalidSDFGException("src_conn must be void. Found: " + src_conn);
        }
        bool found = false;
        for (auto& input : dst_node.inputs()) {
            if (input == dst_conn) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw InvalidSDFGException("dst_conn not found in library node: " + dst_conn);
        }
    } else if (dynamic_cast<data_flow::LibraryNode*>(&src) &&
               dynamic_cast<data_flow::AccessNode*>(&dst)) {
        auto& src_node = dynamic_cast<data_flow::LibraryNode&>(src);
        auto& dst_node = dynamic_cast<data_flow::AccessNode&>(dst);
        if (dst_conn != "void") {
            throw InvalidSDFGException("dst_conn must be void. Found: " + dst_conn);
        }
        bool found = false;
        for (auto& output : src_node.outputs()) {
            if (output == src_conn) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw InvalidSDFGException("src_conn not found in library node: " + src_conn);
        }

        auto& element_type = types::infer_type(function_, function_.type(dst_node.data()), subset);
        if (!dynamic_cast<const types::Pointer*>(&element_type)) {
            throw InvalidSDFGException("Access node must be a pointer");
        }
    } else {
        throw InvalidSDFGException("Invalid src or dst node type");
    }

    auto& dataflow = state.dataflow();
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, dataflow.graph_);
    auto res = dataflow.edges_.insert(
        {edge.first, std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
                         debug_info, edge.first, dataflow, src, src_conn, dst, dst_conn, subset))});

    return dynamic_cast<data_flow::Memlet&>(*(res.first->second));
};

}  // namespace builder
}  // namespace sdfg

#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/data_flow/library_node.h"

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

std::unordered_set<const control_flow::State*> StructuredSDFGBuilder::determine_loop_nodes(
    const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const {
    std::unordered_set<const control_flow::State*> nodes;
    std::unordered_set<const control_flow::State*> visited;
    std::list<const control_flow::State*> queue = {&start};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();
        if (visited.find(curr) != visited.end()) {
            continue;
        }
        visited.insert(curr);

        nodes.insert(curr);
        if (curr == &end) {
            continue;
        }

        for (auto& iedge : sdfg.in_edges(*curr)) {
            queue.push_back(&iedge.src());
        }
    }

    return nodes;
};

void StructuredSDFGBuilder::traverse(const SDFG& sdfg) {
    // Start of SDFGS
    Sequence& root = *structured_sdfg_->root_;
    const State* start_state = &sdfg.start_state();

    // Analysis
    auto pdom_tree = sdfg.post_dominator_tree();
    auto back_edges = sdfg.back_edges();

    std::unordered_map<const InterstateEdge*, const While*> active_continues;
    std::unordered_map<const InterstateEdge*, const While*> active_breaks;
    this->traverse(sdfg, pdom_tree, back_edges, root, start_state, nullptr, active_continues,
                   active_breaks, false);
};

void StructuredSDFGBuilder::traverse(
    const SDFG& sdfg, std::unordered_map<const State*, const State*>& pdom_tree,
    std::list<const InterstateEdge*>& back_edges, Sequence& scope, const State* begin,
    const State* end, std::unordered_map<const InterstateEdge*, const While*>& active_continues,
    std::unordered_map<const InterstateEdge*, const While*>& active_breaks,
    bool skip_loop_detection) {
    std::list<const State*> queue = {begin};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (current == end) {
            break;
        }

        auto in_edges = sdfg.in_edges(*current);
        auto out_edges = sdfg.out_edges(*current);
        auto out_degree = sdfg.out_degree(*current);

        // Case 1: Start of loop
        std::unordered_set<const InterstateEdge*> loop_edges;
        if (!skip_loop_detection || begin != current) {
            for (auto& iedge : in_edges) {
                // Check if edge is a back edge
                if (std::find(back_edges.begin(), back_edges.end(), &iedge) != back_edges.end()) {
                    loop_edges.insert(&iedge);
                }
            }
        }
        if (!loop_edges.empty()) {
            // Determine body
            std::unordered_set<const control_flow::State*> loop_states;
            for (auto cont : loop_edges) {
                auto loop_nodes = this->determine_loop_nodes(sdfg, cont->src(), cont->dst());
                loop_states.insert(loop_nodes.begin(), loop_nodes.end());
            }

            // Determine breaks
            std::unordered_set<const control_flow::State*> exit_states;
            std::unordered_set<const control_flow::InterstateEdge*> enter_edges;
            std::unordered_set<const control_flow::InterstateEdge*> exit_edges;
            for (auto& node : loop_states) {
                for (auto& edge : sdfg.out_edges(*node)) {
                    if (loop_states.find(&edge.dst()) == loop_states.end()) {
                        exit_edges.insert(&edge);
                        exit_states.insert(&edge.dst());
                    }
                }
                for (auto& edge : sdfg.in_edges(*node)) {
                    if (loop_states.find(&edge.src()) == loop_states.end()) {
                        enter_edges.insert(&edge);
                    }
                }
            }
            assert(exit_states.size() == 1);

            // combine DebugInfo
            DebugInfo dbg_info;
            for (auto edge : enter_edges) {
                dbg_info = DebugInfo::merge(dbg_info, edge->debug_info());
                dbg_info = DebugInfo::merge(dbg_info, edge->dst().debug_info());
            }
            for (auto edge : exit_edges) {
                dbg_info = DebugInfo::merge(dbg_info, edge->debug_info());
            }

            While& loop = this->add_while(scope, {}, dbg_info);

            // Set continues and breaks as active temporarily
            std::unordered_map<const InterstateEdge*, const While*> new_active_continues;
            std::unordered_map<const InterstateEdge*, const While*> new_active_breaks;
            for (auto& entry : active_continues) {
                new_active_continues[entry.first] = entry.second;
            }
            for (auto& entry : active_breaks) {
                new_active_breaks[entry.first] = entry.second;
            }
            for (auto cont : loop_edges) {
                new_active_continues[cont] = &loop;
            }
            for (auto brk : exit_edges) {
                new_active_breaks[brk] = &loop;
            }

            // Traverse body
            this->traverse(sdfg, pdom_tree, back_edges, loop.root(), current, *exit_states.begin(),
                           new_active_continues, new_active_breaks, true);

            // Continue after loop
            for (auto exit_state : exit_states) {
                queue.push_back(exit_state);
            }
            continue;
        }

        // Case 2: Start of if-else
        if (out_degree > 1) {
            this->add_block(scope, current->dataflow(), {}, current->debug_info());

            // Traverse branches recursively
            const State* end_of_if_else = nullptr;
            IfElse& branch = this->add_if_else(scope, current->debug_info());
            for (auto& oedge : out_edges) {
                auto& branch_parent = this->add_case(branch, oedge.condition(), oedge.debug_info());
                this->add_block(branch_parent, oedge.assignments(), oedge.debug_info());

                if (active_continues.find(&oedge) != active_continues.end()) {
                    this->add_continue(branch_parent, *active_continues[&oedge],
                                       oedge.debug_info());
                } else if (active_breaks.find(&oedge) != active_breaks.end()) {
                    this->add_break(branch_parent, *active_breaks[&oedge], oedge.debug_info());
                } else {
                    // Determine end of if-else
                    auto pdom = pdom_tree[current];
                    assert(pdom != nullptr);

                    this->traverse(sdfg, pdom_tree, back_edges, branch_parent, &oedge.dst(), pdom,
                                   active_continues, active_breaks, false);

                    end_of_if_else = pdom;
                }
            }

            if (end_of_if_else != nullptr) {
                queue.push_back(end_of_if_else);
            }
            continue;
        }

        // Case 3: Transition
        if (out_degree == 1) {
            auto& oedge = *out_edges.begin();
            assert(oedge.is_unconditional());
            this->add_block(scope, current->dataflow(), oedge.assignments(), current->debug_info());

            // Determine if transition is a break or continue
            if (active_breaks.find(&oedge) != active_breaks.end()) {
                this->add_break(scope, *active_breaks[&oedge], oedge.debug_info());
            } else if (active_continues.find(&oedge) != active_continues.end()) {
                this->add_continue(scope, *active_continues[&oedge], oedge.debug_info());
            } else {
                queue.push_back(&oedge.dst());
            }
            continue;
        }

        // Case 4: Return
        if (out_degree == 0) {
            this->add_block(scope, current->dataflow(), {}, current->debug_info());
            this->add_return(scope, {}, current->debug_info());
            continue;
        }

        throw std::runtime_error("Failed to detect structured control flow pattern");
    }
    assert(queue.empty());
};

Function& StructuredSDFGBuilder::function() const {
    return static_cast<Function&>(*this->structured_sdfg_);
};

StructuredSDFGBuilder::StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg)
    : FunctionBuilder(), structured_sdfg_(std::move(sdfg)){};

StructuredSDFGBuilder::StructuredSDFGBuilder(const std::string& name)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(name)){};

StructuredSDFGBuilder::StructuredSDFGBuilder(const SDFG& sdfg)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(sdfg.name())) {
    for (auto& entry : sdfg.structures_) {
        this->structured_sdfg_->structures_.insert({entry.first, entry.second->clone()});
    }

    for (auto& entry : sdfg.containers_) {
        this->structured_sdfg_->containers_.insert({entry.first, entry.second->clone()});
    }

    for (auto& arg : sdfg.arguments_) {
        this->structured_sdfg_->arguments_.push_back(arg);
    }

    for (auto& ext : sdfg.externals_) {
        this->structured_sdfg_->externals_.push_back(ext);
    }

    for (auto& entry : sdfg.assumptions_) {
        this->structured_sdfg_->assumptions_.insert({entry.first, entry.second});
    }

    this->traverse(sdfg);
};

StructuredSDFG& StructuredSDFGBuilder::subject() const { return *this->structured_sdfg_; };

std::unique_ptr<StructuredSDFG> StructuredSDFGBuilder::move() {
    return std::move(this->structured_sdfg_);
};

Sequence& StructuredSDFGBuilder::add_sequence(Sequence& parent,
                                              const sdfg::symbolic::Assignments& assignments,
                                              const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Sequence>(new Sequence(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;

    return static_cast<Sequence&>(*parent.children_.back().get());
};

std::pair<Sequence&, Transition&> StructuredSDFGBuilder::add_sequence_before(
    Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<Sequence>(new Sequence(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Sequence&>(new_entry.first);

    return {new_block, new_entry.second};
};

void StructuredSDFGBuilder::remove_child(Sequence& parent, size_t i) {
    parent.children_.erase(parent.children_.begin() + i);
    parent.transitions_.erase(parent.transitions_.begin() + i);
};

void StructuredSDFGBuilder::remove_child(Sequence& parent, ControlFlowNode& child) {
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &child) {
            index = i;
            break;
        }
    }

    parent.children_.erase(parent.children_.begin() + index);
    parent.transitions_.erase(parent.transitions_.begin() + index);
};

void StructuredSDFGBuilder::insert_children(Sequence& parent, Sequence& other, size_t i) {
    parent.children_.insert(parent.children_.begin() + i,
                            std::make_move_iterator(other.children_.begin()),
                            std::make_move_iterator(other.children_.end()));
    parent.transitions_.insert(parent.transitions_.begin() + i,
                               std::make_move_iterator(other.transitions_.begin()),
                               std::make_move_iterator(other.transitions_.end()));
    other.children_.clear();
    other.transitions_.clear();
};

Block& StructuredSDFGBuilder::add_block(Sequence& parent,
                                        const sdfg::symbolic::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Block>(new Block(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;

    return static_cast<Block&>(*parent.children_.back().get());
};

Block& StructuredSDFGBuilder::add_block(Sequence& parent,
                                        const data_flow::DataFlowGraph& data_flow_graph,
                                        const sdfg::symbolic::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Block>(new Block(this->element_counter_, debug_info, data_flow_graph)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;

    return static_cast<Block&>(*parent.children_.back().get());
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_before(
    Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(parent.children_.begin() + index,
                            std::unique_ptr<Block>(new Block(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_before(
    Sequence& parent, ControlFlowNode& block, data_flow::DataFlowGraph& data_flow_graph,
    const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<Block>(new Block(this->element_counter_, debug_info, data_flow_graph)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_after(Sequence& parent,
                                                                      ControlFlowNode& block,
                                                                      const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(parent.children_.begin() + index + 1,
                            std::unique_ptr<Block>(new Block(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_after(
    Sequence& parent, ControlFlowNode& block, data_flow::DataFlowGraph& data_flow_graph,
    const DebugInfo& debug_info) {
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<Block>(new Block(this->element_counter_, debug_info, data_flow_graph)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);

    return {new_block, new_entry.second};
};

For& StructuredSDFGBuilder::add_for(Sequence& parent, const symbolic::Symbol& indvar,
                                    const symbolic::Condition& condition,
                                    const symbolic::Expression& init,
                                    const symbolic::Expression& update,
                                    const sdfg::symbolic::Assignments& assignments,
                                    const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<For>(
        new For(this->element_counter_, debug_info, indvar, init, update, condition)));
    this->element_counter_ = this->element_counter_ + 2;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;

    return static_cast<For&>(*parent.children_.back().get());
};

std::pair<For&, Transition&> StructuredSDFGBuilder::add_for_before(
    Sequence& parent, ControlFlowNode& block, const symbolic::Symbol& indvar,
    const symbolic::Condition& condition, const symbolic::Expression& init,
    const symbolic::Expression& update, const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(parent.children_.begin() + index,
                            std::unique_ptr<For>(new For(this->element_counter_, debug_info, indvar,
                                                         init, update, condition)));
    this->element_counter_ = this->element_counter_ + 2;
    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::For&>(new_entry.first);

    return {new_block, new_entry.second};
};

std::pair<For&, Transition&> StructuredSDFGBuilder::add_for_after(
    Sequence& parent, ControlFlowNode& block, const symbolic::Symbol& indvar,
    const symbolic::Condition& condition, const symbolic::Expression& init,
    const symbolic::Expression& update, const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(parent.children_.begin() + index + 1,
                            std::unique_ptr<For>(new For(this->element_counter_, debug_info, indvar,
                                                         init, update, condition)));
    this->element_counter_ = this->element_counter_ + 2;
    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::For&>(new_entry.first);

    return {new_block, new_entry.second};
};

IfElse& StructuredSDFGBuilder::add_if_else(Sequence& parent, const DebugInfo& debug_info) {
    return this->add_if_else(parent, symbolic::Assignments{}, debug_info);
};

IfElse& StructuredSDFGBuilder::add_if_else(Sequence& parent,
                                           const sdfg::symbolic::Assignments& assignments,
                                           const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<IfElse>(new IfElse(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;
    return static_cast<IfElse&>(*parent.children_.back().get());
};

std::pair<IfElse&, Transition&> StructuredSDFGBuilder::add_if_else_before(
    Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info) {
    // Insert block before current block
    int index = -1;
    for (size_t i = 0; i < parent.children_.size(); i++) {
        if (parent.children_.at(i).get() == &block) {
            index = i;
            break;
        }
    }
    assert(index > -1);

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<IfElse>(new IfElse(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::IfElse&>(new_entry.first);

    return {new_block, new_entry.second};
};

Sequence& StructuredSDFGBuilder::add_case(IfElse& scope, const sdfg::symbolic::Condition cond,
                                          const DebugInfo& debug_info) {
    scope.cases_.push_back(
        std::unique_ptr<Sequence>(new Sequence(this->element_counter_, debug_info)));
    this->element_counter_++;
    scope.conditions_.push_back(cond);
    return *scope.cases_.back();
};

void StructuredSDFGBuilder::remove_case(IfElse& scope, size_t i, const DebugInfo& debug_info) {
    scope.cases_.erase(scope.cases_.begin() + i);
    scope.conditions_.erase(scope.conditions_.begin() + i);
};

While& StructuredSDFGBuilder::add_while(Sequence& parent,
                                        const sdfg::symbolic::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<While>(new While(this->element_counter_, debug_info)));
    this->element_counter_ = this->element_counter_ + 2;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;
    return static_cast<While&>(*parent.children_.back().get());
};

Kernel& StructuredSDFGBuilder::add_kernel(
    Sequence& parent, const std::string& suffix, const DebugInfo& debug_info,
    const symbolic::Expression& gridDim_x_init, const symbolic::Expression& gridDim_y_init,
    const symbolic::Expression& gridDim_z_init, const symbolic::Expression& blockDim_x_init,
    const symbolic::Expression& blockDim_y_init, const symbolic::Expression& blockDim_z_init,
    const symbolic::Expression& blockIdx_x_init, const symbolic::Expression& blockIdx_y_init,
    const symbolic::Expression& blockIdx_z_init, const symbolic::Expression& threadIdx_x_init,
    const symbolic::Expression& threadIdx_y_init, const symbolic::Expression& threadIdx_z_init) {
    parent.children_.push_back(std::unique_ptr<Kernel>(new Kernel(
        this->element_counter_, debug_info, suffix, gridDim_x_init, gridDim_y_init, gridDim_z_init,
        blockDim_x_init, blockDim_y_init, blockDim_z_init, blockIdx_x_init, blockIdx_y_init,
        blockIdx_z_init, threadIdx_x_init, threadIdx_y_init, threadIdx_z_init)));
    this->element_counter_ = this->element_counter_ + 2;
    parent.transitions_.push_back(
        std::unique_ptr<Transition>(new Transition(this->element_counter_, debug_info)));
    this->element_counter_++;
    return static_cast<Kernel&>(*parent.children_.back().get());
};

Continue& StructuredSDFGBuilder::add_continue(Sequence& parent, const While& loop,
                                              const DebugInfo& debug_info) {
    return this->add_continue(parent, loop, symbolic::Assignments{}, debug_info);
};

Continue& StructuredSDFGBuilder::add_continue(Sequence& parent, const While& loop,
                                              const sdfg::symbolic::Assignments& assignments,
                                              const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Continue>(new Continue(this->element_counter_, debug_info, loop)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;
    return static_cast<Continue&>(*parent.children_.back().get());
};

Break& StructuredSDFGBuilder::add_break(Sequence& parent, const While& loop,
                                        const DebugInfo& debug_info) {
    return this->add_break(parent, loop, symbolic::Assignments{}, debug_info);
};

Break& StructuredSDFGBuilder::add_break(Sequence& parent, const While& loop,
                                        const sdfg::symbolic::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Break>(new Break(this->element_counter_, debug_info, loop)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;
    return static_cast<Break&>(*parent.children_.back().get());
};

Return& StructuredSDFGBuilder::add_return(Sequence& parent,
                                          const sdfg::symbolic::Assignments& assignments,
                                          const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Return>(new Return(this->element_counter_, debug_info)));
    this->element_counter_++;
    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->element_counter_, debug_info, assignments)));
    this->element_counter_++;
    return static_cast<Return&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::convert_while(Sequence& parent, While& loop,
                                          const symbolic::Symbol& indvar,
                                          const symbolic::Condition& condition,
                                          const symbolic::Expression& init,
                                          const symbolic::Expression& update) {
    // Insert for loop
    size_t index = 0;
    for (auto& entry : parent.children_) {
        if (entry.get() == &loop) {
            break;
        }
        index++;
    }
    auto iter = parent.children_.begin() + index;
    auto& new_iter = *parent.children_.insert(
        iter + 1, std::unique_ptr<For>(new For(this->element_counter_, loop.debug_info(), indvar,
                                               init, update, condition)));
    this->element_counter_ = this->element_counter_ + 2;
    auto& for_loop = dynamic_cast<For&>(*new_iter);
    this->insert_children(for_loop.root(), loop.root(), 0);

    // Remove while loop
    parent.children_.erase(parent.children_.begin() + index);

    return for_loop;
};

void StructuredSDFGBuilder::clear_sequence(Sequence& parent) {
    parent.children_.clear();
    parent.transitions_.clear();
};

Sequence& StructuredSDFGBuilder::parent(const ControlFlowNode& node) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&this->structured_sdfg_->root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                if (&sequence_stmt->at(i).first == &node) {
                    return *sequence_stmt;
                }
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return this->structured_sdfg_->root();
};

Kernel& StructuredSDFGBuilder::convert_into_kernel() {
    auto old_root = std::move(this->structured_sdfg_->root_);
    this->structured_sdfg_->root_ =
        std::unique_ptr<Sequence>(new Sequence(this->element_counter_, old_root->debug_info()));
    this->element_counter_++;
    auto& new_root = this->structured_sdfg_->root();
    auto& kernel = this->add_kernel(new_root, this->function().name());

    this->insert_children(kernel.root(), *old_root, 0);

    types::Scalar gridDim_x(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.gridDim_x()->get_name(), gridDim_x);
    types::Scalar gridDim_y(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.gridDim_y()->get_name(), gridDim_y);
    types::Scalar gridDim_z(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.gridDim_z()->get_name(), gridDim_z);
    types::Scalar blockDim_x(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockDim_x()->get_name(), blockDim_x);
    types::Scalar blockDim_y(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockDim_y()->get_name(), blockDim_y);
    types::Scalar blockDim_z(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockDim_z()->get_name(), blockDim_z);
    types::Scalar blockIdx_x(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockIdx_x()->get_name(), blockIdx_x);
    types::Scalar blockIdx_y(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockIdx_y()->get_name(), blockIdx_y);
    types::Scalar blockIdx_z(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.blockIdx_z()->get_name(), blockIdx_z);
    types::Scalar threadIdx_x(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.threadIdx_x()->get_name(), threadIdx_x);
    types::Scalar threadIdx_y(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.threadIdx_y()->get_name(), threadIdx_y);
    types::Scalar threadIdx_z(types::PrimitiveType::Int32, types::DeviceLocation::nvptx, 0);
    add_container(kernel.threadIdx_z()->get_name(), threadIdx_z);

    kernel.root().replace(symbolic::symbol("gridDim.x"), kernel.gridDim_x());
    kernel.root().replace(symbolic::symbol("gridDim.y"), kernel.gridDim_y());
    kernel.root().replace(symbolic::symbol("gridDim.z"), kernel.gridDim_z());

    kernel.root().replace(symbolic::symbol("blockDim.x"), kernel.blockDim_x());
    kernel.root().replace(symbolic::symbol("blockDim.y"), kernel.blockDim_y());
    kernel.root().replace(symbolic::symbol("blockDim.z"), kernel.blockDim_z());

    kernel.root().replace(symbolic::symbol("blockIdx.x"), kernel.blockIdx_x());
    kernel.root().replace(symbolic::symbol("blockIdx.y"), kernel.blockIdx_y());
    kernel.root().replace(symbolic::symbol("blockIdx.z"), kernel.blockIdx_z());

    kernel.root().replace(symbolic::symbol("threadIdx.x"), kernel.threadIdx_x());
    kernel.root().replace(symbolic::symbol("threadIdx.y"), kernel.threadIdx_y());
    kernel.root().replace(symbolic::symbol("threadIdx.z"), kernel.threadIdx_z());

    return kernel;
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& StructuredSDFGBuilder::add_access(structured_control_flow::Block& block,
                                                         const std::string& data,
                                                         const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex, std::unique_ptr<data_flow::AccessNode>(new data_flow::AccessNode(
                     this->element_counter_, debug_info, vertex, block.dataflow(), data))});
    this->element_counter_++;
    return dynamic_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::Tasklet& StructuredSDFGBuilder::add_tasklet(
    structured_control_flow::Block& block, const data_flow::TaskletCode code,
    const std::pair<std::string, sdfg::types::Scalar>& output,
    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
    const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex, std::unique_ptr<data_flow::Tasklet>(new data_flow::Tasklet(
                     this->element_counter_, debug_info, vertex, block.dataflow(), code, output,
                     inputs, symbolic::__true__()))});
    this->element_counter_++;
    return dynamic_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& StructuredSDFGBuilder::add_memlet(
    structured_control_flow::Block& block, data_flow::DataFlowNode& src,
    const std::string& src_conn, data_flow::DataFlowNode& dst, const std::string& dst_conn,
    const data_flow::Subset& subset, const DebugInfo& debug_info) {
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, block.dataflow_->graph_);
    auto res = block.dataflow_->edges_.insert(
        {edge.first, std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
                         this->element_counter_, debug_info, edge.first, block.dataflow(), src,
                         src_conn, dst, dst_conn, subset))});
    this->element_counter_++;
    return dynamic_cast<data_flow::Memlet&>(*(res.first->second));
};

data_flow::LibraryNode& StructuredSDFGBuilder::add_library_node(
    structured_control_flow::Block& block, const data_flow::LibraryNodeType& call,
    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
    const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
    const bool has_side_effect, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex, std::unique_ptr<data_flow::LibraryNode>(new data_flow::LibraryNode(
                     this->element_counter_, debug_info, vertex, block.dataflow(), outputs, inputs,
                     call, has_side_effect))});
    this->element_counter_++;
    return dynamic_cast<data_flow::LibraryNode&>(*(res.first->second));
}

void StructuredSDFGBuilder::remove_memlet(structured_control_flow::Block& block,
                                          const data_flow::Memlet& edge) {
    auto& graph = block.dataflow();
    auto e = edge.edge();
    boost::remove_edge(e, graph.graph_);
    graph.edges_.erase(e);
};

void StructuredSDFGBuilder::remove_node(structured_control_flow::Block& block,
                                        const data_flow::DataFlowNode& node) {
    auto& graph = block.dataflow();
    auto v = node.vertex();
    boost::remove_vertex(v, graph.graph_);
    graph.nodes_.erase(v);
};

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block,
                                       const data_flow::Tasklet& node) {
    auto& graph = block.dataflow();
    auto vertex = node.vertex();

    std::unordered_set<const data_flow::DataFlowNode*> to_delete = {&node};

    // Delete incoming
    std::list<const data_flow::Memlet*> iedges;
    for (auto& iedge : graph.in_edges(node)) {
        iedges.push_back(&iedge);
    }
    for (auto iedge : iedges) {
        auto& src = iedge->src();
        to_delete.insert(&src);

        auto edge = iedge->edge();
        graph.edges_.erase(edge);
        boost::remove_edge(edge, graph.graph_);
    }

    // Delete outgoing
    std::list<const data_flow::Memlet*> oedges;
    for (auto& oedge : graph.out_edges(node)) {
        oedges.push_back(&oedge);
    }
    for (auto oedge : oedges) {
        auto& dst = oedge->dst();
        to_delete.insert(&dst);

        auto edge = oedge->edge();
        graph.edges_.erase(edge);
        boost::remove_edge(edge, graph.graph_);
    }

    // Delete nodes
    for (auto obsolete_node : to_delete) {
        if (graph.in_degree(*obsolete_node) == 0 && graph.out_degree(*obsolete_node) == 0) {
            auto vertex = obsolete_node->vertex();
            graph.nodes_.erase(vertex);
            boost::remove_vertex(vertex, graph.graph_);
        }
    }
};

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block,
                                       const data_flow::AccessNode& node) {
    auto& graph = block.dataflow();
    auto vertex = node.vertex();
    assert(graph.out_degree(node) == 0);

    std::list<const data_flow::Memlet*> tmp;
    std::list<const data_flow::DataFlowNode*> queue = {&node};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (current != &node) {
            if (dynamic_cast<const data_flow::AccessNode*>(current)) {
                if (graph.in_degree(*current) > 0 || graph.out_degree(*current) > 0) {
                    continue;
                }
            }
        }

        tmp.clear();
        for (auto& iedge : graph.in_edges(*current)) {
            tmp.push_back(&iedge);
        }
        for (auto iedge : tmp) {
            auto& src = iedge->src();
            queue.push_back(&src);

            auto edge = iedge->edge();
            graph.edges_.erase(edge);
            boost::remove_edge(edge, graph.graph_);
        }

        auto vertex = current->vertex();
        graph.nodes_.erase(vertex);
        boost::remove_vertex(vertex, graph.graph_);
    }
};

data_flow::AccessNode& StructuredSDFGBuilder::symbolic_expression_to_dataflow(
    structured_control_flow::Block& parent, const symbolic::Expression& expr) {
    auto& sdfg = this->subject();

    codegen::CPPLanguageExtension language_extension;

    // Base cases
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);

        // Determine type
        types::Scalar sym_type = types::Scalar(types::PrimitiveType::Void);
        if (symbolic::is_nvptx(sym)) {
            sym_type = types::Scalar(types::PrimitiveType::Int32);
        } else {
            sym_type = static_cast<const types::Scalar&>(sdfg.type(sym->get_name()));
        }

        // Add new container for intermediate result
        auto tmp = this->find_new_name();
        this->add_container(tmp, sym_type);

        // Create dataflow graph
        auto& input_node = this->add_access(parent, sym->get_name());
        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet = this->add_tasklet(parent, data_flow::TaskletCode::assign,
                                          {"_out", sym_type}, {{"_in", sym_type}});
        this->add_memlet(parent, input_node, "void", tasklet, "_in", {symbolic::integer(0)});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {symbolic::integer(0)});

        return output_node;
    } else if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Int64));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet = this->add_tasklet(
            parent, data_flow::TaskletCode::assign,
            {"_out", types::Scalar(types::PrimitiveType::Int64)},
            {{language_extension.expression(expr), types::Scalar(types::PrimitiveType::Int64)}});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::BooleanAtom>(*expr)) {
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Bool));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet = this->add_tasklet(
            parent, data_flow::TaskletCode::assign,
            {"_out", types::Scalar(types::PrimitiveType::Bool)},
            {{language_extension.expression(expr), types::Scalar(types::PrimitiveType::Bool)}});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::Or>(*expr)) {
        auto or_expr = SymEngine::rcp_static_cast<const SymEngine::Or>(expr);
        assert(or_expr->get_container().size() == 2);

        std::vector<data_flow::AccessNode*> input_nodes;
        std::vector<std::pair<std::string, types::Scalar>> input_types;
        for (auto& arg : or_expr->get_container()) {
            auto& input_node = symbolic_expression_to_dataflow(parent, arg);
            input_nodes.push_back(&input_node);
            input_types.push_back(
                {"_in" + std::to_string(input_types.size() + 1),
                 static_cast<const types::Scalar&>(sdfg.type(input_node.data()))});
        }

        // Add new container for intermediate result
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Bool));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet =
            this->add_tasklet(parent, data_flow::TaskletCode::logical_or,
                              {"_out", types::Scalar(types::PrimitiveType::Bool)}, input_types);
        for (size_t i = 0; i < input_nodes.size(); i++) {
            this->add_memlet(parent, *input_nodes.at(i), "void", tasklet, input_types.at(i).first,
                             {symbolic::integer(0)});
        }
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::And>(*expr)) {
        auto and_expr = SymEngine::rcp_static_cast<const SymEngine::And>(expr);
        assert(and_expr->get_container().size() == 2);

        std::vector<data_flow::AccessNode*> input_nodes;
        std::vector<std::pair<std::string, types::Scalar>> input_types;
        for (auto& arg : and_expr->get_container()) {
            auto& input_node = symbolic_expression_to_dataflow(parent, arg);
            input_nodes.push_back(&input_node);
            input_types.push_back(
                {"_in" + std::to_string(input_types.size() + 1),
                 static_cast<const types::Scalar&>(sdfg.type(input_node.data()))});
        }

        // Add new container for intermediate result
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Bool));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet =
            this->add_tasklet(parent, data_flow::TaskletCode::logical_and,
                              {"_out", types::Scalar(types::PrimitiveType::Bool)}, input_types);
        for (size_t i = 0; i < input_nodes.size(); i++) {
            this->add_memlet(parent, *input_nodes.at(i), "void", tasklet, input_types.at(i).first,
                             {symbolic::integer(0)});
        }
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
        return output_node;
    } else {
        throw std::runtime_error("Unsupported expression type");
    }
};

}  // namespace builder
}  // namespace sdfg

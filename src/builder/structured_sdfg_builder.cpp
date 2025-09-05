#include "sdfg/builder/structured_sdfg_builder.h"

#include <cstddef>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/utils.h"

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

std::unordered_set<const control_flow::State*> StructuredSDFGBuilder::
    determine_loop_nodes(const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const {
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

bool post_dominates(
    const State* pdom,
    const State* node,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree
) {
    if (pdom == node) {
        return true;
    }

    auto current = pdom_tree.at(node);
    while (current != nullptr) {
        if (current == pdom) {
            return true;
        }
        current = pdom_tree.at(current);
    }

    return false;
}

const control_flow::State* StructuredSDFGBuilder::find_end_of_if_else(
    const SDFG& sdfg,
    const State* current,
    std::vector<const InterstateEdge*>& out_edges,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree
) {
    // Best-effort approach: Check if post-dominator of current dominates all out edges
    auto pdom = pdom_tree.at(current);
    for (auto& edge : out_edges) {
        if (!post_dominates(pdom, &edge->dst(), pdom_tree)) {
            return nullptr;
        }
    }

    return pdom;
}

void StructuredSDFGBuilder::traverse(const SDFG& sdfg) {
    // Start of SDFGS
    Sequence& root = *structured_sdfg_->root_;
    const State* start_state = &sdfg.start_state();

    auto pdom_tree = sdfg.post_dominator_tree();

    std::unordered_set<const InterstateEdge*> breaks;
    std::unordered_set<const InterstateEdge*> continues;
    for (auto& edge : sdfg.back_edges()) {
        continues.insert(edge);
    }

    std::unordered_set<const control_flow::State*> visited;
    this->traverse_with_loop_detection(sdfg, root, start_state, nullptr, continues, breaks, pdom_tree, visited);
};

void StructuredSDFGBuilder::traverse_with_loop_detection(
    const SDFG& sdfg,
    Sequence& scope,
    const State* current,
    const State* end,
    const std::unordered_set<const InterstateEdge*>& continues,
    const std::unordered_set<const InterstateEdge*>& breaks,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
    std::unordered_set<const control_flow::State*>& visited
) {
    if (current == end) {
        return;
    }

    auto in_edges = sdfg.in_edges(*current);

    // Loop detection
    std::unordered_set<const InterstateEdge*> loop_edges;
    for (auto& iedge : in_edges) {
        if (continues.find(&iedge) != continues.end()) {
            loop_edges.insert(&iedge);
        }
    }
    if (!loop_edges.empty()) {
        // 1. Determine nodes of loop body
        std::unordered_set<const control_flow::State*> body;
        for (auto back_edge : loop_edges) {
            auto loop_nodes = this->determine_loop_nodes(sdfg, back_edge->src(), back_edge->dst());
            body.insert(loop_nodes.begin(), loop_nodes.end());
        }

        // 2. Determine exit states and exit edges
        std::unordered_set<const control_flow::State*> exit_states;
        std::unordered_set<const control_flow::InterstateEdge*> exit_edges;
        for (auto node : body) {
            for (auto& edge : sdfg.out_edges(*node)) {
                if (body.find(&edge.dst()) == body.end()) {
                    exit_edges.insert(&edge);
                    exit_states.insert(&edge.dst());
                }
            }
        }
        if (exit_states.size() != 1) {
            throw UnstructuredControlFlowException();
        }
        const control_flow::State* exit_state = *exit_states.begin();

        for (auto& edge : breaks) {
            exit_edges.insert(edge);
        }

        // Collect debug information (could be removed when this is computed dynamically)
        DebugInfo dbg_info = current->debug_info();
        for (auto& edge : in_edges) {
            dbg_info = DebugInfo::merge(dbg_info, edge.debug_info());
        }
        for (auto node : body) {
            dbg_info = DebugInfo::merge(dbg_info, node->debug_info());
        }
        for (auto edge : exit_edges) {
            dbg_info = DebugInfo::merge(dbg_info, edge->debug_info());
        }

        // 3. Add while loop
        While& loop = this->add_while(scope, {}, dbg_info);

        std::unordered_set<const control_flow::State*> loop_visited(visited);
        this->traverse_without_loop_detection(
            sdfg, loop.root(), current, exit_state, continues, exit_edges, pdom_tree, loop_visited
        );

        this->traverse_with_loop_detection(sdfg, scope, exit_state, end, continues, breaks, pdom_tree, visited);
    } else {
        this->traverse_without_loop_detection(sdfg, scope, current, end, continues, breaks, pdom_tree, visited);
    }
};

void StructuredSDFGBuilder::traverse_without_loop_detection(
    const SDFG& sdfg,
    Sequence& scope,
    const State* current,
    const State* end,
    const std::unordered_set<const InterstateEdge*>& continues,
    const std::unordered_set<const InterstateEdge*>& breaks,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
    std::unordered_set<const control_flow::State*>& visited
) {
    std::list<const State*> queue = {current};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();
        if (curr == end) {
            continue;
        }

        if (visited.find(curr) != visited.end()) {
            throw UnstructuredControlFlowException();
        }
        visited.insert(curr);

        auto out_edges = sdfg.out_edges(*curr);
        auto out_degree = sdfg.out_degree(*curr);

        // Case 1: Sink node
        if (out_degree == 0) {
            this->add_block(scope, curr->dataflow(), {}, curr->debug_info());
            this->add_return(scope, {}, curr->debug_info());
            continue;
        }

        // Case 2: Transition
        if (out_degree == 1) {
            auto& oedge = *out_edges.begin();
            if (!oedge.is_unconditional()) {
                throw UnstructuredControlFlowException();
            }
            this->add_block(scope, curr->dataflow(), oedge.assignments(), curr->debug_info());

            if (continues.find(&oedge) != continues.end()) {
                this->add_continue(scope, {}, oedge.debug_info());
            } else if (breaks.find(&oedge) != breaks.end()) {
                this->add_break(scope, {}, oedge.debug_info());
            } else {
                bool starts_loop = false;
                for (auto& iedge : sdfg.in_edges(oedge.dst())) {
                    if (continues.find(&iedge) != continues.end()) {
                        starts_loop = true;
                        break;
                    }
                }
                if (!starts_loop) {
                    queue.push_back(&oedge.dst());
                } else {
                    this->traverse_with_loop_detection(
                        sdfg, scope, &oedge.dst(), end, continues, breaks, pdom_tree, visited
                    );
                }
            }
            continue;
        }

        // Case 3: Branches
        if (out_degree > 1) {
            this->add_block(scope, curr->dataflow(), {}, curr->debug_info());

            std::vector<const InterstateEdge*> out_edges_vec;
            for (auto& edge : out_edges) {
                out_edges_vec.push_back(&edge);
            }

            // Best-effort approach: Find end of if-else
            // If not found, the branches may repeat paths yielding a large SDFG
            const control_flow::State* local_end = this->find_end_of_if_else(sdfg, curr, out_edges_vec, pdom_tree);
            if (local_end == nullptr) {
                local_end = end;
            }

            auto& if_else = this->add_if_else(scope, {}, curr->debug_info());
            for (size_t i = 0; i < out_degree; i++) {
                auto& out_edge = out_edges_vec[i];

                auto& branch = this->add_case(if_else, out_edge->condition(), out_edge->debug_info());
                if (!out_edge->assignments().empty()) {
                    this->add_block(branch, out_edge->assignments(), out_edge->debug_info());
                }
                if (continues.find(out_edge) != continues.end()) {
                    this->add_continue(branch, {}, out_edge->debug_info());
                } else if (breaks.find(out_edge) != breaks.end()) {
                    this->add_break(branch, {}, out_edge->debug_info());
                } else {
                    std::unordered_set<const control_flow::State*> branch_visited(visited);
                    this->traverse_with_loop_detection(
                        sdfg, branch, &out_edge->dst(), local_end, continues, breaks, pdom_tree, branch_visited
                    );
                }
            }

            if (local_end != end) {
                bool starts_loop = false;
                for (auto& iedge : sdfg.in_edges(*local_end)) {
                    if (continues.find(&iedge) != continues.end()) {
                        starts_loop = true;
                        break;
                    }
                }
                if (!starts_loop) {
                    queue.push_back(local_end);
                } else {
                    this->traverse_with_loop_detection(sdfg, scope, local_end, end, continues, breaks, pdom_tree, visited);
                }
            }
            continue;
        }
    }
}

Function& StructuredSDFGBuilder::function() const { return static_cast<Function&>(*this->structured_sdfg_); };

StructuredSDFGBuilder::StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg)
    : FunctionBuilder(), structured_sdfg_(std::move(sdfg)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(const std::string& name, FunctionType type)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(name, type)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(const SDFG& sdfg)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(sdfg.name(), sdfg.type())) {
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
        this->structured_sdfg_->externals_linkage_types_[ext] = sdfg.linkage_type(ext);
    }

    for (auto& entry : sdfg.assumptions_) {
        this->structured_sdfg_->assumptions_.insert({entry.first, entry.second});
    }

    for (auto& entry : sdfg.metadata_) {
        this->structured_sdfg_->metadata_[entry.first] = entry.second;
    }

    this->traverse(sdfg);
};

StructuredSDFG& StructuredSDFGBuilder::subject() const { return *this->structured_sdfg_; };

std::unique_ptr<StructuredSDFG> StructuredSDFGBuilder::move() {
#ifndef NDEBUG
    this->structured_sdfg_->validate();
#endif

    return std::move(this->structured_sdfg_);
};

Element* StructuredSDFGBuilder::find_element_by_id(const size_t& element_id) const {
    auto& sdfg = this->subject();
    std::list<Element*> queue = {&sdfg.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (current->element_id() == element_id) {
            return current;
        }

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(current)) {
            auto& dataflow = block_stmt->dataflow();
            for (auto& node : dataflow.nodes()) {
                queue.push_back(&node);
            }
            for (auto& edge : dataflow.edges()) {
                queue.push_back(&edge);
            }
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
                queue.push_back(&sequence_stmt->at(i).second);
            }
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            // Do nothing
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            // Do nothing
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            // Do nothing
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(current)) {
            queue.push_back(&map_stmt->root());
        }
    }

    return nullptr;
};

Sequence& StructuredSDFGBuilder::
    add_sequence(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Sequence&>(*parent.children_.back().get());
};

Sequence& StructuredSDFGBuilder::add_sequence_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index, std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Sequence&>(*parent.children_.at(index).get());
};

Sequence& StructuredSDFGBuilder::add_sequence_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Sequence&>(*parent.children_.at(index + 1).get());
};

std::pair<Sequence&, Transition&> StructuredSDFGBuilder::
    add_sequence_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index, std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Sequence&>(new_entry.first);

    return {new_block, new_entry.second};
};

void StructuredSDFGBuilder::remove_child(Sequence& parent, size_t index) {
    parent.children_.erase(parent.children_.begin() + index);
    parent.transitions_.erase(parent.transitions_.begin() + index);
};

void StructuredSDFGBuilder::remove_children(Sequence& parent) {
    parent.children_.clear();
    parent.transitions_.clear();
};

void StructuredSDFGBuilder::move_child(Sequence& source, size_t source_index, Sequence& target) {
    this->move_child(source, source_index, target, target.size());
};

void StructuredSDFGBuilder::move_child(Sequence& source, size_t source_index, Sequence& target, size_t target_index) {
    auto node_ptr = std::move(source.children_.at(source_index));
    auto trans_ptr = std::move(source.transitions_.at(source_index));
    source.children_.erase(source.children_.begin() + source_index);
    source.transitions_.erase(source.transitions_.begin() + source_index);

    trans_ptr->parent_ = &target;
    target.children_.insert(target.children_.begin() + target_index, std::move(node_ptr));
    target.transitions_.insert(target.transitions_.begin() + target_index, std::move(trans_ptr));
};

void StructuredSDFGBuilder::move_children(Sequence& source, Sequence& target) {
    this->move_children(source, target, target.size());
};

void StructuredSDFGBuilder::move_children(Sequence& source, Sequence& target, size_t target_index) {
    target.children_.insert(
        target.children_.begin() + target_index,
        std::make_move_iterator(source.children_.begin()),
        std::make_move_iterator(source.children_.end())
    );
    target.transitions_.insert(
        target.transitions_.begin() + target_index,
        std::make_move_iterator(source.transitions_.begin()),
        std::make_move_iterator(source.transitions_.end())
    );
    for (auto& trans : target.transitions_) {
        trans->parent_ = &target;
    }
    source.children_.clear();
    source.transitions_.clear();
};

Block& StructuredSDFGBuilder::
    add_block(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.back());
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block(
    Sequence& parent,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_.push_back(std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.back());
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index));
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_before(
    Sequence& parent,
    ControlFlowNode& child,
    data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index));
    (*new_block.dataflow_).parent_ = &new_block;
    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index + 1));
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_after(
    Sequence& parent,
    ControlFlowNode& child,
    data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index + 1));
    (*new_block.dataflow_).parent_ = &new_block;
    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::
    add_block_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = static_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_before(
    Sequence& parent, ControlFlowNode& child, data_flow::DataFlowGraph& data_flow_graph, const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::
    add_block_after(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_after(
    Sequence& parent, ControlFlowNode& child, data_flow::DataFlowGraph& data_flow_graph, const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return {new_block, new_entry.second};
};

IfElse& StructuredSDFGBuilder::
    add_if_else(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<IfElse&>(*parent.children_.back().get());
};

IfElse& StructuredSDFGBuilder::add_if_else_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<IfElse&>(*parent.children_.at(index));
};

IfElse& StructuredSDFGBuilder::add_if_else_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<IfElse&>(*parent.children_.at(index + 1));
};

std::pair<IfElse&, Transition&> StructuredSDFGBuilder::
    add_if_else_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::IfElse&>(new_entry.first);

    return {new_block, new_entry.second};
};

Sequence& StructuredSDFGBuilder::add_case(IfElse& scope, const sdfg::symbolic::Condition cond, const DebugInfo& debug_info) {
    scope.cases_.push_back(std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    scope.conditions_.push_back(cond);
    return *scope.cases_.back();
};

void StructuredSDFGBuilder::remove_case(IfElse& scope, size_t index, const DebugInfo& debug_info) {
    scope.cases_.erase(scope.cases_.begin() + index);
    scope.conditions_.erase(scope.conditions_.begin() + index);
};

While& StructuredSDFGBuilder::
    add_while(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<While>(new While(this->new_element_id(), debug_info)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<While&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::add_for(
    Sequence& parent,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_
        .push_back(std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<For&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::add_for_before(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<For&>(*parent.children_.at(index).get());
};

For& StructuredSDFGBuilder::add_for_after(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<For&>(*parent.children_.at(index + 1).get());
};

Map& StructuredSDFGBuilder::add_map(
    Sequence& parent,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_
        .push_back(std::unique_ptr<
                   Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Map&>(*parent.children_.back().get());
};

Map& StructuredSDFGBuilder::add_map_before(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)
        )
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Map&>(*parent.children_.at(index).get());
};

Map& StructuredSDFGBuilder::add_map_after(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)
        )
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Map&>(*parent.children_.at(index + 1).get());
};

Continue& StructuredSDFGBuilder::
    add_continue(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    // Check if continue is in a loop
    analysis::AnalysisManager analysis_manager(this->subject());
    auto& scope_tree_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto current_scope = scope_tree_analysis.parent_scope(&parent);
    bool in_loop = false;
    while (current_scope != nullptr) {
        if (dynamic_cast<structured_control_flow::While*>(current_scope)) {
            in_loop = true;
            break;
        } else if (dynamic_cast<structured_control_flow::For*>(current_scope)) {
            throw UnstructuredControlFlowException();
        }
        current_scope = scope_tree_analysis.parent_scope(current_scope);
    }
    if (!in_loop) {
        throw UnstructuredControlFlowException();
    }

    parent.children_.push_back(std::unique_ptr<Continue>(new Continue(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Continue&>(*parent.children_.back().get());
};

Break& StructuredSDFGBuilder::
    add_break(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    // Check if break is in a loop
    analysis::AnalysisManager analysis_manager(this->subject());
    auto& scope_tree_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto current_scope = scope_tree_analysis.parent_scope(&parent);
    bool in_loop = false;
    while (current_scope != nullptr) {
        if (dynamic_cast<structured_control_flow::While*>(current_scope)) {
            in_loop = true;
            break;
        } else if (dynamic_cast<structured_control_flow::For*>(current_scope)) {
            throw UnstructuredControlFlowException();
        }
        current_scope = scope_tree_analysis.parent_scope(current_scope);
    }
    if (!in_loop) {
        throw UnstructuredControlFlowException();
    }

    parent.children_.push_back(std::unique_ptr<Break>(new Break(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Break&>(*parent.children_.back().get());
};

Return& StructuredSDFGBuilder::
    add_return(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Return>(new Return(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Return&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::convert_while(
    Sequence& parent,
    While& loop,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update
) {
    int index = parent.index(loop);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    auto iter = parent.children_.begin() + index;
    auto& new_iter = *parent.children_.insert(
        iter + 1,
        std::unique_ptr<For>(new For(this->new_element_id(), loop.debug_info(), indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    auto& for_loop = dynamic_cast<For&>(*new_iter);
    this->move_children(loop.root(), for_loop.root());

    // Remove while loop
    parent.children_.erase(parent.children_.begin() + index);

    return for_loop;
};

Map& StructuredSDFGBuilder::convert_for(Sequence& parent, For& loop) {
    int index = parent.index(loop);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    auto iter = parent.children_.begin() + index;
    auto& new_iter = *parent.children_.insert(
        iter + 1,
        std::unique_ptr<Map>(new Map(
            this->new_element_id(),
            loop.debug_info(),
            loop.indvar(),
            loop.init(),
            loop.update(),
            loop.condition(),
            ScheduleType_Sequential::create()
        ))
    );

    // Increment element id for body node
    this->new_element_id();

    auto& map = dynamic_cast<Map&>(*new_iter);
    this->move_children(loop.root(), map.root());

    // Remove for loop
    parent.children_.erase(parent.children_.begin() + index);

    return map;
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
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop_stmt->root());
        }
    }

    return this->structured_sdfg_->root();
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& StructuredSDFGBuilder::
    add_access(structured_control_flow::Block& block, const std::string& data, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::AccessNode>(
             new data_flow::AccessNode(this->new_element_id(), debug_info, vertex, block.dataflow(), data)
         )}
    );

    return dynamic_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::ConstantNode& StructuredSDFGBuilder::add_constant(
    structured_control_flow::Block& block, const std::string& data, const types::IType& type, const DebugInfo& debug_info
) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::ConstantNode>(
             new data_flow::ConstantNode(this->new_element_id(), debug_info, vertex, block.dataflow(), data, type)
         )}
    );

    return dynamic_cast<data_flow::ConstantNode&>(*(res.first->second));
};


data_flow::Tasklet& StructuredSDFGBuilder::add_tasklet(
    structured_control_flow::Block& block,
    const data_flow::TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs,
    const DebugInfo& debug_info
) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::Tasklet>(
             new data_flow::Tasklet(this->new_element_id(), debug_info, vertex, block.dataflow(), code, output, inputs)
         )}
    );

    return dynamic_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& StructuredSDFGBuilder::add_memlet(
    structured_control_flow::Block& block,
    data_flow::DataFlowNode& src,
    const std::string& src_conn,
    data_flow::DataFlowNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, block.dataflow_->graph_);
    auto res = block.dataflow_->edges_.insert(
        {edge.first,
         std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
             this->new_element_id(), debug_info, edge.first, block.dataflow(), src, src_conn, dst, dst_conn, subset, base_type
         ))}
    );

    auto& memlet = dynamic_cast<data_flow::Memlet&>(*(res.first->second));
#ifndef NDEBUG
    memlet.validate(*this->structured_sdfg_);
#endif

    return memlet;
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    const types::IType* src_type = nullptr;
    if (auto cnode = dynamic_cast<data_flow::ConstantNode*>(&src)) {
        src_type = &cnode->type();
    } else {
        src_type = &this->structured_sdfg_->type(src.data());
    }
    auto& base_type = types::infer_type(*this->structured_sdfg_, *src_type, subset);
    if (base_type.type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, *src_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    auto& dst_type = this->structured_sdfg_->type(dst.data());
    auto& base_type = types::infer_type(*this->structured_sdfg_, dst_type, subset);
    if (base_type.type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(block, src, src_conn, dst, "void", subset, dst_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::LibraryNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::LibraryNode& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_reference_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, "ref", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_dereference_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    bool derefs_src,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    if (derefs_src) {
        return this->add_memlet(block, src, "void", dst, "deref", {symbolic::zero()}, base_type, debug_info);
    } else {
        return this->add_memlet(block, src, "deref", dst, "void", {symbolic::zero()}, base_type, debug_info);
    }
};

void StructuredSDFGBuilder::remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge) {
    auto& graph = block.dataflow();
    auto e = edge.edge();
    boost::remove_edge(e, graph.graph_);
    graph.edges_.erase(e);
};

void StructuredSDFGBuilder::remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node) {
    auto& graph = block.dataflow();
    auto v = node.vertex();
    boost::remove_vertex(v, graph.graph_);
    graph.nodes_.erase(v);
};

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block, const data_flow::CodeNode& node) {
    auto& graph = block.dataflow();

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

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node) {
    auto& graph = block.dataflow();
    if (graph.out_degree(node) != 0) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Access node has outgoing edges");
    }

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

void StructuredSDFGBuilder::add_dataflow(const data_flow::DataFlowGraph& from, Block& to) {
    auto& to_dataflow = to.dataflow();

    std::unordered_map<graph::Vertex, graph::Vertex> node_mapping;
    for (auto& entry : from.nodes_) {
        auto vertex = boost::add_vertex(to_dataflow.graph_);
        to_dataflow.nodes_.insert({vertex, entry.second->clone(this->new_element_id(), vertex, to_dataflow)});
        node_mapping.insert({entry.first, vertex});
    }

    for (auto& entry : from.edges_) {
        auto src = node_mapping[entry.second->src().vertex()];
        auto dst = node_mapping[entry.second->dst().vertex()];

        auto edge = boost::add_edge(src, dst, to_dataflow.graph_);

        to_dataflow.edges_.insert(
            {edge.first,
             entry.second->clone(
                 this->new_element_id(), edge.first, to_dataflow, *to_dataflow.nodes_[src], *to_dataflow.nodes_[dst]
             )}
        );
    }
};

} // namespace builder
} // namespace sdfg

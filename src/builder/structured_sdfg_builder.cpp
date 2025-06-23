#include "sdfg/builder/structured_sdfg_builder.h"

#include <cstddef>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/types/utils.h"

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

bool post_dominates(
    const State* pdom, const State* node,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree) {
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
    const SDFG& sdfg, const State* current, std::vector<const InterstateEdge*>& out_edges,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree) {
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
    this->traverse_with_loop_detection(sdfg, root, start_state, nullptr, continues, breaks,
                                       pdom_tree, visited);
};

void StructuredSDFGBuilder::traverse_with_loop_detection(
    const SDFG& sdfg, Sequence& scope, const State* current, const State* end,
    const std::unordered_set<const InterstateEdge*>& continues,
    const std::unordered_set<const InterstateEdge*>& breaks,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
    std::unordered_set<const control_flow::State*>& visited) {
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
        this->traverse_without_loop_detection(sdfg, loop.root(), current, exit_state, continues,
                                              exit_edges, pdom_tree, loop_visited);

        this->traverse_with_loop_detection(sdfg, scope, exit_state, end, continues, breaks,
                                           pdom_tree, visited);
    } else {
        this->traverse_without_loop_detection(sdfg, scope, current, end, continues, breaks,
                                              pdom_tree, visited);
    }
};

void StructuredSDFGBuilder::traverse_without_loop_detection(
    const SDFG& sdfg, Sequence& scope, const State* current, const State* end,
    const std::unordered_set<const InterstateEdge*>& continues,
    const std::unordered_set<const InterstateEdge*>& breaks,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
    std::unordered_set<const control_flow::State*>& visited) {
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
                this->add_continue(scope, oedge.debug_info());
            } else if (breaks.find(&oedge) != breaks.end()) {
                this->add_break(scope, oedge.debug_info());
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
                    this->traverse_with_loop_detection(sdfg, scope, &oedge.dst(), end, continues,
                                                       breaks, pdom_tree, visited);
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
            const control_flow::State* local_end =
                this->find_end_of_if_else(sdfg, curr, out_edges_vec, pdom_tree);
            if (local_end == nullptr) {
                local_end = end;
            }

            auto& if_else = this->add_if_else(scope, curr->debug_info());
            for (size_t i = 0; i < out_degree; i++) {
                auto& out_edge = out_edges_vec[i];

                auto& branch =
                    this->add_case(if_else, out_edge->condition(), out_edge->debug_info());
                if (!out_edge->assignments().empty()) {
                    this->add_block(branch, out_edge->assignments(), out_edge->debug_info());
                }
                if (continues.find(out_edge) != continues.end()) {
                    this->add_continue(branch, out_edge->debug_info());
                } else if (breaks.find(out_edge) != breaks.end()) {
                    this->add_break(branch, out_edge->debug_info());
                } else {
                    std::unordered_set<const control_flow::State*> branch_visited(visited);
                    this->traverse_with_loop_detection(sdfg, branch, &out_edge->dst(), local_end,
                                                       continues, breaks, pdom_tree,
                                                       branch_visited);
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
                    this->traverse_with_loop_detection(sdfg, scope, local_end, end, continues,
                                                       breaks, pdom_tree, visited);
                }
            }
            continue;
        }
    }
}

Function& StructuredSDFGBuilder::function() const {
    return static_cast<Function&>(*this->structured_sdfg_);
};

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

Sequence& StructuredSDFGBuilder::add_sequence(Sequence& parent,
                                              const sdfg::control_flow::Assignments& assignments,
                                              const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

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
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Block not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

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
                                        const sdfg::control_flow::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    auto& new_block = dynamic_cast<structured_control_flow::Block&>(*parent.children_.back().get());
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block(Sequence& parent,
                                        const data_flow::DataFlowGraph& data_flow_graph,
                                        const sdfg::control_flow::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    auto& new_block = dynamic_cast<structured_control_flow::Block&>(*parent.children_.back().get());
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
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
                            std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

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

    parent.children_.insert(parent.children_.begin() + index,
                            std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

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
                            std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

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

    parent.children_.insert(parent.children_.begin() + index + 1,
                            std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return {new_block, new_entry.second};
};

For& StructuredSDFGBuilder::add_for(Sequence& parent, const symbolic::Symbol& indvar,
                                    const symbolic::Condition& condition,
                                    const symbolic::Expression& init,
                                    const symbolic::Expression& update,
                                    const sdfg::control_flow::Assignments& assignments,
                                    const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<For>(
        new For(this->new_element_id(), debug_info, indvar, init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

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
                            std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar,
                                                         init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

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
                            std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar,
                                                         init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::For&>(new_entry.first);

    return {new_block, new_entry.second};
};

IfElse& StructuredSDFGBuilder::add_if_else(Sequence& parent, const DebugInfo& debug_info) {
    return this->add_if_else(parent, control_flow::Assignments{}, debug_info);
};

IfElse& StructuredSDFGBuilder::add_if_else(Sequence& parent,
                                           const sdfg::control_flow::Assignments& assignments,
                                           const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

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
        std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::IfElse&>(new_entry.first);

    return {new_block, new_entry.second};
};

Sequence& StructuredSDFGBuilder::add_case(IfElse& scope, const sdfg::symbolic::Condition cond,
                                          const DebugInfo& debug_info) {
    scope.cases_.push_back(
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    scope.conditions_.push_back(cond);
    return *scope.cases_.back();
};

void StructuredSDFGBuilder::remove_case(IfElse& scope, size_t i, const DebugInfo& debug_info) {
    scope.cases_.erase(scope.cases_.begin() + i);
    scope.conditions_.erase(scope.conditions_.begin() + i);
};

While& StructuredSDFGBuilder::add_while(Sequence& parent,
                                        const sdfg::control_flow::Assignments& assignments,
                                        const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<While>(new While(this->new_element_id(), debug_info)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    return static_cast<While&>(*parent.children_.back().get());
};

Continue& StructuredSDFGBuilder::add_continue(Sequence& parent, const DebugInfo& debug_info) {
    return this->add_continue(parent, control_flow::Assignments{}, debug_info);
};

Continue& StructuredSDFGBuilder::add_continue(Sequence& parent,
                                              const sdfg::control_flow::Assignments& assignments,
                                              const DebugInfo& debug_info) {
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

    parent.children_.push_back(
        std::unique_ptr<Continue>(new Continue(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    return static_cast<Continue&>(*parent.children_.back().get());
};

Break& StructuredSDFGBuilder::add_break(Sequence& parent, const DebugInfo& debug_info) {
    return this->add_break(parent, control_flow::Assignments{}, debug_info);
};

Break& StructuredSDFGBuilder::add_break(Sequence& parent,
                                        const sdfg::control_flow::Assignments& assignments,
                                        const DebugInfo& debug_info) {
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

    parent.children_.push_back(
        std::unique_ptr<Break>(new Break(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    return static_cast<Break&>(*parent.children_.back().get());
};

Return& StructuredSDFGBuilder::add_return(Sequence& parent,
                                          const sdfg::control_flow::Assignments& assignments,
                                          const DebugInfo& debug_info) {
    parent.children_.push_back(
        std::unique_ptr<Return>(new Return(this->new_element_id(), debug_info)));

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    return static_cast<Return&>(*parent.children_.back().get());
};

Map& StructuredSDFGBuilder::add_map(Sequence& parent, const symbolic::Symbol& indvar,
                                    const symbolic::Condition& condition,
                                    const symbolic::Expression& init,
                                    const symbolic::Expression& update,
                                    const ScheduleType& schedule_type,
                                    const sdfg::control_flow::Assignments& assignments,
                                    const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Map>(new Map(
        this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), debug_info, parent, assignments)));

    return static_cast<Map&>(*parent.children_.back().get());
};

std::pair<Map&, Transition&> StructuredSDFGBuilder::add_map_before(
    Sequence& parent, ControlFlowNode& block, const symbolic::Symbol& indvar,
    const symbolic::Condition& condition, const symbolic::Expression& init,
    const symbolic::Expression& update, const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
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
                            std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar,
                                                         init, update, condition, schedule_type)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Map&>(new_entry.first);

    return {new_block, new_entry.second};
};

std::pair<Map&, Transition&> StructuredSDFGBuilder::add_map_after(
    Sequence& parent, ControlFlowNode& block, const symbolic::Symbol& indvar,
    const symbolic::Condition& condition, const symbolic::Expression& init,
    const symbolic::Expression& update, const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
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
                            std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar,
                                                         init, update, condition, schedule_type)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent)));

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Map&>(new_entry.first);

    return {new_block, new_entry.second};
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
        iter + 1, std::unique_ptr<For>(new For(this->new_element_id(), loop.debug_info(), indvar,
                                               init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    auto& for_loop = dynamic_cast<For&>(*new_iter);
    this->insert_children(for_loop.root(), loop.root(), 0);

    // Remove while loop
    parent.children_.erase(parent.children_.begin() + index);

    return for_loop;
};

Map& StructuredSDFGBuilder::convert_for(Sequence& parent, For& loop) {
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
        iter + 1, std::unique_ptr<Map>(new Map(this->new_element_id(), loop.debug_info(),
                                               loop.indvar(), loop.init(), loop.update(),
                                               loop.condition(), ScheduleType_Sequential)));

    // Increment element id for body node
    this->new_element_id();

    auto& map = dynamic_cast<Map&>(*new_iter);
    this->insert_children(map.root(), loop.root(), 0);

    // Remove for loop
    parent.children_.erase(parent.children_.begin() + index);

    return map;
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
        }
    }

    return this->structured_sdfg_->root();
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& StructuredSDFGBuilder::add_access(structured_control_flow::Block& block,
                                                         const std::string& data,
                                                         const DebugInfo& debug_info) {
    // Check: Data exists
    if (!this->subject().exists(data)) {
        throw InvalidSDFGException("Data does not exist in SDFG: " + data);
    }

    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex, std::unique_ptr<data_flow::AccessNode>(new data_flow::AccessNode(
                     this->new_element_id(), debug_info, vertex, block.dataflow(), data))});

    return dynamic_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::Tasklet& StructuredSDFGBuilder::add_tasklet(
    structured_control_flow::Block& block, const data_flow::TaskletCode code,
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

    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex, std::unique_ptr<data_flow::Tasklet>(new data_flow::Tasklet(
                     this->new_element_id(), debug_info, vertex, block.dataflow(), code, output,
                     inputs, symbolic::__true__()))});

    return dynamic_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& StructuredSDFGBuilder::add_memlet(
    structured_control_flow::Block& block, data_flow::DataFlowNode& src,
    const std::string& src_conn, data_flow::DataFlowNode& dst, const std::string& dst_conn,
    const data_flow::Subset& subset, const DebugInfo& debug_info) {
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
    } else {
        throw InvalidSDFGException("Invalid src or dst node type");
    }

    auto edge = boost::add_edge(src.vertex_, dst.vertex_, block.dataflow_->graph_);
    auto res = block.dataflow_->edges_.insert(
        {edge.first, std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
                         this->new_element_id(), debug_info, edge.first, block.dataflow(), src,
                         src_conn, dst, dst_conn, subset))});

    return dynamic_cast<data_flow::Memlet&>(*(res.first->second));
};

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

data_flow::AccessNode& StructuredSDFGBuilder::symbolic_expression_to_dataflow(
    structured_control_flow::Block& parent, const symbolic::Expression& expr) {
    auto& sdfg = this->subject();

    codegen::CPPLanguageExtension language_extension;

    // Base cases
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);

        // Determine type
        types::Scalar sym_type = types::Scalar(types::PrimitiveType::Void);
        if (symbolic::is_nv(sym)) {
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
        this->add_memlet(parent, input_node, "void", tasklet, "_in", {});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {});

        return output_node;
    } else if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Int64));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet = this->add_tasklet(
            parent, data_flow::TaskletCode::assign,
            {"_out", types::Scalar(types::PrimitiveType::Int64)},
            {{language_extension.expression(expr), types::Scalar(types::PrimitiveType::Int64)}});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::BooleanAtom>(*expr)) {
        auto tmp = this->find_new_name();
        this->add_container(tmp, types::Scalar(types::PrimitiveType::Bool));

        auto& output_node = this->add_access(parent, tmp);
        auto& tasklet = this->add_tasklet(
            parent, data_flow::TaskletCode::assign,
            {"_out", types::Scalar(types::PrimitiveType::Bool)},
            {{language_extension.expression(expr), types::Scalar(types::PrimitiveType::Bool)}});
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::Or>(*expr)) {
        auto or_expr = SymEngine::rcp_static_cast<const SymEngine::Or>(expr);
        if (or_expr->get_container().size() != 2) {
            throw InvalidSDFGException(
                "StructuredSDFGBuilder: Or expression must have exactly two arguments");
        }

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
                             {});
        }
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {});
        return output_node;
    } else if (SymEngine::is_a<SymEngine::And>(*expr)) {
        auto and_expr = SymEngine::rcp_static_cast<const SymEngine::And>(expr);
        if (and_expr->get_container().size() != 2) {
            throw InvalidSDFGException(
                "StructuredSDFGBuilder: And expression must have exactly two arguments");
        }

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
                             {});
        }
        this->add_memlet(parent, tasklet, "_out", output_node, "void", {});
        return output_node;
    } else {
        throw std::runtime_error("Unsupported expression type");
    }
};

void StructuredSDFGBuilder::add_dataflow(const data_flow::DataFlowGraph& from, Block& to) {
    auto& to_dataflow = to.dataflow();

    std::unordered_map<graph::Vertex, graph::Vertex> node_mapping;
    for (auto& entry : from.nodes_) {
        auto vertex = boost::add_vertex(to_dataflow.graph_);
        to_dataflow.nodes_.insert(
            {vertex, entry.second->clone(this->new_element_id(), vertex, to_dataflow)});
        node_mapping.insert({entry.first, vertex});
    }

    for (auto& entry : from.edges_) {
        auto src = node_mapping[entry.second->src().vertex()];
        auto dst = node_mapping[entry.second->dst().vertex()];

        auto edge = boost::add_edge(src, dst, to_dataflow.graph_);

        to_dataflow.edges_.insert(
            {edge.first, entry.second->clone(this->new_element_id(), edge.first, to_dataflow,
                                             *to_dataflow.nodes_[src], *to_dataflow.nodes_[dst])});
    }
};

}  // namespace builder
}  // namespace sdfg

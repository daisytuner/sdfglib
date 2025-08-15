#include "sdfg/analysis/users.h"

#include <cassert>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/sets.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

User::User(graph::Vertex vertex, const std::string& container, Element* element, Use use)
    : vertex_(vertex), container_(container), use_(use), element_(element) {

      };

User::User(graph::Vertex vertex, const std::string& container, Element* element, data_flow::DataFlowGraph* parent, Use use)
    : vertex_(vertex), container_(container), use_(use), element_(element), parent_(parent) {

      };

User::~User() {

};

Use User::use() const { return this->use_; };

std::string& User::container() { return this->container_; };

Element* User::element() { return this->element_; };

data_flow::DataFlowGraph* User::parent() { return this->parent_; };

const std::vector<data_flow::Subset> User::subsets() const {
    if (this->container_ == "") {
        return {};
    }

    if (auto access_node = dynamic_cast<data_flow::AccessNode*>(this->element_)) {
        auto& graph = *this->parent_;
        if (this->use_ == Use::READ || this->use_ == Use::VIEW) {
            std::vector<data_flow::Subset> subsets;
            for (auto& iedge : graph.out_edges(*access_node)) {
                subsets.push_back(iedge.subset());
            }
            return subsets;
        } else if (this->use_ == Use::WRITE || this->use_ == Use::MOVE) {
            std::vector<data_flow::Subset> subsets;
            for (auto& oedge : graph.in_edges(*access_node)) {
                subsets.push_back(oedge.subset());
            }
            return subsets;
        }
    }

    // Use of symbol
    return {{}};
};

ForUser::ForUser(
    graph::Vertex vertex,
    const std::string& container,
    Element* element,
    Use use,
    bool is_init,
    bool is_condition,
    bool is_update
)
    : User(vertex, container, element, use), is_init_(is_init), is_condition_(is_condition), is_update_(is_update) {

      };

bool ForUser::is_init() const { return this->is_init_; };

bool ForUser::is_condition() const { return this->is_condition_; };

bool ForUser::is_update() const { return this->is_update_; };

void Users::init_dom_tree() {
    this->dom_tree_.clear();
    this->pdom_tree_.clear();

    // Compute dominator-tree
    auto dom_tree = graph::dominator_tree(this->graph_, this->source_->vertex_);
    for (auto& entry : dom_tree) {
        User* first = this->users_.at(entry.first).get();
        User* second = nullptr;
        if (entry.second != boost::graph_traits<graph::Graph>::null_vertex()) {
            second = this->users_.at(entry.second).get();
        }
        this->dom_tree_.insert({first, second});
    }

    // Compute post-dominator-tree
    auto pdom_tree = graph::post_dominator_tree(this->graph_, this->sink_->vertex_);
    for (auto& entry : pdom_tree) {
        User* first = this->users_.at(entry.first).get();
        User* second = nullptr;
        if (entry.second != boost::graph_traits<graph::Graph>::null_vertex()) {
            second = this->users_.at(entry.second).get();
        }
        this->pdom_tree_.insert({first, second});
    }
};

std::pair<graph::Vertex, graph::Vertex> Users::traverse(data_flow::DataFlowGraph& dataflow) {
    graph::Vertex first = boost::graph_traits<graph::Graph>::null_vertex();
    graph::Vertex last = boost::graph_traits<graph::Graph>::null_vertex();
    for (auto node : dataflow.topological_sort()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (!symbolic::is_pointer(symbolic::symbol(access_node->data()))) {
                if (dataflow.in_degree(*node) > 0) {
                    Use use = Use::WRITE;

                    // Check if the pointer itself is moved (overwritten)
                    for (auto& iedge : dataflow.in_edges(*access_node)) {
                        if (iedge.type() == data_flow::MemletType::Reference ||
                            iedge.type() == data_flow::MemletType::Dereference_Src) {
                            use = Use::MOVE;
                            break;
                        }
                    }

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, access_node->data(), access_node, &dataflow, use));

                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;

                    // Check if the pointer itself is viewed (aliased)
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.type() == data_flow::MemletType::Reference ||
                            oedge.type() == data_flow::MemletType::Dereference_Dst) {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, access_node->data(), access_node, &dataflow, use));

                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(node)) {
            if (tasklet->is_conditional()) {
                auto& condition = tasklet->condition();
                for (auto& atom : symbolic::atoms(condition)) {
                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, atom->get_name(), tasklet, &dataflow, Use::READ));
                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
            }
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(node)) {
            for (auto& symbol : library_node->symbols()) {
                auto v = boost::add_vertex(this->graph_);
                this->add_user(std::make_unique<User>(v, symbol->get_name(), library_node, &dataflow, Use::READ));
                if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                    boost::add_edge(last, v, this->graph_);
                } else {
                    first = v;
                }
                last = v;
            }
        }

        for (auto& oedge : dataflow.out_edges(*node)) {
            std::unordered_set<std::string> used;
            for (auto dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    if (used.find(atom->get_name()) != used.end()) {
                        continue;
                    }
                    used.insert(atom->get_name());

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, atom->get_name(), &oedge, &dataflow, Use::READ));
                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
            }
        }
    }

    return {first, last};
};

std::pair<graph::Vertex, graph::Vertex> Users::traverse(structured_control_flow::ControlFlowNode& node) {
    if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(&node)) {
        // NOP
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", block_stmt, Use::NOP)});
        this->entries_.insert({block_stmt, this->users_.at(s).get()});

        // NOP
        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", block_stmt, Use::NOP)});
        this->exits_.insert({block_stmt, this->users_.at(t).get()});

        auto& dataflow = block_stmt->dataflow();
        auto subgraph = this->traverse(dataflow);

        // May be empty
        if (subgraph.first == boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(s, t, this->graph_);
            return {s, t};
        }

        boost::add_edge(s, subgraph.first, this->graph_);
        boost::add_edge(subgraph.second, t, this->graph_);

        return {s, t};
    } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", sequence_stmt, Use::NOP)});
        this->entries_.insert({sequence_stmt, this->users_.at(s).get()});

        graph::Vertex current = s;
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            auto child = sequence_stmt->at(i);

            auto subgraph = this->traverse(child.first);
            boost::add_edge(current, subgraph.first, this->graph_);
            current = subgraph.second;

            std::unordered_set<std::string> used;
            for (auto& entry : child.second.assignments()) {
                for (auto atom : symbolic::atoms(entry.second)) {
                    if (symbolic::is_pointer(atom)) {
                        continue;
                    }
                    if (used.find(atom->get_name()) != used.end()) {
                        continue;
                    }
                    used.insert(atom->get_name());

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, atom->get_name(), &child.second, Use::READ));

                    boost::add_edge(current, v, this->graph_);
                    current = v;
                }
            }

            for (auto& entry : child.second.assignments()) {
                auto v = boost::add_vertex(this->graph_);
                this->add_user(std::make_unique<User>(v, entry.first->get_name(), &child.second, Use::WRITE));

                boost::add_edge(current, v, this->graph_);
                current = v;
            }
        }

        if (current == boost::graph_traits<graph::Graph>::null_vertex()) {
            this->exits_.insert({sequence_stmt, this->users_.at(s).get()});
            return {s, current};
        }

        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", sequence_stmt, Use::NOP)});
        boost::add_edge(current, t, this->graph_);
        this->exits_.insert({sequence_stmt, this->users_.at(t).get()});

        return {s, t};
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        // NOP
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", if_else_stmt, Use::NOP)});
        this->entries_.insert({if_else_stmt, this->users_.at(s).get()});

        graph::Vertex last = s;

        std::unordered_set<std::string> used;
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            auto& condition = if_else_stmt->at(i).second;
            for (auto atom : symbolic::atoms(condition)) {
                if (used.find(atom->get_name()) != used.end()) {
                    continue;
                }
                used.insert(atom->get_name());

                auto v = boost::add_vertex(this->graph_);

                this->add_user(std::make_unique<User>(v, atom->get_name(), if_else_stmt, Use::READ));

                boost::add_edge(last, v, this->graph_);
                last = v;
            }
        }

        // NOP
        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", if_else_stmt, Use::NOP)});
        this->exits_.insert({if_else_stmt, this->users_.at(t).get()});

        // Forward edge: Potentially missing else case
        if (!if_else_stmt->is_complete()) {
            boost::add_edge(last, t, this->graph_);
        }

        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            auto branch = if_else_stmt->at(i);
            auto subgraph = this->traverse(branch.first);
            boost::add_edge(last, subgraph.first, this->graph_);
            if (subgraph.second != boost::graph_traits<graph::Graph>::null_vertex()) {
                boost::add_edge(subgraph.second, t, this->graph_);
            }
        }

        return {s, t};
    } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        // NOP
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", loop_stmt, Use::NOP)});
        this->entries_.insert({loop_stmt, this->users_.at(s).get()});

        auto subgraph = this->traverse(loop_stmt->root());

        // NOP
        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", loop_stmt, Use::NOP)});
        this->exits_.insert({loop_stmt, this->users_.at(t).get()});

        boost::add_edge(s, subgraph.first, this->graph_);
        if (subgraph.second != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(subgraph.second, t, this->graph_);
        }

        // Empty loop
        boost::add_edge(s, t, this->graph_);
        // Back edge
        boost::add_edge(t, s, this->graph_);

        return {s, t};
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        // NOP
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", for_stmt, Use::NOP)});
        auto last = s;
        this->entries_.insert({for_stmt, this->users_.at(s).get()});

        // NOP
        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", for_stmt, Use::NOP)});
        this->exits_.insert({for_stmt, this->users_.at(t).get()});

        // Init
        for (auto atom : symbolic::atoms(for_stmt->init())) {
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, atom->get_name(), for_stmt, Use::READ, true, false, false));
            boost::add_edge(last, v, this->graph_);
            last = v;
        }
        // Indvar
        auto v = boost::add_vertex(this->graph_);
        this->add_user(std::make_unique<
                       ForUser>(v, for_stmt->indvar()->get_name(), for_stmt, Use::WRITE, true, false, false));

        boost::add_edge(last, v, this->graph_);
        last = v;

        // Condition
        for (auto atom : symbolic::atoms(for_stmt->condition())) {
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, atom->get_name(), for_stmt, Use::READ, false, true, false));

            boost::add_edge(last, v, this->graph_);
            boost::add_edge(v, t, this->graph_);
            last = v;
        }

        auto subgraph = this->traverse(for_stmt->root());
        boost::add_edge(last, subgraph.first, this->graph_);

        // Update
        auto end = subgraph.second;
        for (auto atom : symbolic::atoms(for_stmt->update())) {
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, atom->get_name(), for_stmt, Use::READ, false, false, true));
            boost::add_edge(end, v, this->graph_);
            end = v;
        }

        auto update_v = boost::add_vertex(this->graph_);
        this->add_user(std::make_unique<
                       ForUser>(update_v, for_stmt->indvar()->get_name(), for_stmt, Use::WRITE, false, false, true));

        if (end != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(end, update_v, this->graph_);
        }
        end = update_v;

        if (end != boost::graph_traits<graph::Graph>::null_vertex()) {
            boost::add_edge(end, t, this->graph_);
        }

        // Back edge
        boost::add_edge(t, last, this->graph_);

        return {s, t};
    } else if (auto cont_stmt = dynamic_cast<structured_control_flow::Continue*>(&node)) {
        // Approximated by general back edge in loop scope
        auto v = boost::add_vertex(this->graph_);
        this->users_.insert({v, std::make_unique<User>(v, "", cont_stmt, Use::NOP)});
        this->entries_.insert({cont_stmt, this->users_.at(v).get()});
        this->exits_.insert({cont_stmt, this->users_.at(v).get()});
        return {v, v};
    } else if (auto br_stmt = dynamic_cast<structured_control_flow::Break*>(&node)) {
        // Approximated by general back edge in loop scope
        auto v = boost::add_vertex(this->graph_);
        this->users_.insert({v, std::make_unique<User>(v, "", br_stmt, Use::NOP)});
        this->entries_.insert({br_stmt, this->users_.at(v).get()});
        this->exits_.insert({br_stmt, this->users_.at(v).get()});
        return {v, v};
    } else if (auto ret_stmt = dynamic_cast<structured_control_flow::Return*>(&node)) {
        auto v = boost::add_vertex(this->graph_);
        this->users_.insert({v, std::make_unique<User>(v, "", ret_stmt, Use::NOP)});
        this->entries_.insert({ret_stmt, this->users_.at(v).get()});
        this->exits_.insert({ret_stmt, this->users_.at(v).get()});
        return {v, boost::graph_traits<graph::Graph>::null_vertex()};
    }

    throw std::invalid_argument("Invalid control flow node type");
};

Users::Users(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()) {

      };

Users::Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node)
    : Analysis(sdfg), node_(node) {

      };

void Users::run(analysis::AnalysisManager& analysis_manager) {
    users_.clear();
    graph_.clear();
    source_ = nullptr;
    sink_ = nullptr;
    dom_tree_.clear();
    pdom_tree_.clear();
    users_by_sdfg_.clear();
    users_by_sdfg_loop_condition_.clear();
    users_by_sdfg_loop_init_.clear();
    users_by_sdfg_loop_update_.clear();

    reads_.clear();
    writes_.clear();
    views_.clear();
    moves_.clear();

    this->traverse(node_);
    if (this->users_.empty()) {
        return;
    }

    // Require a single source
    for (auto& entry : this->users_) {
        if (boost::in_degree(entry.first, this->graph_) == 0) {
            if (this->source_ != nullptr) {
                throw InvalidSDFGException("Users Analysis: Non-unique source node");
            }
            this->source_ = entry.second.get();
        }
    }
    if (this->source_ == nullptr) {
        throw InvalidSDFGException("Users Analysis: No source node");
    }

    std::list<User*> sinks;
    for (auto& entry : this->users_) {
        if (boost::out_degree(entry.first, this->graph_) == 0) {
            sinks.push_back(entry.second.get());
        }
    }
    if (sinks.size() == 0) {
        throw InvalidSDFGException("Users Analysis: No sink node");
    }
    if (sinks.size() > 1) {
        // add artificial sink
        auto v = boost::add_vertex(this->graph_);
        auto sink = std::make_unique<User>(v, "", nullptr, Use::NOP);
        this->users_.insert({v, std::move(sink)});
        for (auto sink : sinks) {
            boost::add_edge(sink->vertex_, v, this->graph_);
        }
        sinks.clear();

        this->sink_ = this->users_.at(v).get();
    } else {
        this->sink_ = sinks.front();
    }

    // Collect sub structures
    for (auto& entry : this->users_) {
        auto container = entry.second->container();
        switch (entry.second->use()) {
            case Use::READ: {
                if (this->reads_.find(container) == this->reads_.end()) {
                    this->reads_.insert({container, {}});
                }
                this->reads_[container].push_back(entry.second.get());
                break;
            }
            case Use::WRITE: {
                if (this->writes_.find(container) == this->writes_.end()) {
                    this->writes_.insert({container, {}});
                }
                this->writes_[container].push_back(entry.second.get());
                break;
            }
            case Use::VIEW: {
                if (this->views_.find(container) == this->views_.end()) {
                    this->views_.insert({container, {}});
                }
                this->views_[container].push_back(entry.second.get());
                break;
            }
            case Use::MOVE: {
                if (this->moves_.find(container) == this->moves_.end()) {
                    this->moves_.insert({container, {}});
                }
                this->moves_[container].push_back(entry.second.get());
                break;
            }
            default:
                break;
        }
    }

    this->init_dom_tree();
};

std::vector<User*> Users::uses() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->use() == Use::NOP) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::uses(const std::string& container) const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->container() != container) {
            continue;
        }
        if (entry.second->use() == Use::NOP) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::writes() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->use() != Use::WRITE) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::writes(const std::string& container) const {
    if (this->writes_.find(container) == this->writes_.end()) {
        return {};
    } else {
        return this->writes_.at(container);
    }
};

std::vector<User*> Users::reads() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->use() != Use::READ) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::reads(const std::string& container) const {
    if (this->reads_.find(container) == this->reads_.end()) {
        return {};
    } else {
        return this->reads_.at(container);
    }
};

std::vector<User*> Users::views() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->use() != Use::VIEW) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::views(const std::string& container) const {
    if (this->views_.find(container) == this->views_.end()) {
        return {};
    } else {
        return this->views_.at(container);
    }
};

std::vector<User*> Users::moves() const {
    std::vector<User*> us;
    for (auto& entry : this->users_) {
        if (entry.second->use() != Use::MOVE) {
            continue;
        }
        us.push_back(entry.second.get());
    }

    return us;
};

std::vector<User*> Users::moves(const std::string& container) const {
    if (this->moves_.find(container) == this->moves_.end()) {
        return {};
    } else {
        return this->moves_.at(container);
    }
};

structured_control_flow::ControlFlowNode* Users::scope(User* user) {
    if (auto data_node = dynamic_cast<data_flow::DataFlowNode*>(user->element())) {
        return static_cast<structured_control_flow::Block*>(data_node->get_parent().get_parent());
    } else if (auto memlet = dynamic_cast<data_flow::Memlet*>(user->element())) {
        return static_cast<structured_control_flow::Block*>(memlet->get_parent().get_parent());
    } else if (auto transition = dynamic_cast<structured_control_flow::Transition*>(user->element())) {
        return &transition->parent();
    } else {
        auto user_element = dynamic_cast<structured_control_flow::ControlFlowNode*>(user->element());
        assert(user_element != nullptr && "Users::scope: User element is not a ControlFlowNode");
        return user_element;
    }
}

/****** Domination Analysis ******/

bool Users::dominates(User& user1, User& user) {
    auto dominator = this->dom_tree_.at(&user);
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->dom_tree_.at(dominator);
    }
    return false;
};

bool Users::post_dominates(User& user1, User& user) {
    auto dominator = this->pdom_tree_.at(&user);
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->pdom_tree_.at(dominator);
    }
    return false;
};

const std::unordered_set<User*> Users::all_uses_between(User& user1, User& user2) {
    std::unordered_set<User*> uses;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {&user2};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Stop conditions
        if (current == &user1) {
            continue;
        }

        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        if (current != &user1 && current != &user2 && current->use() != Use::NOP) {
            uses.insert(current);
        }

        // Extend search
        // Backward search to utilize domination user1 over user
        auto [eb, ee] = boost::in_edges(current->vertex_, this->graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::source(edge, this->graph_);
            queue.push_back(this->users_.at(v).get());
        }
    }

    return uses;
};

const std::unordered_set<User*> Users::all_uses_after(User& user) {
    std::unordered_set<User*> uses;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {&user};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        if (current != &user && current->use() != Use::NOP) {
            uses.insert(current);
        }

        // Extend search
        auto [eb, ee] = boost::out_edges(current->vertex_, this->graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::target(edge, this->graph_);
            queue.push_back(this->users_.at(v).get());
        }
    }

    return uses;
};

UsersView::UsersView(Users& users, const structured_control_flow::ControlFlowNode& node) : users_(users) {
    this->entry_ = users.entries_.at(&node);
    this->exit_ = users.exits_.at(&node);

    // Collect sub users
    std::unordered_set<User*> visited;
    std::list<User*> queue = {this->exit_};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Stop conditions
        if (current == this->entry_) {
            continue;
        }

        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        this->sub_users_.insert(current);

        // Extend search
        // Backward search to utilize domination user1 over user
        auto [eb, ee] = boost::in_edges(current->vertex_, users.graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::source(edge, users.graph_);
            queue.push_back(users.users_.at(v).get());
        }
    }

    // Collect sub dom tree
    auto& dom_tree = this->users_.dom_tree_;

    for (auto& user : this->sub_users_) {
        this->sub_dom_tree_[user] = dom_tree[user];
    }
    this->sub_dom_tree_[this->entry_] = nullptr;

    // Collect sub post dom tree
    auto& pdom_tree = this->users_.pdom_tree_;
    for (auto& user : this->sub_users_) {
        this->sub_pdom_tree_[user] = pdom_tree[user];
    }
    this->sub_pdom_tree_[this->exit_] = nullptr;
};

std::vector<User*> UsersView::uses() const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->use() == Use::NOP) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::uses(const std::string& container) const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->container() != container) {
            continue;
        }
        if (user->use() == Use::NOP) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::writes() const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->use() != Use::WRITE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::writes(const std::string& container) const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::WRITE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::reads() const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->use() != Use::READ) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::reads(const std::string& container) const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::READ) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::views() const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->use() != Use::VIEW) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::views(const std::string& container) const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::VIEW) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::moves() const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->use() != Use::MOVE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

std::vector<User*> UsersView::moves(const std::string& container) const {
    std::vector<User*> us;
    for (auto& user : this->sub_users_) {
        if (user->container() != container) {
            continue;
        }
        if (user->use() != Use::MOVE) {
            continue;
        }
        us.push_back(user);
    }

    return us;
};

bool UsersView::dominates(User& user1, User& user) {
    auto dominator = this->sub_dom_tree_[&user];
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->sub_dom_tree_[dominator];
    }
    return false;
};

bool UsersView::post_dominates(User& user1, User& user) {
    auto dominator = this->sub_pdom_tree_[&user];
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->sub_pdom_tree_[dominator];
    }
    return false;
};

std::unordered_set<User*> UsersView::all_uses_between(User& user1, User& user2) {
    assert(this->sub_users_.find(&user1) != this->sub_users_.end());
    assert(this->sub_users_.find(&user2) != this->sub_users_.end());

    std::unordered_set<User*> uses;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {&user1};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        if (current != &user1 && current != &user2 && current->use() != Use::NOP) {
            uses.insert(current);
        }

        // Stop conditions
        if (current == exit_) {
            continue;
        }

        if (current == &user2) {
            continue;
        }

        // Extend search
        auto [eb, ee] = boost::out_edges(current->vertex_, this->users_.graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::target(edge, this->users_.graph_);
            queue.push_back(this->users_.users_.at(v).get());
        }
    }

    return uses;
};

std::unordered_set<User*> UsersView::all_uses_after(User& user) {
    assert(this->sub_users_.find(&user) != this->sub_users_.end());

    std::unordered_set<User*> uses;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {&user};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        if (current != &user && current->use() != Use::NOP) {
            uses.insert(current);
        }

        if (current == exit_) {
            continue;
        }

        // Extend search
        auto [eb, ee] = boost::out_edges(current->vertex_, this->users_.graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::target(edge, this->users_.graph_);
            queue.push_back(this->users_.users_.at(v).get());
        }
    }

    return uses;
};

User* Users::
    get_user(const std::string& container, Element* element, Use use, bool is_init, bool is_condition, bool is_update) {
    if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element)) {
        if (is_init) {
            auto tmp = users_by_sdfg_loop_init_.at(container).at(for_loop).at(use);
            return tmp;
        } else if (is_condition) {
            return users_by_sdfg_loop_condition_.at(container).at(for_loop).at(use);
        } else if (is_update) {
            return users_by_sdfg_loop_update_.at(container).at(for_loop).at(use);
        }
    }
    auto tmp = users_by_sdfg_.at(container).at(element).at(use);
    return tmp;
}

void Users::add_user(std::unique_ptr<User> user) {
    auto vertex = user->vertex_;
    this->users_.insert({vertex, std::move(user)});

    auto user_ptr = this->users_.at(vertex).get();
    auto* target_structure = &users_by_sdfg_;
    if (auto for_user = dynamic_cast<ForUser*>(user_ptr)) {
        auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(user_ptr->element());
        if (for_loop == nullptr) {
            throw std::invalid_argument("Invalid user type");
        }
        if (for_user->is_init()) {
            target_structure = &users_by_sdfg_loop_init_;
        } else if (for_user->is_condition()) {
            target_structure = &users_by_sdfg_loop_condition_;
        } else if (for_user->is_update()) {
            target_structure = &users_by_sdfg_loop_update_;
        } else {
            throw std::invalid_argument("Invalid user type");
        }
    }

    if (target_structure->find(user_ptr->container()) == target_structure->end()) {
        target_structure->insert({user_ptr->container(), {}});
    }
    if ((*target_structure)[user_ptr->container()].find(user_ptr->element()) ==
        (*target_structure)[user_ptr->container()].end()) {
        target_structure->at(user_ptr->container()).insert({user_ptr->element(), {}});
    }
    target_structure->at(user_ptr->container()).at(user_ptr->element()).insert({user_ptr->use(), user_ptr});
}

std::unordered_set<std::string> Users::locals(structured_control_flow::ControlFlowNode& node) {
    auto& sdfg = this->sdfg_;

    // Locals have no uses outside of the node
    // We can check this by comparing the number of uses of the container in the view and the total
    // number of uses of the container in the users map.
    std::unordered_set<std::string> locals;
    UsersView view(*this, node);
    for (auto& user : view.uses()) {
        if (!sdfg.is_transient(user->container())) {
            continue;
        }
        if (view.uses(user->container()).size() == this->uses(user->container()).size()) {
            locals.insert(user->container());
        }
    }

    return locals;
};

} // namespace analysis
} // namespace sdfg

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
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

User::User(graph::Vertex vertex, const std::string& container, Element* element, Use use)
    : vertex_(vertex), container_(container), use_(use), element_(element) {

      };

User::User(graph::Vertex vertex, const std::string& container, Element* element,
           data_flow::DataFlowGraph* parent, Use use)
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
    return {{sdfg::symbolic::integer(0)}};
};

ForUser::ForUser(graph::Vertex vertex, const std::string& container, Element* element, Use use,
                 bool is_init, bool is_condition, bool is_update)
    : User(vertex, container, element, use),
      is_init_(is_init),
      is_condition_(is_condition),
      is_update_(is_update) {

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
                    for (auto& iedge : dataflow.in_edges(*access_node)) {
                        if (iedge.src_conn() == "refs" || iedge.dst_conn() == "refs") {
                            use = Use::MOVE;
                            break;
                        }
                    }

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, access_node->data(), access_node,
                                                          &dataflow, use));

                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.src_conn() == "refs" || oedge.dst_conn() == "refs") {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(std::make_unique<User>(v, access_node->data(), access_node,
                                                          &dataflow, use));

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
                    auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(
                        std::make_unique<User>(v, sym->get_name(), tasklet, &dataflow, Use::READ));
                    if (last != boost::graph_traits<graph::Graph>::null_vertex()) {
                        boost::add_edge(last, v, this->graph_);
                    } else {
                        first = v;
                    }
                    last = v;
                }
            }
        }

        for (auto& oedge : dataflow.out_edges(*node)) {
            std::unordered_set<std::string> used;
            for (auto dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                    if (used.find(sym->get_name()) != used.end()) {
                        continue;
                    }
                    used.insert(sym->get_name());

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(
                        std::make_unique<User>(v, sym->get_name(), &oedge, &dataflow, Use::READ));
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

std::pair<graph::Vertex, graph::Vertex> Users::traverse(
    structured_control_flow::ControlFlowNode& node) {
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
                    auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                    if (symbolic::is_pointer(sym)) {
                        continue;
                    }
                    if (used.find(sym->get_name()) != used.end()) {
                        continue;
                    }
                    used.insert(sym->get_name());

                    auto v = boost::add_vertex(this->graph_);
                    this->add_user(
                        std::make_unique<User>(v, sym->get_name(), &child.second, Use::READ));

                    boost::add_edge(current, v, this->graph_);
                    current = v;
                }
            }

            for (auto& entry : child.second.assignments()) {
                auto v = boost::add_vertex(this->graph_);
                this->add_user(
                    std::make_unique<User>(v, entry.first->get_name(), &child.second, Use::WRITE));

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
                auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                if (used.find(sym->get_name()) != used.end()) {
                    continue;
                }
                used.insert(sym->get_name());

                auto v = boost::add_vertex(this->graph_);

                this->add_user(std::make_unique<User>(v, sym->get_name(), if_else_stmt, Use::READ));

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
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(&node)) {
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
            auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, sym->get_name(), for_stmt, Use::READ, true,
                                                     false, false));
            boost::add_edge(last, v, this->graph_);
            last = v;
        }
        // Indvar
        auto v = boost::add_vertex(this->graph_);
        this->add_user(std::make_unique<ForUser>(v, for_stmt->indvar()->get_name(), for_stmt,
                                                 Use::WRITE, true, false, false));

        boost::add_edge(last, v, this->graph_);
        last = v;

        // Condition
        for (auto atom : symbolic::atoms(for_stmt->condition())) {
            auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, sym->get_name(), for_stmt, Use::READ, false,
                                                     true, false));

            boost::add_edge(last, v, this->graph_);
            boost::add_edge(v, t, this->graph_);
            last = v;
        }

        auto subgraph = this->traverse(for_stmt->root());
        boost::add_edge(last, subgraph.first, this->graph_);

        // Update
        auto end = subgraph.second;
        for (auto atom : symbolic::atoms(for_stmt->update())) {
            auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
            auto v = boost::add_vertex(this->graph_);
            this->add_user(std::make_unique<ForUser>(v, sym->get_name(), for_stmt, Use::READ, false,
                                                     false, true));
            boost::add_edge(end, v, this->graph_);
            end = v;
        }

        auto update_v = boost::add_vertex(this->graph_);
        this->add_user(std::make_unique<ForUser>(update_v, for_stmt->indvar()->get_name(), for_stmt,
                                                 Use::WRITE, false, false, true));

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
    } else if (auto kern_stmt = dynamic_cast<structured_control_flow::Kernel*>(&node)) {
        // NOP
        auto s = boost::add_vertex(this->graph_);
        this->users_.insert({s, std::make_unique<User>(s, "", kern_stmt, Use::NOP)});
        this->entries_.insert({kern_stmt, this->users_.at(s).get()});

        auto subgraph = this->traverse(kern_stmt->root());
        boost::add_edge(s, subgraph.first, this->graph_);
        if (subgraph.second == boost::graph_traits<graph::Graph>::null_vertex()) {
            this->exits_.insert({kern_stmt, this->users_.at(s).get()});
            return {s, subgraph.second};
        }

        auto t = boost::add_vertex(this->graph_);
        this->users_.insert({t, std::make_unique<User>(t, "", kern_stmt, Use::NOP)});
        boost::add_edge(subgraph.second, t, this->graph_);
        this->exits_.insert({kern_stmt, this->users_.at(t).get()});

        return {s, t};
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
            assert(this->source_ == nullptr);
            this->source_ = entry.second.get();
        }
    }
    assert(this->source_ != nullptr);

    // Sink may be empty
    for (auto& entry : this->users_) {
        if (boost::out_degree(entry.first, this->graph_) == 0) {
            assert(this->sink_ == nullptr);
            this->sink_ = entry.second.get();
        }
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

/****** Domination Analysis ******/

const std::unordered_map<User*, User*>& Users::dominator_tree() {
    if (this->dom_tree_.empty()) {
        this->init_dom_tree();
    }
    return this->dom_tree_;
};

bool Users::dominates(User& user1, User& user) {
    if (this->dom_tree_.empty()) {
        this->init_dom_tree();
    }
    auto dominator = this->dom_tree_.at(&user);
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->dom_tree_.at(dominator);
    }
    return false;
};

const std::unordered_map<User*, User*>& Users::post_dominator_tree() {
    if (this->pdom_tree_.empty()) {
        this->init_dom_tree();
    }
    return this->pdom_tree_;
};

bool Users::post_dominates(User& user1, User& user) {
    if (this->pdom_tree_.empty()) {
        this->init_dom_tree();
    }
    auto dominator = this->pdom_tree_.at(&user);
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = this->pdom_tree_.at(dominator);
    }
    return false;
};

const std::unordered_set<User*> Users::all_uses_between(User& user1, User& user) {
    assert(this->dominates(user1, user));

    std::unordered_set<User*> uses;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {&user};
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

        if (current != &user1 && current != &user) {
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

const std::unordered_set<User*> Users::all_uses_after(User& user1) {
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

        if (current != &user1) {
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

const std::unordered_set<User*> Users::sources(const std::string& container) const {
    auto source = this->source_;

    std::unordered_set<User*> sources;
    std::unordered_set<User*> visited;
    std::list<User*> queue = {source};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        if (current->container() == container) {
            sources.insert(current);
            continue;
        }

        // Extend search
        auto [eb, ee] = boost::out_edges(current->vertex_, this->graph_);
        auto edges = std::ranges::subrange(eb, ee);
        for (auto edge : edges) {
            auto v = boost::target(edge, this->graph_);
            queue.push_back(this->users_.at(v).get());
        }
    }

    return sources;
};

/****** Happens-Before Analysis ******/

std::unordered_map<std::string, std::vector<data_flow::Subset>> Users::read_subsets() {
    std::unordered_map<std::string, std::vector<data_flow::Subset>> readset;
    for (auto& read : this->users_) {
        if (read.second->use() != Use::READ) {
            continue;
        }

        auto& data = read.second->container();
        if (readset.find(data) == readset.end()) {
            readset[data] = std::vector<data_flow::Subset>();
        }
        auto& subsets = read.second->subsets();
        for (auto& subset : subsets) {
            readset[data].push_back(subset);
        }
    }
    return readset;
};

std::unordered_map<std::string, std::vector<data_flow::Subset>> Users::write_subsets() {
    std::unordered_map<std::string, std::vector<data_flow::Subset>> writeset;
    for (auto& write : this->users_) {
        if (write.second->use() != Use::WRITE) {
            continue;
        }

        auto& data = write.second->container();
        if (writeset.find(data) == writeset.end()) {
            writeset[data] = std::vector<data_flow::Subset>();
        }
        auto& subsets = write.second->subsets();
        for (auto& subset : subsets) {
            writeset[data].push_back(subset);
        }
    }
    return writeset;
};

std::unordered_set<std::string> Users::locals(StructuredSDFG& sdfg,
                                              structured_control_flow::ControlFlowNode& node) {
    // Collect all node elements
    Users local_users(sdfg, node);
    analysis::AnalysisManager analysis_manager(sdfg_);
    local_users.run(analysis_manager);
    std::unordered_map<std::string, std::unordered_set<Element*>> elements;
    for (auto& entry : local_users.users_) {
        if (entry.second->use() == Use::NOP) {
            continue;
        }
        if (!sdfg.is_transient(entry.second->container())) {
            continue;
        }

        if (elements.find(entry.second->container()) == elements.end()) {
            elements[entry.second->container()] = {};
        }
        elements[entry.second->container()].insert(entry.second->element());
    }

    // Determine locals
    for (auto& entry : this->users_) {
        if (entry.second->use() == Use::NOP) {
            continue;
        }

        auto& container = entry.second->container();
        auto element = entry.second->element();
        if (elements.find(container) == elements.end()) {
            continue;
        }
        // used outside of node
        if (elements[container].find(element) == elements[container].end()) {
            elements.erase(container);
        }
    }

    std::unordered_set<std::string> locals;
    for (auto& entry : elements) {
        locals.insert(entry.first);
    }
    return locals;
};

UsersView::UsersView(Users& users, structured_control_flow::ControlFlowNode& node) : users_(users) {
    auto& entry = users.entries_.at(&node);
    auto& exit = users.exits_.at(&node);
    assert(users.dominates(*entry, *exit));
    assert(users.post_dominates(*exit, *entry));

    this->entry_ = entry;
    this->exit_ = exit;
    this->sub_users_ = users.all_uses_between(*entry, *exit);
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

std::unordered_map<User*, User*> UsersView::dominator_tree() {
    if (!this->sub_dom_tree_.empty()) {
        return this->sub_dom_tree_;
    }

    auto dom_tree = this->users_.dominator_tree();
    std::unordered_map<User*, User*> sub_dom_tree;
    for (auto& entry : this->sub_users_) {
        sub_dom_tree[entry] = dom_tree[entry];
    }
    sub_dom_tree[this->entry_] = nullptr;
    return sub_dom_tree;
};

bool UsersView::dominates(User& user1, User& user) {
    auto dom_tree = this->dominator_tree();
    auto dominator = dom_tree[&user];
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = dom_tree[dominator];
    }
    return false;
};

std::unordered_map<User*, User*> UsersView::post_dominator_tree() {
    if (!this->sub_pdom_tree_.empty()) {
        return this->sub_pdom_tree_;
    }

    auto pdom_tree = this->users_.post_dominator_tree();
    std::unordered_map<User*, User*> sub_pdom_tree;
    for (auto& entry : this->sub_users_) {
        sub_pdom_tree[entry] = pdom_tree[entry];
    }
    sub_pdom_tree[this->exit_] = nullptr;
    return sub_pdom_tree;
};

bool UsersView::post_dominates(User& user1, User& user) {
    auto pdom_tree = this->post_dominator_tree();
    auto dominator = pdom_tree[&user];
    while (dominator != nullptr) {
        if (dominator == &user1) {
            return true;
        }
        dominator = pdom_tree[dominator];
    }
    return false;
};

std::unordered_set<User*> UsersView::all_uses_between(User& user1, User& user) {
    assert(this->sub_users_.find(&user1) != this->sub_users_.end());
    assert(this->sub_users_.find(&user) != this->sub_users_.end());
    assert(this->dominates(user1, user));
    bool post_dominates = this->post_dominates(user, user1);

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

        if (current != &user1 && current != &user) {
            uses.insert(current);
        }

        // Stop conditions
        if (current == exit_) {
            continue;
        }

        if (current == &user) {
            continue;
        } else if (!post_dominates) {
            if (this->post_dominates(*current, user)) {
                continue;
            }
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

std::unordered_set<User*> UsersView::all_uses_after(User& user1) {
    assert(this->sub_users_.find(&user1) != this->sub_users_.end());

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

        if (current != &user1) {
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

std::unordered_map<std::string, std::vector<data_flow::Subset>> UsersView::read_subsets() {
    std::unordered_map<std::string, std::vector<data_flow::Subset>> readset;
    for (auto& read : this->sub_users_) {
        if (read->use() != Use::READ) {
            continue;
        }

        auto& data = read->container();
        if (readset.find(data) == readset.end()) {
            readset[data] = std::vector<data_flow::Subset>();
        }
        auto& subsets = read->subsets();
        for (auto& subset : subsets) {
            readset[data].push_back(subset);
        }
    }
    return readset;
};

std::unordered_map<std::string, std::vector<data_flow::Subset>> UsersView::write_subsets() {
    std::unordered_map<std::string, std::vector<data_flow::Subset>> writeset;
    for (auto& write : this->sub_users_) {
        if (write->use() != Use::WRITE) {
            continue;
        }

        auto& data = write->container();
        if (writeset.find(data) == writeset.end()) {
            writeset[data] = std::vector<data_flow::Subset>();
        }
        auto& subsets = write->subsets();
        for (auto& subset : subsets) {
            writeset[data].push_back(subset);
        }
    }
    return writeset;
};

std::unordered_set<std::string> UsersView::locals(StructuredSDFG& sdfg,
                                                  structured_control_flow::ControlFlowNode& node) {
    // Collect all node elements
    Users local_users(sdfg, node);
    analysis::AnalysisManager analysis_manager(users_.sdfg_);
    local_users.run(analysis_manager);
    std::unordered_map<std::string, std::unordered_set<Element*>> elements;
    for (auto& entry : local_users.users_) {
        if (entry.second->use() == Use::NOP) {
            continue;
        }
        if (!sdfg.is_transient(entry.second->container())) {
            continue;
        }

        if (elements.find(entry.second->container()) == elements.end()) {
            elements[entry.second->container()] = {};
        }
        elements[entry.second->container()].insert(entry.second->element());
    }

    // Determine locals
    for (auto& entry : this->sub_users_) {
        if (entry->use() == Use::NOP) {
            continue;
        }

        auto& container = entry->container();
        auto element = entry->element();
        if (elements.find(container) == elements.end()) {
            continue;
        }
        // used outside of node
        if (elements[container].find(element) == elements[container].end()) {
            elements.erase(container);
        }
    }

    std::unordered_set<std::string> locals;
    for (auto& entry : elements) {
        locals.insert(entry.first);
    }
    return locals;
};

User* Users::get_user(const std::string& container, Element* element, Use use, bool is_init,
                      bool is_condition, bool is_update) {
    if (auto for_loop = dynamic_cast<structured_control_flow::For*>(element)) {
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
        auto for_loop = dynamic_cast<structured_control_flow::For*>(user_ptr->element());
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
    target_structure->at(user_ptr->container())
        .at(user_ptr->element())
        .insert({user_ptr->use(), user_ptr});
}

}  // namespace analysis
}  // namespace sdfg

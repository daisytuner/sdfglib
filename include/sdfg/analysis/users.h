#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class Users;
class UsersView;
class DominanceAnalysis;
class DataDependencyAnalysis;

enum Use {
    NOP, // No-op
    READ,
    WRITE,
    VIEW,
    MOVE
};

class User {
    friend class Users;
    friend class UsersView;
    friend class DataDependencyAnalysis;
    friend class DominanceAnalysis;

private:
    graph::Vertex vertex_;

    std::string container_;
    Use use_;

    Element* element_;

public:
    User(graph::Vertex vertex, const std::string& container, Element* element, Use use);

    virtual ~User() = default;

    Use use() const;

    std::string& container();

    Element* element();

    const std::vector<data_flow::Subset> subsets() const;
};

class ForUser : public User {
private:
    bool is_init_;
    bool is_condition_;
    bool is_update_;

public:
    ForUser(
        graph::Vertex vertex,
        const std::string& container,
        Element* element,
        Use use,
        bool is_init,
        bool is_condition,
        bool is_update
    );

    bool is_init() const;

    bool is_condition() const;

    bool is_update() const;
};

class Users : public Analysis {
    friend class AnalysisManager;
    friend class UsersView;
    friend class DominanceAnalysis;

private:
    structured_control_flow::ControlFlowNode& node_;

    graph::Graph graph_;
    User* source_;
    User* sink_;

    std::unordered_map<graph::Vertex, std::unique_ptr<User>, boost::hash<graph::Vertex>> users_;

    std::unordered_map<const structured_control_flow::ControlFlowNode*, User*> entries_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, User*> exits_;

    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>> users_by_sdfg_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_init_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_condition_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_update_;

    std::unordered_map<std::string, std::vector<User*>> reads_;
    std::unordered_map<std::string, std::vector<User*>> writes_;
    std::unordered_map<std::string, std::vector<User*>> views_;
    std::unordered_map<std::string, std::vector<User*>> moves_;

    std::pair<graph::Vertex, graph::Vertex> traverse(data_flow::DataFlowGraph& dataflow);

    std::pair<graph::Vertex, graph::Vertex> traverse(structured_control_flow::ControlFlowNode& node);

    void add_user(std::unique_ptr<User> user);

public:
    Users(StructuredSDFG& sdfg);

    Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

    void run(analysis::AnalysisManager& analysis_manager) override;

    bool has_user(
        const std::string& container,
        Element* element,
        Use use,
        bool is_init = false,
        bool is_condition = false,
        bool is_update = false
    );

    User* get_user(
        const std::string& container,
        Element* element,
        Use use,
        bool is_init = false,
        bool is_condition = false,
        bool is_update = false
    );

    /**** Users ****/

    std::vector<User*> uses() const;

    std::vector<User*> uses(const std::string& container) const;

    size_t num_uses(const std::string& container) const;

    std::vector<User*> writes() const;

    std::vector<User*> writes(const std::string& container) const;

    size_t num_writes(const std::string& container) const;

    std::vector<User*> reads() const;

    std::vector<User*> reads(const std::string& container) const;

    size_t num_reads(const std::string& container) const;

    std::vector<User*> views() const;

    std::vector<User*> views(const std::string& container) const;

    size_t num_views(const std::string& container) const;

    std::vector<User*> moves() const;

    std::vector<User*> moves(const std::string& container) const;

    size_t num_moves(const std::string& container) const;

    static structured_control_flow::ControlFlowNode* scope(User* user);

    std::unordered_set<std::string> locals(structured_control_flow::ControlFlowNode& node);

    const std::unordered_set<User*> all_uses_between(User& user1, User& user2);

    const std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();
};

class UsersView {
private:
    Users& users_;
    User* entry_;
    User* exit_;

    std::unordered_set<User*> sub_users_;

public:
    UsersView(Users& users, const structured_control_flow::ControlFlowNode& node);

    /**** Users ****/

    std::vector<User*> uses() const;

    std::vector<User*> uses(const std::string& container) const;

    std::vector<User*> writes() const;

    std::vector<User*> writes(const std::string& container) const;

    std::vector<User*> reads() const;

    std::vector<User*> reads(const std::string& container) const;

    std::vector<User*> views() const;

    std::vector<User*> views(const std::string& container) const;

    std::vector<User*> moves() const;

    std::vector<User*> moves(const std::string& container) const;

    std::unordered_set<User*> all_uses_between(User& user1, User& user2);

    std::unordered_set<User*> all_uses_after(User& user);

    const std::vector<std::string> all_containers_in_order();
};

} // namespace analysis
} // namespace sdfg

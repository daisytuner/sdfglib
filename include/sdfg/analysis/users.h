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

enum Use {
    NOP,  // No-op
    READ,
    WRITE,
    VIEW,
    MOVE
};

class User {
    friend class Users;
    friend class UsersView;

   private:
    graph::Vertex vertex_;

    std::string container_;
    Use use_;

    Element* element_;
    data_flow::DataFlowGraph* parent_;

   public:
    User(graph::Vertex vertex, const std::string& container, Element* element, Use use);

    User(graph::Vertex vertex, const std::string& container, Element* element,
         data_flow::DataFlowGraph* parent, Use use);

    virtual ~User();

    Use use() const;

    std::string& container();

    Element* element();

    data_flow::DataFlowGraph* parent();

    const std::vector<data_flow::Subset> subsets() const;
};

class ForUser : public User {
   private:
    bool is_init_;
    bool is_condition_;
    bool is_update_;

   public:
    ForUser(graph::Vertex vertex, const std::string& container, Element* element, Use use,
            bool is_init, bool is_condition, bool is_update);

    bool is_init() const;

    bool is_condition() const;

    bool is_update() const;
};

class Users : public Analysis {
    friend class AnalysisManager;
    friend class UsersView;

   private:
    structured_control_flow::ControlFlowNode& node_;

    // Graph representation
    graph::Graph graph_;
    std::map<graph::Vertex, std::unique_ptr<User>> users_;

    std::unordered_map<structured_control_flow::ControlFlowNode*, User*> entries_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, User*> exits_;

    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_init_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_condition_;
    std::unordered_map<std::string, std::unordered_map<Element*, std::unordered_map<Use, User*>>>
        users_by_sdfg_loop_update_;

    // Graph analysis
    User* source_;
    User* sink_;
    std::unordered_map<User*, User*> dom_tree_;
    std::unordered_map<User*, User*> pdom_tree_;

    std::unordered_map<std::string, std::vector<User*>> reads_;
    std::unordered_map<std::string, std::vector<User*>> writes_;
    std::unordered_map<std::string, std::vector<User*>> views_;
    std::unordered_map<std::string, std::vector<User*>> moves_;

    void init_dom_tree();

    std::pair<graph::Vertex, graph::Vertex> traverse(data_flow::DataFlowGraph& dataflow);

    std::pair<graph::Vertex, graph::Vertex> traverse(
        structured_control_flow::ControlFlowNode& node);

    void add_user(std::unique_ptr<User> user);

   public:
    Users(StructuredSDFG& sdfg);

    Users(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

    void run(analysis::AnalysisManager& analysis_manager) override;

    /**** Internals ****/

    User* get_user(const std::string& container, Element* element, Use use, bool is_init = false,
                   bool is_condition = false, bool is_update = false);

    /**** User API ****/

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

    /****** Domination Analysis ******/

    const std::unordered_set<User*> sources(const std::string& container) const;

    const std::unordered_map<User*, User*>& dominator_tree();

    bool dominates(User& user1, User& user);

    const std::unordered_map<User*, User*>& post_dominator_tree();

    bool post_dominates(User& user1, User& user);

    const std::unordered_set<User*> all_uses_between(User& user1, User& user);

    const std::unordered_set<User*> all_uses_after(User& user1);

    /****** Locals ******/

    std::unordered_map<std::string, std::vector<data_flow::Subset>> read_subsets();

    std::unordered_map<std::string, std::vector<data_flow::Subset>> write_subsets();

    std::unordered_set<std::string> locals(StructuredSDFG& sdfg,
                                           structured_control_flow::ControlFlowNode& node);
};

class UsersView {
   private:
    Users& users_;
    User* entry_;
    User* exit_;

    std::unordered_set<User*> sub_users_;

    std::unordered_map<User*, User*> sub_dom_tree_;
    std::unordered_map<User*, User*> sub_pdom_tree_;

   public:
    UsersView(Users& users, structured_control_flow::ControlFlowNode& node);

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

    std::unordered_map<User*, User*> dominator_tree();

    bool dominates(User& user1, User& user);

    std::unordered_map<User*, User*> post_dominator_tree();

    bool post_dominates(User& user1, User& user);

    std::unordered_set<User*> all_uses_between(User& user1, User& user);

    std::unordered_set<User*> all_uses_after(User& user1);

    std::unordered_map<std::string, std::vector<data_flow::Subset>> read_subsets();

    std::unordered_map<std::string, std::vector<data_flow::Subset>> write_subsets();

    std::unordered_set<std::string> locals(StructuredSDFG& sdfg,
                                           structured_control_flow::ControlFlowNode& node);
};

}  // namespace analysis
}  // namespace sdfg
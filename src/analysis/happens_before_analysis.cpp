#include "sdfg/analysis/happens_before_analysis.h"

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

HappensBeforeAnalysis::HappensBeforeAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg),
      node_(sdfg.root()){

      };

HappensBeforeAnalysis::HappensBeforeAnalysis(StructuredSDFG& sdfg,
                                             structured_control_flow::Sequence& node)
    : Analysis(sdfg),
      node_(node){

      };

void HappensBeforeAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    results_.clear();

    std::unordered_set<User*> open_reads;
    std::unordered_map<User*, std::unordered_set<User*>> open_reads_after_writes;
    std::unordered_map<User*, std::unordered_set<User*>> closed_reads_after_write;

    auto& users = analysis_manager.get<Users>();
    visit_sequence(users, node_, open_reads, open_reads_after_writes, closed_reads_after_write);

    for (auto& entry : open_reads_after_writes) {
        closed_reads_after_write.insert(entry);
    }

    for (auto& entry : closed_reads_after_write) {
        if (results_.find(entry.first->container()) == results_.end()) {
            results_.insert({entry.first->container(), {}});
        }
        results_.at(entry.first->container()).insert(entry);
    }
};

/****** Visitor API ******/

void HappensBeforeAnalysis::visit_block(
    analysis::Users& users, structured_control_flow::Block& block,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    auto& dataflow = block.dataflow();

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

                    auto current_user = users.get_user(access_node->data(), access_node, use);

                    if (use == Use::WRITE) {
                        std::unordered_map<User*, std::unordered_set<User*>> to_close;
                        for (auto& user : open_reads_after_writes) {
                            if (user.first->container() == access_node->data()) {
                                to_close.insert(user);
                            }
                        }
                        for (auto& user : to_close) {
                            open_reads_after_writes.erase(user.first);
                            closed_reads_after_write.insert(user);
                        }
                        open_reads_after_writes.insert({current_user, {}});
                    }
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.src_conn() == "refs" || oedge.dst_conn() == "refs") {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    auto current_user = users.get_user(access_node->data(), access_node, use);

                    if (use == Use::READ) {
                        bool found = false;
                        for (auto& user : open_reads_after_writes) {
                            if (user.first->container() == access_node->data()) {
                                user.second.insert(current_user);
                                found = true;
                            }
                        }
                        if (!found) {
                            open_reads.insert(current_user);
                        }
                    }
                }
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(node)) {
            if (tasklet->is_conditional()) {
                auto& condition = tasklet->condition();
                for (auto& atom : symbolic::atoms(condition)) {
                    auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                    auto current_user = users.get_user(sym->get_name(), tasklet, Use::READ);
                    {
                        bool found = false;
                        for (auto& user : open_reads_after_writes) {
                            if (user.first->container() == sym->get_name()) {
                                user.second.insert(current_user);
                                found = true;
                            }
                        }
                        if (!found) {
                            open_reads.insert(current_user);
                        }
                    }
                }
            }
        }

        for (auto& oedge : dataflow.out_edges(*node)) {
            std::unordered_set<std::string> used;
            for (auto& dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                    used.insert(sym->get_name());
                }
            }
            for (auto& sym : used) {
                auto current_user = users.get_user(sym, &oedge, Use::READ);

                {
                    bool found = false;
                    for (auto& user : open_reads_after_writes) {
                        if (user.first->container() == sym) {
                            user.second.insert(current_user);
                            found = true;
                        }
                    }
                    if (!found) {
                        open_reads.insert(current_user);
                    }
                }
            }
        }
    }
}

void HappensBeforeAnalysis::visit_for(
    analysis::Users& users, structured_control_flow::For& for_loop,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    // Read Init
    for (auto atom : symbolic::atoms(for_loop.init())) {
        auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
        auto current_user = users.get_user(sym->get_name(), &for_loop, Use::READ, true);

        bool found = false;
        for (auto& user : open_reads_after_writes) {
            if (user.first->container() == sym->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            open_reads.insert(current_user);
        }
    }

    {
        // Write Induction Variable
        auto current_user =
            users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, true);

        std::unordered_set<User*> to_close;
        for (auto& user : open_reads_after_writes) {
            if (user.first->container() == for_loop.indvar()->get_name()) {
                to_close.insert(user.first);
            }
        }
        for (auto& user : to_close) {
            closed_reads_after_write.insert({user, open_reads_after_writes.at(user)});
            open_reads_after_writes.erase(user);
        }
        open_reads_after_writes.insert({current_user, {}});
    }
    {
        // Write Update
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE,
                                           false, false, true);
        open_reads_after_writes.insert({current_user, {}});
    }

    // Read Condition - Never written in body
    for (auto atom : symbolic::atoms(for_loop.condition())) {
        auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
        auto current_user = users.get_user(sym->get_name(), &for_loop, Use::READ, false, true);

        bool found = false;
        for (auto& user : open_reads_after_writes) {
            if (user.first->container() == sym->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            open_reads.insert(current_user);
        }
    }

    std::unordered_map<User*, std::unordered_set<User*>> open_reads_after_writes_for;
    std::unordered_map<User*, std::unordered_set<User*>> closed_reads_after_writes_for;
    std::unordered_set<User*> open_reads_for;

    visit_sequence(users, for_loop.root(), open_reads_for, open_reads_after_writes_for,
                   closed_reads_after_writes_for);

    for (auto& entry : closed_reads_after_writes_for) {
        closed_reads_after_write.insert(entry);
    }

    // Read Update
    for (auto atom : symbolic::atoms(for_loop.update())) {
        auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
        auto current_user =
            users.get_user(sym->get_name(), &for_loop, Use::READ, false, false, true);

        // Add for body
        for (auto& user : open_reads_after_writes_for) {
            if (user.first->container() == sym->get_name()) {
                user.second.insert(current_user);
            }
        }

        // Add to outside
        bool found = false;
        for (auto& user : open_reads_after_writes) {
            if (user.first->container() == sym->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            open_reads.insert(current_user);
        }
    }

    // Handle open reads of for
    for (auto open_read : open_reads_for) {
        // Add recursive
        for (auto& user : open_reads_after_writes_for) {
            if (user.first->container() == open_read->container()) {
                user.second.insert(open_read);
            }
        }

        bool found = false;
        for (auto& entry : open_reads_after_writes) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            open_reads.insert(open_read);
        }
    }

    // Merge open reads_after_writes
    for (auto& entry : open_reads_after_writes_for) {
        open_reads_after_writes.insert(entry);
    }
}

void HappensBeforeAnalysis::visit_if_else(
    analysis::Users& users, structured_control_flow::IfElse& if_else,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    // Read Conditions
    for (int i = 0; i < if_else.size(); i++) {
        auto child = if_else.at(i).second;
        for (auto atom : symbolic::atoms(child)) {
            auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
            auto current_user = users.get_user(sym->get_name(), &if_else, Use::READ);

            bool found = false;
            for (auto& user : open_reads_after_writes) {
                if (user.first->container() == sym->get_name()) {
                    user.second.insert(current_user);
                    found = true;
                }
            }
            if (!found) {
                open_reads.insert(current_user);
            }
        }
    }

    std::vector<std::unordered_set<User*>> open_reads_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>>
        open_reads_after_writes_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>>
        closed_reads_after_writes_branches(if_else.size());
    for (int i = 0; i < if_else.size(); i++) {
        auto& child = if_else.at(i).first;
        visit_sequence(users, child, open_reads_branches.at(i),
                       open_reads_after_writes_branches.at(i),
                       closed_reads_after_writes_branches.at(i));
    }

    // merge partial open reads
    for (int i = 0; i < if_else.size(); i++) {
        for (auto& entry : open_reads_branches.at(i)) {
            bool found = false;
            for (auto& entry2 : open_reads_after_writes) {
                if (entry2.first->container() == entry->container()) {
                    entry2.second.insert(entry);
                    found = true;
                }
            }
            if (!found) {
                open_reads.insert(entry);
            }
        }
    }

    // merge closed writes
    for (auto& closing : closed_reads_after_writes_branches) {
        for (auto& entry : closing) {
            closed_reads_after_write.insert(entry);
        }
    }

    // Close open reads_after_writes for complete branches
    if (if_else.is_complete()) {
        std::unordered_map<User*, std::unordered_set<User*>> to_close;
        std::unordered_set<std::string> candidates;
        std::unordered_set<std::string> candidates_tmp;

        /* Complete close open reads_after_writes
        1. get candidates from first iteration
        2. iterate over all branches and prune candidates
        3. find prior writes for remaining candidates
        4. close open reads_after_writes for all candidates
        */
        for (auto& entry : open_reads_after_writes_branches.at(0)) {
            candidates.insert(entry.first->container());
        }
        for (auto& entry : closed_reads_after_writes_branches.at(0)) {
            candidates.insert(entry.first->container());
        }

        for (int i = 1; i < if_else.size(); i++) {
            for (auto& entry : open_reads_after_writes_branches.at(i)) {
                if (candidates.find(entry.first->container()) != candidates.end()) {
                    candidates_tmp.insert(entry.first->container());
                }
            }
            candidates.swap(candidates_tmp);
            candidates_tmp.clear();
        }

        for (auto& entry : open_reads_after_writes) {
            if (candidates.find(entry.first->container()) != candidates.end()) {
                to_close.insert(entry);
            }
        }

        for (auto& entry : to_close) {
            open_reads_after_writes.erase(entry.first);
            closed_reads_after_write.insert(entry);
        }
    }

    // merge open reads_after_writes
    for (auto& branch : open_reads_after_writes_branches) {
        for (auto& entry : branch) {
            open_reads_after_writes.insert(entry);
        }
    }
}

void HappensBeforeAnalysis::visit_while(
    analysis::Users& users, structured_control_flow::While& while_loop,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    std::unordered_map<User*, std::unordered_set<User*>> open_reads_after_writes_while;
    std::unordered_map<User*, std::unordered_set<User*>> closed_reads_after_writes_while;
    std::unordered_set<User*> open_reads_while;

    visit_sequence(users, while_loop.root(), open_reads_while, open_reads_after_writes_while,
                   closed_reads_after_writes_while);

    for (auto& entry : closed_reads_after_writes_while) {
        closed_reads_after_write.insert(entry);
    }

    for (auto open_read : open_reads_while) {
        // Add recursively to loop
        for (auto& entry : open_reads_after_writes_while) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
            }
        }

        // Add to outside
        bool found = false;
        for (auto& entry : open_reads_after_writes) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            open_reads.insert(open_read);
        }
    }

    // Keep open reads_after_writes open after loop
    for (auto& entry : open_reads_after_writes_while) {
        open_reads_after_writes.insert(entry);
    }
}

void HappensBeforeAnalysis::visit_return(
    analysis::Users& users, structured_control_flow::Return& return_statement,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    // close all open reads_after_writes
    for (auto& entry : open_reads_after_writes) {
        closed_reads_after_write.insert(entry);
    }
    open_reads_after_writes.clear();
}

void HappensBeforeAnalysis::visit_kernel(
    analysis::Users& users, structured_control_flow::Kernel& kernel,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    visit_sequence(users, kernel.root(), open_reads, open_reads_after_writes,
                   closed_reads_after_write);
}

void HappensBeforeAnalysis::visit_sequence(
    analysis::Users& users, structured_control_flow::Sequence& sequence,
    std::unordered_set<User*>& open_reads,
    std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write) {
    for (int i = 0; i < sequence.size(); i++) {
        auto child = sequence.at(i);
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child.first)) {
            visit_block(users, *block, open_reads, open_reads_after_writes,
                        closed_reads_after_write);
        } else if (auto for_loop = dynamic_cast<structured_control_flow::For*>(&child.first)) {
            visit_for(users, *for_loop, open_reads, open_reads_after_writes,
                      closed_reads_after_write);
        } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
            visit_if_else(users, *if_else, open_reads, open_reads_after_writes,
                          closed_reads_after_write);
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&child.first)) {
            visit_while(users, *while_loop, open_reads, open_reads_after_writes,
                        closed_reads_after_write);
        } else if (auto return_statement =
                       dynamic_cast<structured_control_flow::Return*>(&child.first)) {
            visit_return(users, *return_statement, open_reads, open_reads_after_writes,
                         closed_reads_after_write);
        } else if (auto kernel = dynamic_cast<structured_control_flow::Kernel*>(&child.first)) {
            visit_kernel(users, *kernel, open_reads, open_reads_after_writes,
                         closed_reads_after_write);
        } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&child.first)) {
            visit_sequence(users, *sequence, open_reads, open_reads_after_writes,
                           closed_reads_after_write);
        }

        // handle transitions read
        for (auto& entry : child.second.assignments()) {
            for (auto& atom : symbolic::atoms(entry.second)) {
                auto sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(atom);
                if (symbolic::is_pointer(sym)) {
                    continue;
                }
                auto current_user = users.get_user(sym->get_name(), &child.second, Use::READ);

                bool found = false;
                for (auto& user : open_reads_after_writes) {
                    if (user.first->container() == sym->get_name()) {
                        user.second.insert(current_user);
                        found = true;
                    }
                }
                if (!found) {
                    assert(open_reads.insert(current_user).second);
                }
            }
        }

        // handle transitions write
        for (auto& entry : child.second.assignments()) {
            auto current_user = users.get_user(entry.first->get_name(), &child.second, Use::WRITE);

            std::unordered_set<User*> to_close;
            for (auto& user : open_reads_after_writes) {
                if (user.first->container() == entry.first->get_name()) {
                    to_close.insert(user.first);
                }
            }
            for (auto& user : to_close) {
                closed_reads_after_write.insert({user, open_reads_after_writes.at(user)});
                open_reads_after_writes.erase(user);
            }
            open_reads_after_writes.insert({current_user, {}});
        }
    }
}

std::unordered_set<User*> HappensBeforeAnalysis::reads_after_write(User& write) {
    assert(write.use() == Use::WRITE);
    if (results_.find(write.container()) == results_.end()) {
        return {};
    }
    auto& raws = results_.at(write.container());
    assert(raws.find(&write) != raws.end());

    auto& reads_for_write = raws.at(&write);

    std::unordered_set<User*> reads;
    for (auto& entry : reads_for_write) {
        reads.insert(entry);
    }

    return reads;
};

std::unordered_map<User*, std::unordered_set<User*>> HappensBeforeAnalysis::reads_after_writes(
    const std::string& container) {
    if (results_.find(container) == results_.end()) {
        return {};
    }
    return results_.at(container);
};

std::unordered_map<User*, std::unordered_set<User*>>
HappensBeforeAnalysis::reads_after_write_groups(const std::string& container) {
    auto reads = this->reads_after_writes(container);

    std::unordered_map<User*, std::unordered_set<User*>> read_to_writes_map;
    for (auto& entry : reads) {
        for (auto& read : entry.second) {
            if (read_to_writes_map.find(read) == read_to_writes_map.end()) {
                read_to_writes_map[read] = {};
            }
            read_to_writes_map[read].insert(entry.first);
        }
    }
    return read_to_writes_map;
};

}  // namespace analysis
}  // namespace sdfg

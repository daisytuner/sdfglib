#include "sdfg/analysis/reference_analysis.h"

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/sets.h"

namespace sdfg {
namespace analysis {

ReferenceAnalysis::ReferenceAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()) {

      };

void ReferenceAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    results_.clear();

    std::unordered_set<User*> undefined;
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions;

    auto& users = analysis_manager.get<Users>();
    visit_sequence(users, node_, undefined, open_definitions, closed_definitions);

    for (auto& entry : open_definitions) {
        closed_definitions.insert(entry);
    }

    for (auto& entry : closed_definitions) {
        if (results_.find(entry.first->container()) == results_.end()) {
            results_.insert({entry.first->container(), {}});
        }
        results_.at(entry.first->container()).insert(entry);
    }
};

/****** Visitor API ******/

void ReferenceAnalysis::visit_block(
    analysis::Users& users,
    structured_control_flow::Block& block,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& dataflow = block.dataflow();

    for (auto node : dataflow.topological_sort()) {
        if (!dynamic_cast<data_flow::AccessNode*>(node) || dynamic_cast<data_flow::ConstantNode*>(node)) {
            continue;
        }
        auto access_node = static_cast<data_flow::AccessNode*>(node);

        // New definition (Move)
        if (dataflow.in_degree(*node) == 1) {
            auto& iedge = *dataflow.in_edges(*access_node).begin();
            if (iedge.type() == data_flow::MemletType::Reference) {
                auto current_user = users.get_user(access_node->data(), access_node, Use::MOVE);
                // Close all definitions that we dominate
                std::unordered_map<User*, std::unordered_set<User*>> to_close;
                for (auto& user : open_definitions) {
                    to_close.insert(user);
                }
                for (auto& user : to_close) {
                    open_definitions.erase(user.first);
                    closed_definitions.insert(user);
                }

                // Start new definition
                open_definitions.insert({current_user, {}});

                continue;
            }
        }

        auto read_user = users.get_user(access_node->data(), access_node, Use::READ);
        auto write_user = users.get_user(access_node->data(), access_node, Use::WRITE);
        auto view_user = users.get_user(access_node->data(), access_node, Use::VIEW);
        if (!read_user && !write_user && !view_user) {
            continue;
        }

        // Add uses to open definitions
        bool found = false;
        for (auto& move : open_definitions) {
            if (move.first->container() != access_node->data()) {
                continue;
            }

            if (read_user) {
                move.second.insert(read_user);
            }
            if (write_user) {
                move.second.insert(write_user);
            }
            if (view_user) {
                move.second.insert(view_user);
            }
        }

        if (!found) {
            if (read_user) {
                undefined.insert(read_user);
            }
            if (write_user) {
                undefined.insert(write_user);
            }
            if (view_user) {
                undefined.insert(view_user);
            }
        }
    }
}

void ReferenceAnalysis::visit_if_else(
    analysis::Users& users,
    structured_control_flow::IfElse& if_else,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // Read Conditions
    for (size_t i = 0; i < if_else.size(); i++) {
        auto child = if_else.at(i).second;
        for (auto sym : symbolic::atoms(child)) {
            auto current_user = users.get_user(sym->get_name(), &if_else, Use::READ);

            bool found = false;
            for (auto& user : open_definitions) {
                if (user.first->container() == sym->get_name()) {
                    user.second.insert(current_user);
                    found = true;
                }
            }
            if (!found) {
                undefined.insert(current_user);
            }
        }
    }

    std::vector<std::unordered_set<User*>> undefined_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>> open_definitions_branches(if_else.size());
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>> closed_definitions_branches(if_else.size());
    for (size_t i = 0; i < if_else.size(); i++) {
        auto& child = if_else.at(i).first;
        visit_sequence(
            users, child, undefined_branches.at(i), open_definitions_branches.at(i), closed_definitions_branches.at(i)
        );
    }

    for (auto& closing : closed_definitions_branches) {
        for (auto& entry : closing) {
            closed_definitions.insert(entry);
        }
    }

    for (size_t i = 0; i < if_else.size(); i++) {
        for (auto& entry : undefined_branches.at(i)) {
            bool found = false;
            for (auto& entry2 : open_definitions) {
                if (entry2.first->container() == entry->container()) {
                    entry2.second.insert(entry);
                    found = true;
                }
            }
            if (!found) {
                undefined.insert(entry);
            }
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
        for (auto& entry : open_definitions_branches.at(0)) {
            candidates.insert(entry.first->container());
        }
        for (auto& entry : closed_definitions_branches.at(0)) {
            candidates.insert(entry.first->container());
        }

        for (size_t i = 1; i < if_else.size(); i++) {
            for (auto& entry : open_definitions_branches.at(i)) {
                if (candidates.find(entry.first->container()) != candidates.end()) {
                    candidates_tmp.insert(entry.first->container());
                }
            }
            candidates.swap(candidates_tmp);
            candidates_tmp.clear();
        }

        for (auto& entry : open_definitions) {
            if (candidates.find(entry.first->container()) != candidates.end()) {
                to_close.insert(entry);
            }
        }

        for (auto& entry : to_close) {
            open_definitions.erase(entry.first);
            closed_definitions.insert(entry);
        }
    } else {
        // Over-Approximation: Add all branch definitions to outer definitions or undefined
        // Here we add writes to read sets as "anti-reads"
        for (auto& branch : open_definitions_branches) {
            for (auto& def : branch) {
                bool found = false;
                for (auto& entry : open_definitions) {
                    entry.second.insert(def.first);
                    found = true;
                }
                if (!found) {
                    undefined.insert(def.first);
                }
            }
        }
    }

    for (auto& branch : open_definitions_branches) {
        for (auto& entry : branch) {
            open_definitions.insert(entry);
        }
    }
}

void ReferenceAnalysis::visit_for(
    analysis::Users& users,
    structured_control_flow::StructuredLoop& for_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // Visit body
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_body;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_body;
    std::unordered_set<User*> undefined_body;
    visit_sequence(users, for_loop.root(), undefined_body, open_definitions_body, closed_definitions_body);

    for (auto& entry : closed_definitions_body) {
        closed_definitions.insert(entry);
    }

    for (auto& open : undefined_body) {
        // Loop-carried dependencies
        for (auto& entry : open_definitions_body) {
            if (entry.first->container() == open->container()) {
                entry.second.insert(open);
            }
        }

        bool found = false;
        for (auto entry : open_definitions) {
            if (entry.first->container() != open->container()) {
                continue;
            }

            entry.second.insert(open);
        }

        if (!found) {
            undefined.insert(open);
        }
    }

    for (auto& entry : open_definitions_body) {
        open_definitions.insert(entry);
    }
}

void ReferenceAnalysis::visit_while(
    analysis::Users& users,
    structured_control_flow::While& while_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // Visit body
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_body;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_body;
    std::unordered_set<User*> undefined_body;
    visit_sequence(users, while_loop.root(), undefined_body, open_definitions_body, closed_definitions_body);

    for (auto& entry : closed_definitions_body) {
        closed_definitions.insert(entry);
    }

    for (auto& open : undefined_body) {
        // Loop-carried dependencies
        for (auto& entry : open_definitions_body) {
            if (entry.first->container() == open->container()) {
                entry.second.insert(open);
            }
        }

        bool found = false;
        for (auto entry : open_definitions) {
            if (entry.first->container() != open->container()) {
                continue;
            }

            entry.second.insert(open);
        }

        if (!found) {
            undefined.insert(open);
        }
    }

    for (auto& entry : open_definitions_body) {
        open_definitions.insert(entry);
    }
}

void ReferenceAnalysis::visit_return(
    analysis::Users& users,
    structured_control_flow::Return& return_statement,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    for (auto& entry : open_definitions) {
        closed_definitions.insert(entry);
    }
    open_definitions.clear();
}

void ReferenceAnalysis::visit_sequence(
    analysis::Users& users,
    structured_control_flow::Sequence& sequence,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    for (size_t i = 0; i < sequence.size(); i++) {
        auto child = sequence.at(i);
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child.first)) {
            visit_block(users, *block, undefined, open_definitions, closed_definitions);
        } else if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child.first)) {
            visit_for(users, *for_loop, undefined, open_definitions, closed_definitions);
        } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
            visit_if_else(users, *if_else, undefined, open_definitions, closed_definitions);
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&child.first)) {
            visit_while(users, *while_loop, undefined, open_definitions, closed_definitions);
        } else if (auto return_statement = dynamic_cast<structured_control_flow::Return*>(&child.first)) {
            visit_return(users, *return_statement, undefined, open_definitions, closed_definitions);
        } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&child.first)) {
            visit_sequence(users, *sequence, undefined, open_definitions, closed_definitions);
        }
    }
}

std::unordered_set<User*> ReferenceAnalysis::defines(User& move) {
    assert(move.use() == Use::MOVE);
    if (results_.find(move.container()) == results_.end()) {
        return {};
    }
    auto& raws = results_.at(move.container());
    assert(raws.find(&move) != raws.end());

    auto& users_for_move = raws.at(&move);

    std::unordered_set<User*> users;
    for (auto& entry : users_for_move) {
        users.insert(entry);
    }

    return users;
};

std::unordered_map<User*, std::unordered_set<User*>> ReferenceAnalysis::definitions(const std::string& container) {
    if (results_.find(container) == results_.end()) {
        return {};
    }
    return results_.at(container);
};

std::unordered_set<User*> ReferenceAnalysis::defined_by(User& user) {
    auto definitions = this->definitions(user.container());

    std::unordered_set<User*> moves;
    for (auto& entry : definitions) {
        for (auto& r : entry.second) {
            if (&user == r) {
                moves.insert(entry.first);
            }
        }
    }
    return moves;
};

std::unordered_map<User*, std::unordered_set<User*>> ReferenceAnalysis::defined_by(const std::string& container) {
    auto moves = this->definitions(container);

    std::unordered_map<User*, std::unordered_set<User*>> uses_to_move_map;
    for (auto& entry : moves) {
        for (auto& use : entry.second) {
            if (uses_to_move_map.find(use) == uses_to_move_map.end()) {
                uses_to_move_map[use] = {};
            }
            uses_to_move_map[use].insert(entry.first);
        }
    }
    return uses_to_move_map;
};

} // namespace analysis
} // namespace sdfg

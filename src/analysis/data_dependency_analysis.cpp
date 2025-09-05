#include "sdfg/analysis/data_dependency_analysis.h"

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

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()) {

      };

DataDependencyAnalysis::DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node)
    : Analysis(sdfg), node_(node) {

      };

void DataDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    results_.clear();
    loop_carried_dependencies_.clear();

    std::unordered_set<User*> undefined;
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions;

    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& users = analysis_manager.get<Users>();
    visit_sequence(users, assumptions_analysis, node_, undefined, open_definitions, closed_definitions);

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

void DataDependencyAnalysis::visit_block(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::Block& block,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    auto& dataflow = block.dataflow();

    for (auto node : dataflow.topological_sort()) {
        if (dynamic_cast<data_flow::ConstantNode*>(node) != nullptr) {
            continue;
        }

        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(node)) {
            if (!symbolic::is_pointer(symbolic::symbol(access_node->data()))) {
                if (dataflow.in_degree(*node) > 0) {
                    Use use = Use::WRITE;
                    for (auto& iedge : dataflow.in_edges(*access_node)) {
                        if (iedge.type() == data_flow::MemletType::Reference ||
                            iedge.type() == data_flow::MemletType::Dereference_Src) {
                            use = Use::MOVE;
                            break;
                        }
                    }

                    auto current_user = users.get_user(access_node->data(), access_node, use);

                    if (use == Use::WRITE) {
                        // Close all definitions that we supersede
                        std::unordered_map<User*, std::unordered_set<User*>> to_close;
                        for (auto& user : open_definitions) {
                            if (supersedes(*user.first, *current_user, assumptions_analysis)) {
                                to_close.insert(user);
                            }
                        }
                        for (auto& user : to_close) {
                            open_definitions.erase(user.first);
                            closed_definitions.insert(user);
                        }

                        // Add new definition
                        open_definitions.insert({current_user, {}});
                    }
                }
                if (dataflow.out_degree(*access_node) > 0) {
                    Use use = Use::READ;
                    for (auto& oedge : dataflow.out_edges(*access_node)) {
                        if (oedge.type() == data_flow::MemletType::Reference ||
                            oedge.type() == data_flow::MemletType::Dereference_Dst) {
                            use = Use::VIEW;
                            break;
                        }
                    }

                    auto current_user = users.get_user(access_node->data(), access_node, use);

                    if (use == Use::READ) {
                        // Find all definitions that we read from
                        bool found = false;
                        for (auto& user : open_definitions) {
                            if (intersects(*user.first, *current_user, assumptions_analysis)) {
                                user.second.insert(current_user);
                                found = true;
                            }
                        }
                        if (!found) {
                            undefined.insert(current_user);
                        } else {
                            bool supersedes_all = true;
                            for (auto& user : open_definitions) {
                                if (user.first->container() == current_user->container()) {
                                    supersedes_all &= supersedes(*user.first, *current_user, assumptions_analysis);
                                }
                            }
                            if (!supersedes_all) {
                                undefined.insert(current_user);
                            }
                        }
                    }
                }
            }
        } else if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(node)) {
            for (auto& symbol : library_node->symbols()) {
                auto current_user = users.get_user(symbol->get_name(), library_node, Use::READ);
                {
                    // Find all definitions that we read from
                    bool found = false;
                    bool superseded_all = false;
                    for (auto& user : open_definitions) {
                        if (intersects(*user.first, *current_user, assumptions_analysis)) {
                            user.second.insert(current_user);
                            if (!found) {
                                found = true;
                                superseded_all = true;
                            }
                            superseded_all &= supersedes(*current_user, *user.first, assumptions_analysis);
                        }
                    }
                    if (!superseded_all) {
                        undefined.insert(current_user);
                    }
                }
            }
        }

        for (auto& oedge : dataflow.out_edges(*node)) {
            std::unordered_set<std::string> used;
            for (auto& dim : oedge.subset()) {
                for (auto atom : symbolic::atoms(dim)) {
                    used.insert(atom->get_name());
                }
            }
            for (auto& atom : used) {
                auto current_user = users.get_user(atom, &oedge, Use::READ);

                {
                    // Find all definitions that we read from
                    bool found = false;
                    bool superseded_all = false;
                    for (auto& user : open_definitions) {
                        if (intersects(*user.first, *current_user, assumptions_analysis)) {
                            user.second.insert(current_user);
                            if (!found) {
                                found = true;
                                superseded_all = true;
                            }
                            superseded_all &= supersedes(*current_user, *user.first, assumptions_analysis);
                        }
                    }
                    if (!superseded_all) {
                        undefined.insert(current_user);
                    }
                }
            }
        }
    }
}

void DataDependencyAnalysis::visit_for(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::StructuredLoop& for_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // Init - Read
    for (auto atom : symbolic::atoms(for_loop.init())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, true);

        bool found = false;
        for (auto& user : open_definitions) {
            if (user.first->container() == atom->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(current_user);
        }
    }

    // Init - Write
    {
        // Write Induction Variable
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, true);

        std::unordered_set<User*> to_close;
        for (auto& user : open_definitions) {
            if (supersedes(*user.first, *current_user, assumptions_analysis)) {
                to_close.insert(user.first);
            }
        }
        for (auto& user : to_close) {
            closed_definitions.insert({user, open_definitions.at(user)});
            open_definitions.erase(user);
        }
        open_definitions.insert({current_user, {}});

        // Improve: If loop is executed at least once, we can close the init's definition
        // TODO: Implement this
    }

    // Update - Write
    {
        // Exists after loop and inside body
        auto current_user = users.get_user(for_loop.indvar()->get_name(), &for_loop, Use::WRITE, false, false, true);
        open_definitions.insert({current_user, {}});

        // Improve: If loop is executed at least once, we can close the init's definition
        // TODO: Implement this
    }

    // Condition - Read
    for (auto atom : symbolic::atoms(for_loop.condition())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, false, true);

        bool found = false;
        for (auto& user : open_definitions) {
            if (user.first->container() == atom->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(current_user);
        }
    }

    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_for;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_for;
    std::unordered_set<User*> undefined_for;

    // Add assumptions for body
    visit_sequence(
        users, assumptions_analysis, for_loop.root(), undefined_for, open_definitions_for, closed_definitions_for
    );

    // Update - Read
    for (auto atom : symbolic::atoms(for_loop.update())) {
        auto current_user = users.get_user(atom->get_name(), &for_loop, Use::READ, false, false, true);

        bool found = false;
        for (auto& user : open_definitions_for) {
            if (user.first->container() == atom->get_name()) {
                user.second.insert(current_user);
                found = true;
            }
        }
        if (!found) {
            undefined_for.insert(current_user);
        }
    }

    // Merge for with outside

    // Closed definitions are simply merged
    for (auto& entry : closed_definitions_for) {
        closed_definitions.insert(entry);
    }

    // Undefined reads are matched or forwarded
    for (auto open_read : undefined_for) {
        bool found = false;
        for (auto& entry : open_definitions) {
            if (intersects(*entry.first, *open_read, assumptions_analysis)) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(open_read);
        } else {
            bool subset_all = true;
            for (auto& entry : open_definitions) {
                if (entry.first->container() == open_read->container()) {
                    subset_all &= supersedes_restrictive(*open_read, *entry.first, assumptions_analysis);
                }
            }
            if (!subset_all) {
                undefined.insert(open_read);
            }
        }
    }

    // Open definitions may close outside open definitions after loop
    std::unordered_set<User*> to_close;
    for (auto& previous : open_definitions) {
        for (auto& user : open_definitions_for) {
            if (supersedes(*previous.first, *user.first, assumptions_analysis)) {
                to_close.insert(previous.first);
                break;
            }
        }
    }
    for (auto& user : to_close) {
        closed_definitions.insert({user, open_definitions.at(user)});
        open_definitions.erase(user);
    }

    // Add loop-carried dependencies

    // Criterion 1: Loop is monotonic -> indvar never takes the same value twice
    bool is_monotonic = LoopAnalysis::is_monotonic(&for_loop, assumptions_analysis);
    if (is_monotonic) {
        // Case: Can analyze
        assert(this->loop_carried_dependencies_.insert({&for_loop, {}}).second);
        auto& dependencies = this->loop_carried_dependencies_.at(&for_loop);

        // We can focus on written containers

        // Case 1: Read-Write between iterations
        for (auto& read : undefined_for) {
            for (auto& write : open_definitions_for) {
                if (loop_depends(*write.first, *read, assumptions_analysis, for_loop)) {
                    dependencies[read->container()] = LOOP_CARRIED_DEPENDENCY_READ_WRITE;
                    write.second.insert(read);
                }
            }
        }

        // Case 2: Write-Write between iterations
        for (auto& write : open_definitions_for) {
            if (dependencies.find(write.first->container()) != dependencies.end()) {
                continue;
            }
            for (auto& write_2 : open_definitions_for) {
                if (loop_depends(*write.first, *write_2.first, assumptions_analysis, for_loop)) {
                    dependencies.insert({write.first->container(), LOOP_CARRIED_DEPENDENCY_WRITE_WRITE});
                    break;
                }
            }
        }
    } else {
        // Case: Cannot analyze
        assert(this->loop_carried_dependencies_.insert({&for_loop, {}}).second);
        auto& dependencies = this->loop_carried_dependencies_.at(&for_loop);

        // Over-Approximation:
        // Add loop-carried dependencies for all open reads to all open writes
        for (auto& read : undefined_for) {
            for (auto& write : open_definitions_for) {
                if (intersects(*write.first, *read, assumptions_analysis)) {
                    write.second.insert(read);
                    dependencies.insert({read->container(), LOOP_CARRIED_DEPENDENCY_READ_WRITE});
                }
            }
        }
        // Add loop-carried dependencies for writes
        for (auto& write : open_definitions_for) {
            if (dependencies.find(write.first->container()) == dependencies.end()) {
                dependencies.insert({write.first->container(), LOOP_CARRIED_DEPENDENCY_WRITE_WRITE});
            }
        }
    }

    // Add open definitions from for to outside
    for (auto& entry : open_definitions_for) {
        open_definitions.insert(entry);
    }
}

void DataDependencyAnalysis::visit_if_else(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::IfElse& if_else,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // Read Conditions
    for (size_t i = 0; i < if_else.size(); i++) {
        auto child = if_else.at(i).second;
        for (auto atom : symbolic::atoms(child)) {
            auto current_user = users.get_user(atom->get_name(), &if_else, Use::READ);

            bool found = false;
            for (auto& user : open_definitions) {
                if (user.first->container() == atom->get_name()) {
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
    std::vector<std::unordered_map<User*, std::unordered_set<User*>>> closed_definitionss_branches(if_else.size());
    for (size_t i = 0; i < if_else.size(); i++) {
        auto& child = if_else.at(i).first;
        visit_sequence(
            users,
            assumptions_analysis,
            child,
            undefined_branches.at(i),
            open_definitions_branches.at(i),
            closed_definitionss_branches.at(i)
        );
    }

    // merge partial open reads
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

    // merge closed writes
    for (auto& closing : closed_definitionss_branches) {
        for (auto& entry : closing) {
            closed_definitions.insert(entry);
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
        for (auto& entry : closed_definitionss_branches.at(0)) {
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
                    if (intersects(*entry.first, *def.first, assumptions_analysis)) {
                        entry.second.insert(def.first);
                        found = true;
                    }
                }
                if (!found) {
                    undefined.insert(def.first);
                }
            }
        }
    }

    // merge open reads_after_writes
    for (auto& branch : open_definitions_branches) {
        for (auto& entry : branch) {
            open_definitions.insert(entry);
        }
    }
}

void DataDependencyAnalysis::visit_while(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::While& while_loop,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    std::unordered_map<User*, std::unordered_set<User*>> open_definitions_while;
    std::unordered_map<User*, std::unordered_set<User*>> closed_definitions_while;
    std::unordered_set<User*> undefined_while;

    visit_sequence(
        users, assumptions_analysis, while_loop.root(), undefined_while, open_definitions_while, closed_definitions_while
    );

    // Scope-local closed definitions
    for (auto& entry : closed_definitions_while) {
        closed_definitions.insert(entry);
    }

    for (auto open_read : undefined_while) {
        // Over-Approximation: Add loop-carried dependencies for all open reads
        for (auto& entry : open_definitions_while) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
            }
        }

        // Connect to outside
        bool found = false;
        for (auto& entry : open_definitions) {
            if (entry.first->container() == open_read->container()) {
                entry.second.insert(open_read);
                found = true;
            }
        }
        if (!found) {
            undefined.insert(open_read);
        }
    }

    // Add open definitions from while to outside
    for (auto& entry : open_definitions_while) {
        open_definitions.insert(entry);
    }
}

void DataDependencyAnalysis::visit_return(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::Return& return_statement,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    // close all open reads_after_writes
    for (auto& entry : open_definitions) {
        closed_definitions.insert(entry);
    }
    open_definitions.clear();
}

void DataDependencyAnalysis::visit_sequence(
    analysis::Users& users,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::Sequence& sequence,
    std::unordered_set<User*>& undefined,
    std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
    std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
) {
    for (size_t i = 0; i < sequence.size(); i++) {
        auto child = sequence.at(i);
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child.first)) {
            visit_block(users, assumptions_analysis, *block, undefined, open_definitions, closed_definitions);
        } else if (auto for_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child.first)) {
            visit_for(users, assumptions_analysis, *for_loop, undefined, open_definitions, closed_definitions);
        } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
            visit_if_else(users, assumptions_analysis, *if_else, undefined, open_definitions, closed_definitions);
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&child.first)) {
            visit_while(users, assumptions_analysis, *while_loop, undefined, open_definitions, closed_definitions);
        } else if (auto return_statement = dynamic_cast<structured_control_flow::Return*>(&child.first)) {
            visit_return(users, assumptions_analysis, *return_statement, undefined, open_definitions, closed_definitions);
        } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&child.first)) {
            visit_sequence(users, assumptions_analysis, *sequence, undefined, open_definitions, closed_definitions);
        }

        // handle transitions read
        for (auto& entry : child.second.assignments()) {
            for (auto& atom : symbolic::atoms(entry.second)) {
                if (symbolic::is_pointer(atom)) {
                    continue;
                }
                auto current_user = users.get_user(atom->get_name(), &child.second, Use::READ);

                bool found = false;
                for (auto& user : open_definitions) {
                    if (user.first->container() == atom->get_name()) {
                        user.second.insert(current_user);
                        found = true;
                    }
                }
                if (!found) {
                    undefined.insert(current_user);
                }
            }
        }

        // handle transitions write
        for (auto& entry : child.second.assignments()) {
            auto current_user = users.get_user(entry.first->get_name(), &child.second, Use::WRITE);

            std::unordered_set<User*> to_close;
            for (auto& user : open_definitions) {
                if (user.first->container() == entry.first->get_name()) {
                    to_close.insert(user.first);
                }
            }
            for (auto& user : to_close) {
                closed_definitions.insert({user, open_definitions.at(user)});
                open_definitions.erase(user);
            }
            open_definitions.insert({current_user, {}});
        }
    }
}

bool DataDependencyAnalysis::loop_depends(
    User& previous,
    User& current,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::StructuredLoop& loop
) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();

    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(loop.root(), *previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(loop.root(), *current_scope, true);

    // Check if previous subset is subset of any current subset
    for (auto& previous_subset : previous_subsets) {
        for (auto& current_subset : current_subsets) {
            if (symbolic::maps::intersects(
                    previous_subset, current_subset, loop.indvar(), previous_assumptions, current_assumptions
                )) {
                return true;
            }
        }
    }

    return false;
}

bool DataDependencyAnalysis::supersedes(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();
    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Check if previous subset is subset of any current subset
    for (auto& previous_subset : previous_subsets) {
        bool found = false;
        for (auto& current_subset : current_subsets) {
            if (symbolic::is_subset(previous_subset, current_subset, previous_assumptions, current_assumptions)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

bool DataDependencyAnalysis::
    supersedes_restrictive(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();
    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Check if previous subset is subset of any current subset
    for (auto& previous_subset : previous_subsets) {
        bool found = false;
        for (auto& current_subset : current_subsets) {
            if (symbolic::is_subset(previous_subset, current_subset, previous_assumptions, previous_assumptions)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

bool DataDependencyAnalysis::intersects(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis) {
    if (previous.container() != current.container()) {
        return false;
    }
    // Shortcut for scalars
    auto& type = this->sdfg_.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return true;
    }

    auto& previous_subsets = previous.subsets();
    auto& current_subsets = current.subsets();

    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Check if any current subset intersects with any previous subset
    bool found = false;
    for (auto& current_subset : current_subsets) {
        for (auto& previous_subset : previous_subsets) {
            if (!symbolic::is_disjoint(current_subset, previous_subset, current_assumptions, previous_assumptions)) {
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }

    return found;
}

/****** Public API ******/

std::unordered_set<User*> DataDependencyAnalysis::defines(User& write) {
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

std::unordered_map<User*, std::unordered_set<User*>> DataDependencyAnalysis::definitions(const std::string& container) {
    if (results_.find(container) == results_.end()) {
        return {};
    }
    return results_.at(container);
};

std::unordered_map<User*, std::unordered_set<User*>> DataDependencyAnalysis::defined_by(const std::string& container) {
    auto reads = this->definitions(container);

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

std::unordered_set<User*> DataDependencyAnalysis::defined_by(User& read) {
    assert(read.use() == Use::READ);
    auto definitions = this->definitions(read.container());

    std::unordered_set<User*> writes;
    for (auto& entry : definitions) {
        for (auto& r : entry.second) {
            if (&read == r) {
                writes.insert(entry.first);
            }
        }
    }
    return writes;
};

bool DataDependencyAnalysis::available(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_carried_dependencies_.find(&loop) != this->loop_carried_dependencies_.end();
};

const std::unordered_map<std::string, LoopCarriedDependency>& DataDependencyAnalysis::
    dependencies(structured_control_flow::StructuredLoop& loop) const {
    return this->loop_carried_dependencies_.at(&loop);
};

} // namespace analysis
} // namespace sdfg

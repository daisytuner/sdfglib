#include "sdfg/passes/schedule/stride_minimization.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/transformations/loop_interchange.h"

namespace sdfg {
namespace passes {

StrideMinimization::StrideMinimization()
    : Pass(){

      };

bool StrideMinimization::is_admissible(std::vector<std::string>& current,
                                       std::vector<std::string>& target,
                                       std::unordered_set<std::string>& allowed_swaps) {
    std::vector<size_t> permutation_indices;
    for (auto& p : current) {
        auto it = std::find(target.begin(), target.end(), p);
        permutation_indices.push_back(std::distance(target.begin(), it));
    }
    for (size_t i = 0; i < permutation_indices.size(); i++) {
        for (size_t j = 0; j < permutation_indices.size() - i - 1; j++) {
            if (permutation_indices[j] > permutation_indices[j + 1]) {
                std::string swap =
                    target[permutation_indices[j]] + "_" + target[permutation_indices[j + 1]];
                if (allowed_swaps.find(swap) == allowed_swaps.end()) {
                    return false;
                }
                size_t tmp = permutation_indices[j];
                permutation_indices[j] = permutation_indices[j + 1];
                permutation_indices[j + 1] = tmp;
            }
        }
    }

    return true;
};

std::unordered_set<std::string> StrideMinimization::allowed_swaps(
    Schedule& schedule, std::vector<structured_control_flow::ControlFlowNode*>& nested_loops) {
    auto& builder = schedule.builder();

    std::unordered_set<std::string> allowed_swaps;
    for (size_t i = 0; i < nested_loops.size() - 1; i++) {
        if (dynamic_cast<structured_control_flow::While*>(nested_loops.at(i)) ||
            dynamic_cast<structured_control_flow::While*>(nested_loops.at(i + 1))) {
            continue;
        } else {
            auto first_loop = dynamic_cast<structured_control_flow::For*>(nested_loops.at(i));
            auto second_loop = dynamic_cast<structured_control_flow::For*>(nested_loops.at(i + 1));
            auto& parent = builder.parent(*first_loop);
            transformations::LoopInterchange loop_interchange(parent, *first_loop, *second_loop);
            if (loop_interchange.can_be_applied(schedule)) {
                allowed_swaps.insert(first_loop->indvar()->get_name() + "_" +
                                     second_loop->indvar()->get_name());
            }
        }
    }
    return allowed_swaps;
};

std::pair<bool, std::vector<std::string>> StrideMinimization::can_be_applied(
    Schedule& schedule, std::vector<structured_control_flow::ControlFlowNode*>& nested_loops) {
    if (nested_loops.size() < 2) {
        return {false, {}};
    }
    auto& builder = schedule.builder();
    auto& sdfg = builder.subject();

    size_t while_loops = 0;
    std::vector<std::string> permutation;
    for (auto& loop : nested_loops) {
        if (!dynamic_cast<structured_control_flow::For*>(loop)) {
            permutation.push_back("WHILE_" + std::to_string(while_loops++));
        } else {
            auto for_loop = dynamic_cast<structured_control_flow::For*>(loop);
            permutation.push_back(for_loop->indvar()->get_name());
        }
    }

    auto admissible_swaps = allowed_swaps(schedule, nested_loops);

    // Collect all memory accesses of body
    structured_control_flow::ControlFlowNode* nested_root = nullptr;
    if (auto for_loop = dynamic_cast<structured_control_flow::For*>(nested_loops.at(0))) {
        nested_root = &for_loop->root();
    } else if (auto while_loop =
                   dynamic_cast<structured_control_flow::While*>(nested_loops.at(0))) {
        nested_root = &while_loop->root();
    } else {
        assert(false);
    }
    auto& analysis_manager = schedule.analysis_manager();
    auto& all_users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users(all_users, *nested_root);

    // Collect all memory accesses
    std::vector<data_flow::Subset> subsets;
    for (auto& use : users.reads()) {
        auto element = use->element();
        if (!dynamic_cast<data_flow::AccessNode*>(element)) {
            continue;
        }
        auto access = dynamic_cast<data_flow::AccessNode*>(element);
        auto& graph = access->get_parent();
        for (auto& oedge : graph.out_edges(*access)) {
            auto& subset = oedge.subset();

            // Filter the subset for only dimensions using the permutation
            data_flow::Subset filtered_subset;
            for (auto& dim : subset) {
                for (auto& perm : permutation) {
                    if (symbolic::uses(dim, perm)) {
                        filtered_subset.push_back(dim);
                        break;
                    }
                }
            }
            if (filtered_subset.empty()) {
                continue;
            }

            subsets.push_back(filtered_subset);
        }
    }
    for (auto& use : users.writes()) {
        auto element = use->element();
        if (!dynamic_cast<data_flow::AccessNode*>(element)) {
            continue;
        }
        auto access = dynamic_cast<data_flow::AccessNode*>(element);
        auto& graph = access->get_parent();
        for (auto& iedge : graph.in_edges(*access)) {
            auto& subset = iedge.subset();

            // Filter the subset for only dimensions using the permutation
            data_flow::Subset filtered_subset;
            for (auto& dim : subset) {
                for (auto& perm : permutation) {
                    if (symbolic::uses(dim, perm)) {
                        filtered_subset.push_back(dim);
                        break;
                    }
                }
            }
            if (filtered_subset.empty()) {
                continue;
            }

            subsets.push_back(filtered_subset);
        }
    }

    // Test all permutations (must start from a sorted sequence)
    std::vector<std::string> current = permutation;
    std::sort(current.begin(), current.end());

    std::vector<std::string> best_permutation;
    std::vector<std::map<size_t, size_t>> best_scores;
    do {
        if (!is_admissible(permutation, current, admissible_swaps)) {
            continue;
        }

        // Init scores per dimension
        std::vector<std::map<size_t, size_t>> scores;
        for (auto& var : current) {
            scores.push_back({});
        }

        // Compute skipped dimensions and sum up
        for (auto& subset : subsets) {
            std::vector<size_t> strides;
            for (auto& var : current) {
                bool found = false;
                for (size_t j = 0; j < subset.size(); j++) {
                    auto& dim = subset.at(j);
                    if (symbolic::uses(dim, var)) {
                        strides.push_back(subset.size() - j);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    strides.push_back(0);
                }
            }

            for (size_t i = 0; i < current.size(); i++) {
                auto& score_dim = scores[i];
                if (score_dim.find(strides[i]) == score_dim.end()) {
                    score_dim[strides[i]] = 1;
                } else {
                    score_dim[strides[i]]++;
                }
            }
        }

        // Compare scores
        if (best_permutation.empty()) {
            best_permutation = current;
            best_scores = scores;
        } else {
            bool better = false;
            bool equal = true;
            for (int i = scores.size() - 1; i >= 0; i--) {
                auto& best_score_dim = best_scores[i];
                auto& score_dim = scores[i];

                // Decide by maximal stride
                auto best_max_stride = best_score_dim.rbegin()->first;
                auto max_stride = score_dim.rbegin()->first;
                if (max_stride < best_max_stride) {
                    better = true;
                    equal = false;
                    break;
                } else if (max_stride == best_max_stride) {
                    // Decide by number of occurences
                    auto best_max_stride_count = best_score_dim.rbegin()->second;
                    auto max_stride_count = score_dim.rbegin()->second;
                    if (max_stride_count < best_max_stride_count) {
                        better = true;
                        equal = false;
                        break;
                    } else if (max_stride_count == best_max_stride_count) {
                        continue;
                    } else {
                        equal = false;
                        break;
                    }
                } else {
                    equal = false;
                    break;
                }
            }
            if (better) {
                best_permutation = current;
                best_scores = scores;
            } else if (equal) {
                // If equal and original permutation
                if (std::equal(current.begin(), current.end(), permutation.begin())) {
                    best_permutation = current;
                    best_scores = scores;
                }
            }
        }
    } while (std::next_permutation(current.begin(), current.end()));

    // Check if permutation is better than original
    return {best_permutation != permutation, best_permutation};
};

void StrideMinimization::apply(Schedule& schedule,
                               std::vector<structured_control_flow::ControlFlowNode*>& nested_loops,
                               std::vector<std::string> target_permutation) {
    auto& builder = schedule.builder();

    size_t while_loops = 0;
    std::vector<std::string> permutation;
    for (auto& loop : nested_loops) {
        if (!dynamic_cast<structured_control_flow::For*>(loop)) {
            permutation.push_back("WHILE_" + std::to_string(while_loops++));
        } else {
            auto for_loop = dynamic_cast<structured_control_flow::For*>(loop);
            permutation.push_back(for_loop->indvar()->get_name());
        }
    }
    std::vector<size_t> permutation_indices;
    for (auto& p : permutation) {
        auto it = std::find(target_permutation.begin(), target_permutation.end(), p);
        permutation_indices.push_back(std::distance(target_permutation.begin(), it));
    }

    // Bubble sort permutation indices
    for (size_t i = 0; i < permutation_indices.size(); i++) {
        for (size_t j = 0; j < permutation_indices.size() - i - 1; j++) {
            if (permutation_indices[j] > permutation_indices[j + 1]) {
                auto first_loop = static_cast<structured_control_flow::For*>(nested_loops.at(j));
                auto second_loop =
                    static_cast<structured_control_flow::For*>(nested_loops.at(j + 1));
                auto& parent = builder.parent(*first_loop);
                transformations::LoopInterchange loop_interchange(parent, *first_loop,
                                                                  *second_loop);
                if (!loop_interchange.can_be_applied(schedule)) {
                    throw std::runtime_error("Loop interchange cannot be applied");
                }
                loop_interchange.apply(schedule);
                std::swap(permutation_indices[j], permutation_indices[j + 1]);
            }
        }
    }
};

std::vector<structured_control_flow::ControlFlowNode*> StrideMinimization::children(
    structured_control_flow::ControlFlowNode* node,
    const std::unordered_map<structured_control_flow::ControlFlowNode*,
                             structured_control_flow::ControlFlowNode*>& tree) const {
    // Find unique child
    std::vector<structured_control_flow::ControlFlowNode*> c;
    for (auto& entry : tree) {
        if (entry.second == node) {
            c.push_back(entry.first);
        }
    }
    return c;
};

std::list<std::vector<structured_control_flow::ControlFlowNode*>>
StrideMinimization::loop_tree_paths(
    structured_control_flow::ControlFlowNode* loop,
    const std::unordered_map<structured_control_flow::ControlFlowNode*,
                             structured_control_flow::ControlFlowNode*>& tree) const {
    // Collect all paths in tree starting from loop recursively (DFS)
    std::list<std::vector<structured_control_flow::ControlFlowNode*>> paths;
    auto children = this->children(loop, tree);
    if (children.empty()) {
        paths.push_back({loop});
        return paths;
    }

    for (auto& child : children) {
        auto p = this->loop_tree_paths(child, tree);
        for (auto& path : p) {
            path.insert(path.begin(), loop);
            paths.push_back(path);
        }
    }

    return paths;
};

std::string StrideMinimization::name() { return "StrideMinimization"; };

bool StrideMinimization::run_pass(Schedule& schedule) {
    bool applied = false;

    // Compute loop tree
    auto tree = schedule.loop_tree();

    // Collect outermost loops
    std::list<structured_control_flow::ControlFlowNode*> outer_loops;
    for (auto& node : tree) {
        if (node.second == nullptr) {
            outer_loops.push_back(node.first);
        }
    }

    // Apply stride minimization
    for (auto& outer_loop : outer_loops) {
        auto paths = this->loop_tree_paths(outer_loop, tree);
        for (auto& path : paths) {
            auto [applicable, permutation] = this->can_be_applied(schedule, path);
            if (applicable) {
                this->apply(schedule, path, permutation);
                applied = true;
                break;
            }
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

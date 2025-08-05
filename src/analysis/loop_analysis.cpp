#include "sdfg/analysis/loop_analysis.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace analysis {

LoopAnalysis::LoopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void LoopAnalysis::
    run(structured_control_flow::ControlFlowNode& scope, structured_control_flow::ControlFlowNode* parent_loop) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&scope};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Loop detected
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->loops_.insert(while_stmt);
            this->loop_tree_[while_stmt] = parent_loop;
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->loops_.insert(loop_stmt);
            auto res = this->indvars_.insert({loop_stmt->indvar()->get_name(), loop_stmt});
            if (!res.second) {
                throw sdfg::InvalidSDFGException("Found multiple loops with same indvar");
            }
            this->loop_tree_[loop_stmt] = parent_loop;
        }

        if (dynamic_cast<structured_control_flow::Block*>(current)) {
            continue;
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->run(while_stmt->root(), while_stmt);
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->run(for_stmt->root(), for_stmt);
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            continue;
        } else {
            throw std::runtime_error("Unsupported control flow node type");
        }
    }
}

void LoopAnalysis::run(AnalysisManager& analysis_manager) {
    this->loops_.clear();
    this->loop_tree_.clear();
    this->indvars_.clear();
    this->run(this->sdfg_.root(), nullptr);
}

const std::unordered_set<structured_control_flow::ControlFlowNode*> LoopAnalysis::loops() const { return this->loops_; }

structured_control_flow::ControlFlowNode* LoopAnalysis::find_loop_by_indvar(const std::string& indvar) {
    return this->indvars_.at(indvar);
}

bool LoopAnalysis::is_monotonic(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    auto assums = assumptions_analysis.get(*loop, true);

    return symbolic::series::is_monotonic(loop->update(), loop->indvar(), assums);
}

bool LoopAnalysis::is_contiguous(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    auto assums = assumptions_analysis.get(*loop, true);

    return symbolic::series::is_contiguous(loop->update(), loop->indvar(), assums);
}

symbolic::Expression LoopAnalysis::
    canonical_bound(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    if (!LoopAnalysis::is_monotonic(loop, assumptions_analysis)) {
        return SymEngine::null;
    }

    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(loop->condition());
    } catch (const std::runtime_error& e) {
        return SymEngine::null;
    }

    bool has_complex_clauses = false;
    for (auto& clause : cnf) {
        if (clause.size() > 1) {
            has_complex_clauses = true;
            break;
        }
    }
    if (has_complex_clauses) {
        return SymEngine::null;
    }

    auto indvar = loop->indvar();
    symbolic::Expression bound = SymEngine::null;
    for (auto& clause : cnf) {
        for (auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(literal);
                auto lhs = lt->get_args()[0];
                auto rhs = lt->get_args()[1];
                if (SymEngine::eq(*lhs, *indvar)) {
                    if (bound == SymEngine::null) {
                        bound = rhs;
                    } else {
                        bound = symbolic::min(bound, rhs);
                    }
                } else {
                    return SymEngine::null;
                }
            } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(literal);
                auto lhs = le->get_args()[0];
                auto rhs = le->get_args()[1];
                if (SymEngine::eq(*lhs, *indvar)) {
                    if (bound == SymEngine::null) {
                        bound = symbolic::add(rhs, symbolic::one());
                    } else {
                        bound = symbolic::min(bound, symbolic::add(rhs, symbolic::one()));
                    }
                } else {
                    return SymEngine::null;
                }
            } else {
                return SymEngine::null;
            }
        }
    }

    return bound;
}

symbolic::Integer LoopAnalysis::stride(structured_control_flow::StructuredLoop* loop) {
    auto expr = loop->update();
    auto indvar = loop->indvar();

    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add_expr = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        if (add_expr->get_args().size() != 2) {
            return SymEngine::null;
        }
        auto arg1 = add_expr->get_args()[0];
        auto arg2 = add_expr->get_args()[1];
        if (symbolic::eq(arg1, indvar)) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg2)) {
                return SymEngine::rcp_static_cast<const SymEngine::Integer>(arg2);
            }
        }
        if (symbolic::eq(arg2, indvar)) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg1)) {
                return SymEngine::rcp_static_cast<const SymEngine::Integer>(arg1);
            }
        }
    }
    return SymEngine::null;
}

const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
LoopAnalysis::loop_tree() const {
    return this->loop_tree_;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::parent_loop(structured_control_flow::ControlFlowNode* loop
) const {
    return this->loop_tree_.at(loop);
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::outermost_loops() const {
    std::vector<structured_control_flow::ControlFlowNode*> outermost_loops_;
    for (const auto& [loop, parent] : this->loop_tree_) {
        if (parent == nullptr) {
            outermost_loops_.push_back(loop);
        }
    }
    return outermost_loops_;
}

std::vector<sdfg::structured_control_flow::ControlFlowNode*> LoopAnalysis::children(
    sdfg::structured_control_flow::ControlFlowNode* node,
    const std::unordered_map<
        sdfg::structured_control_flow::ControlFlowNode*,
        sdfg::structured_control_flow::ControlFlowNode*>& tree
) const {
    // Find unique child
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> c;
    for (auto& entry : tree) {
        if (entry.second == node) {
            c.push_back(entry.first);
        }
    }
    return c;
};

std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> LoopAnalysis::loop_tree_paths(
    sdfg::structured_control_flow::ControlFlowNode* loop,
    const std::unordered_map<
        sdfg::structured_control_flow::ControlFlowNode*,
        sdfg::structured_control_flow::ControlFlowNode*>& tree
) const {
    // Collect all paths in tree starting from loop recursively (DFS)
    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> paths;
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

} // namespace analysis
} // namespace sdfg

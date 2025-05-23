#include "sdfg/passes/structured_control_flow/loop_bound_normalization.h"

#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

bool LoopBoundNormalization::apply(builder::StructuredSDFGBuilder& builder,
                                   structured_control_flow::For& loop) {
    auto indvar = loop.indvar();
    auto condition = loop.condition();
    auto update = loop.update();

    // Condition must be of form indvar != bound
    if (!SymEngine::is_a<SymEngine::Unequality>(*condition)) {
        return false;
    }
    auto eq = SymEngine::rcp_static_cast<const SymEngine::Unequality>(condition);
    auto eq_args = eq->get_args();
    auto bound = eq_args.at(0);
    auto symbol = eq_args.at(1);
    if (symbolic::uses(symbol, indvar) && symbolic::uses(bound, indvar)) {
        return false;
    } else if (symbolic::uses(bound, indvar)) {
        bound = eq_args.at(1);
        symbol = eq_args.at(0);
    }
    if (symbolic::strict_monotonicity(symbol, indvar) != symbolic::Sign::POSITIVE) {
        return false;
    }

    // Check if monotonic update
    // TODO: Support more complex updates
    auto match = symbolic::affine(update, indvar);
    if (match.first == SymEngine::null) {
        return false;
    }
    auto first_term = match.first;
    auto second_term = match.second;
    if (!SymEngine::is_a<SymEngine::Integer>(*first_term) ||
        !SymEngine::is_a<SymEngine::Integer>(*second_term)) {
        return false;
    }
    auto multiplier = SymEngine::rcp_static_cast<const SymEngine::Integer>(first_term);
    auto offset = SymEngine::rcp_static_cast<const SymEngine::Integer>(second_term);
    if (multiplier->as_int() >= 1 && offset->as_int() >= 0) {
        auto new_bound = symbolic::Lt(symbol, bound);
        loop.condition() = new_bound;
        return true;
    } else if (multiplier->as_int() <= 1 && offset->as_int() < 0) {
        auto new_bound = symbolic::Gt(symbol, bound);
        loop.condition() = new_bound;
        return true;
    } else {
        return false;
    }

    return true;
};

LoopBoundNormalization::LoopBoundNormalization()
    : Pass() {

      };

std::string LoopBoundNormalization::name() { return "LoopBoundNormalization"; };

bool LoopBoundNormalization::run_pass(builder::StructuredSDFGBuilder& builder,
                                      analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // If sequence, attempt promotion
        if (auto match = dynamic_cast<structured_control_flow::For*>(current)) {
            applied |= this->apply(builder, *match);
        }

        // Add children to queue
        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

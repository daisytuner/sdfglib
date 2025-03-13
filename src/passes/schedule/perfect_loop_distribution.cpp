#include "sdfg/passes/schedule/perfect_loop_distribution.h"

namespace sdfg {
namespace passes {

bool PerfectLoopDistribution::can_be_applied(Schedule& schedule,
                                             structured_control_flow::Sequence& parent,
                                             structured_control_flow::For& loop) {
    if (loop.root().size() == 1) {
        return false;
    }

    bool has_subloop = false;
    for (size_t i = 0; i < loop.root().size(); i++) {
        // skip blocks
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&loop.root().at(i).first)) {
            continue;
        }
        if (auto subloop = dynamic_cast<structured_control_flow::For*>(&loop.root().at(i).first)) {
            has_subloop = true;
            break;
        }
        // if not a block or a loop, then we can't apply the transformation
        return false;
    }
    if (!has_subloop) {
        return false;
    }

    transformations::LoopDistribute loop_distribute(parent, loop);
    if (!loop_distribute.can_be_applied(schedule)) {
        return false;
    }

    return true;
};

void PerfectLoopDistribution::apply(Schedule& schedule, structured_control_flow::Sequence& parent,
                                    structured_control_flow::For& loop) {
    transformations::LoopDistribute loop_distribute(parent, loop);
    loop_distribute.apply(schedule);
};

PerfectLoopDistribution::PerfectLoopDistribution()
    : Pass(){

      };

std::string PerfectLoopDistribution::name() { return "PerfectLoopDistribution"; };

bool PerfectLoopDistribution::run_pass(Schedule& schedule) {
    bool applied = false;

    // Traverse structured SDFG
    auto& builder = schedule.builder();
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                if (auto loop = dynamic_cast<structured_control_flow::For*>(&child.first)) {
                    if (this->can_be_applied(schedule, *sequence_stmt, *loop)) {
                        this->apply(schedule, *sequence_stmt, *loop);
                        applied = true;
                        break;
                    }
                }
            }
            if (applied) {
                break;
            }

            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg

#include "sdfg/passes/scheduler/highway_scheduler.h"

#include "sdfg/transformations/highway_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction HighwayScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    const SchedulerLoopInfo& loop_info
) {
    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        // Apply Highway vectorization to the loop
        transformations::HighwayTransform highway_transform(*map_node);
        if (highway_transform.can_be_applied(builder, analysis_manager)) {
            highway_transform.apply(builder, analysis_manager);
        }
    }

    return NEXT;
}

SchedulerAction HighwayScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    const SchedulerLoopInfo& loop_info
) {
    // HighwayScheduler does not handle while loops
    return NEXT;
}

bool HighwayScheduler::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Initialize queue with innermost loops
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& loop_tree = loop_analysis.loop_tree();
    auto& flop_analysis = analysis_manager.get<analysis::FlopAnalysis>();

    std::list<structured_control_flow::ControlFlowNode*> queue;
    std::unordered_map<structured_control_flow::ControlFlowNode*, SchedulerLoopInfo> scheduling_info_map;
    for (auto& entry : loop_tree) {
        auto& loop = entry.first;
        if (loop_analysis.children(loop).size() != 0) {
            continue;
        }
        queue.push_back(loop);

        SchedulerLoopInfo info;
        info.loop_info = loop_analysis.loop_info(loop);
        info.flop = flop_analysis.get(loop);
        scheduling_info_map[loop] = info;
    }
    if (queue.empty()) {
        return false;
    }

    // Scheduling state machine
    bool applied = false;
    while (!queue.empty()) {
        auto loop = queue.front();
        queue.pop_front();

        auto scheduling_info = scheduling_info_map.at(loop);
        scheduling_info_map.erase(loop);
        if (scheduling_info.loop_info.has_side_effects) {
            continue;
        }

        SchedulerAction action;
        if (auto while_loop = dynamic_cast<structured_control_flow::While*>(loop)) {
            action = schedule(builder, analysis_manager, *while_loop, scheduling_info);
        } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            action = schedule(builder, analysis_manager, *structured_loop, scheduling_info);
        } else {
            throw InvalidSDFGException("LoopScheduler encountered non-loop in loop analysis.");
        }

        switch (action) {
            case SchedulerAction::NEXT: {
                applied = true;
                break;
            }
            case SchedulerAction::CHILDREN: {
                throw InvalidSDFGException("HighwayScheduler cannot schedule non-innermost loops.");
                break;
            }
        }
    }

    return applied;
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg

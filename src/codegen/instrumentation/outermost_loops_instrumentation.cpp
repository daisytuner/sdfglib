#include "sdfg/codegen/instrumentation/outermost_loops_instrumentation.h"

#include "sdfg/analysis/loop_tree_analysis.h"

namespace sdfg {
namespace codegen {

OutermostLoopsInstrumentation::OutermostLoopsInstrumentation(Schedule& schedule)
    : Instrumentation(schedule) {
        auto& analysis_manager = schedule.analysis_manager();
        auto& loop_tree_analysis = analysis_manager.get<analysis::LoopTreeAnalysis>();
        auto ols = loop_tree_analysis.outermost_loops();
        for (auto loop : ols) {
            this->outermost_loops_.insert(loop);
        }
    }

bool OutermostLoopsInstrumentation::should_instrument(const structured_control_flow::ControlFlowNode& node) const {
    return this->outermost_loops_.count(&node);
}

void OutermostLoopsInstrumentation::begin_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
    stream << "__daisy_instrument_enter();" << std::endl;
}

void OutermostLoopsInstrumentation::end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
    stream << "__daisy_instrument_exit(";
    stream << "\"" << schedule_.sdfg().name() << "_" << node.name() << "\", ";
    stream << "\"" << node.debug_info().filename() << "\", ";
    stream << node.debug_info().start_line() << ", ";
    stream << node.debug_info().end_line() << ", ";
    stream << node.debug_info().start_column() << ", ";
    stream << node.debug_info().end_column() << ");" << std::endl;
}

}
}
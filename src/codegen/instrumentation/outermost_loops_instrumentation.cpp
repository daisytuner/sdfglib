#include "sdfg/codegen/instrumentation/outermost_loops_instrumentation.h"

#include "sdfg/analysis/loop_tree_analysis.h"

namespace sdfg {
namespace codegen {

OutermostLoopsInstrumentation::OutermostLoopsInstrumentation(StructuredSDFG& sdfg)
    : Instrumentation(sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopTreeAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();
    for (auto loop : ols) {
        this->outermost_loops_.insert(loop);
    }
}

bool OutermostLoopsInstrumentation::should_instrument(
    const structured_control_flow::ControlFlowNode& node) const {
    return this->outermost_loops_.count(&node);
}

void OutermostLoopsInstrumentation::begin_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
    stream << "__daisy_instrument_enter();" << std::endl;
}

void OutermostLoopsInstrumentation::end_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream) const {
    std::string region_name = sdfg_.name() + "_" + node.element_id();

    bool has_metadata = sdfg_.metadata().find("source_file") != sdfg_.metadata().end() &&
                        sdfg_.metadata().find("features_path") != sdfg_.metadata().end();

    if (has_metadata) {
        stream << "__daisy_instrument_exit_with_metadata(";
    } else {
        stream << "__daisy_instrument_exit(";
    }

    stream << "\"" << region_name << "\", ";
    stream << "\"" << node.debug_info().filename() << "\", ";
    if (sdfg_.metadata().find("function") != sdfg_.metadata().end()) {
        stream << "\"" << sdfg_.metadata("function") << "\", ";
    }
    stream << node.debug_info().start_line() << ", ";
    stream << node.debug_info().end_line() << ", ";
    stream << node.debug_info().start_column() << ", ";
    stream << node.debug_info().end_column();

    if (has_metadata) {
        stream << ", ";
        stream << "\"" << sdfg_.metadata("source_file") << "\", ";

        std::string features_path = sdfg_.metadata("features_path") + "/" + region_name + ".npz";
        stream << "\"" << features_path << "\"";
    }

    stream << ");" << std::endl;
}

}  // namespace codegen
}  // namespace sdfg
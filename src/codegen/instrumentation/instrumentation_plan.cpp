#include "sdfg/codegen/instrumentation/instrumentation_plan.h"

#include "sdfg/analysis/loop_analysis.h"

namespace sdfg {
namespace codegen {

bool InstrumentationPlan::should_instrument(const structured_control_flow::ControlFlowNode& node) const {
    return this->nodes_.count(&node);
}

void InstrumentationPlan::begin_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream)
    const {
    std::string region_name = sdfg_.name() + "_" + std::to_string(node.element_id());
    auto& dbg_info = node.debug_info();

    // Create metadata variable
    std::string metdata_var = sdfg_.name() + "_" + std::to_string(node.element_id());
    stream << "__daisy_metadata_t " << metdata_var << ";" << std::endl;
    stream << metdata_var << ".region_name = \"" << region_name << "\";" << std::endl;
    stream << metdata_var << ".function_name = \"" << sdfg_.metadata("function") << "\";" << std::endl;
    stream << metdata_var << ".file_name = \"" << dbg_info.filename() << "\";" << std::endl;
    stream << metdata_var << ".line_begin = " << dbg_info.start_line() << ";" << std::endl;
    stream << metdata_var << ".line_end = " << dbg_info.end_line() << ";" << std::endl;
    stream << metdata_var << ".column_begin = " << dbg_info.start_column() << ";" << std::endl;
    stream << metdata_var << ".column_end = " << dbg_info.end_column() << ";" << std::endl;

    stream << "__daisy_instrumentation_enter(__daisy_instrumentation_ctx, &" << metdata_var << ");" << std::endl;
}

void InstrumentationPlan::end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream)
    const {
    std::string metdata_var = sdfg_.name() + "_" + std::to_string(node.element_id());
    stream << "__daisy_instrumentation_exit(__daisy_instrumentation_ctx, &" << metdata_var << ");" << std::endl;
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<
        InstrumentationPlan>(sdfg, std::unordered_set<const structured_control_flow::ControlFlowNode*>{});
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_set<const structured_control_flow::ControlFlowNode*> nodes;
    for (auto loop : ols) {
        nodes.insert(loop);
    }

    return std::make_unique<InstrumentationPlan>(sdfg, nodes);
}

} // namespace codegen
} // namespace sdfg

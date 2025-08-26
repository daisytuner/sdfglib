#include "sdfg/codegen/instrumentation/instrumentation_plan.h"

#include "sdfg/analysis/loop_analysis.h"

namespace sdfg {
namespace codegen {

void InstrumentationPlan::update(const structured_control_flow::ControlFlowNode& node, InstrumentationEventType event_type) {
    this->nodes_[&node] = event_type;
}

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
    stream << metdata_var << ".function_name = \"" << dbg_info.function() << "\";" << std::endl;
    stream << metdata_var << ".file_name = \"" << dbg_info.filename() << "\";" << std::endl;
    stream << metdata_var << ".line_begin = " << dbg_info.start_line() << ";" << std::endl;
    stream << metdata_var << ".line_end = " << dbg_info.end_line() << ";" << std::endl;
    stream << metdata_var << ".column_begin = " << dbg_info.start_column() << ";" << std::endl;
    stream << metdata_var << ".column_end = " << dbg_info.end_column() << ";" << std::endl;
    if (!(this->loopnest_indices_.empty())) {
        stream << metdata_var << ".loopnest_index = " << this->loopnest_indices_.at(&node) << ";" << std::endl;
    }

    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "__daisy_instrumentation_enter(__daisy_instrumentation_ctx, &" << metdata_var << ", "
               << "__DAISY_EVENT_SET_CPU" << ");" << std::endl;
    } else {
        stream << "__daisy_instrumentation_enter(__daisy_instrumentation_ctx, &" << metdata_var << ", "
               << "__DAISY_EVENT_SET_CUDA" << ");" << std::endl;
    }
}

void InstrumentationPlan::end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream)
    const {
    std::string metdata_var = sdfg_.name() + "_" + std::to_string(node.element_id());
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "__daisy_instrumentation_exit(__daisy_instrumentation_ctx, &" << metdata_var << ", "
               << "__DAISY_EVENT_SET_CPU" << ");" << std::endl;
    } else {
        stream << "__daisy_instrumentation_exit(__daisy_instrumentation_ctx, &" << metdata_var << ", "
               << "__DAISY_EVENT_SET_CUDA" << ");" << std::endl;
    }
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<InstrumentationPlan>(
        sdfg, std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType>{}
    );
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType> nodes;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, size_t> loopnest_indices;
    for (size_t i = 0; i < ols.size(); i++) {
        auto& loop = ols[i];
        if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            if (map_node->schedule_type().value() == "CUDA") {
                nodes.insert({loop, InstrumentationEventType::CUDA});
                continue;
            }
        }
        loopnest_indices[loop] = i;
        nodes.insert({loop, InstrumentationEventType::CPU}); // Default to CPU if not CUDA
    }
    return std::make_unique<InstrumentationPlan>(sdfg, nodes, loopnest_indices);
}

} // namespace codegen
} // namespace sdfg

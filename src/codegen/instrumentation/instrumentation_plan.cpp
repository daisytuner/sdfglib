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
    auto& metadata = sdfg_.metadata();
    std::string sdfg_name = sdfg_.name();
    std::string sdfg_file = metadata.at("sdfg_file");
    std::string arg_capture_path = metadata.at("arg_capture_path");
    std::string features_file = metadata.at("features_file");

    std::string region_uuid = sdfg_name + "_" + std::to_string(node.element_id());

    // Create region id variable
    std::string region_id_var = sdfg_name + "_" + std::to_string(node.element_id()) + "_id";

    // Create metadata variable
    std::string metadata_var = sdfg_name + "_" + std::to_string(node.element_id()) + "_md";
    stream << "__daisy_metadata_t " << metadata_var << ";" << std::endl;

    // Source metadata
    auto& dbg_info = node.debug_info();
    stream << metadata_var << ".file_name = \"" << dbg_info.filename() << "\";" << std::endl;
    stream << metadata_var << ".function_name = \"" << dbg_info.function() << "\";" << std::endl;
    stream << metadata_var << ".line_begin = " << dbg_info.start_line() << ";" << std::endl;
    stream << metadata_var << ".line_end = " << dbg_info.end_line() << ";" << std::endl;
    stream << metadata_var << ".column_begin = " << dbg_info.start_column() << ";" << std::endl;
    stream << metadata_var << ".column_end = " << dbg_info.end_column() << ";" << std::endl;

    // Docc metadata
    stream << metadata_var << ".sdfg_name = \"" << sdfg_name << "\";" << std::endl;
    stream << metadata_var << ".sdfg_file = \"" << sdfg_file << "\";" << std::endl;
    stream << metadata_var << ".arg_capture_path = \"" << arg_capture_path << "\";" << std::endl;
    stream << metadata_var << ".features_file = \"" << features_file << "\";" << std::endl;
    stream << metadata_var << ".element_id = " << node.element_id() << ";" << std::endl;
    if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(&node)) {
        stream << metadata_var << ".element_type = \"map\";" << std::endl;
        stream << metadata_var << ".target_type = \"" << map_node->schedule_type().value() << "\";" << std::endl;
    } else if (dynamic_cast<const structured_control_flow::For*>(&node)) {
        stream << metadata_var << ".element_type = \"for\";" << std::endl;
        stream << metadata_var << ".target_type = \"SEQUENTIAL\";" << std::endl;
    } else if (dynamic_cast<const structured_control_flow::While*>(&node)) {
        stream << metadata_var << ".element_type = \"while\";" << std::endl;
        stream << metadata_var << ".target_type = \"SEQUENTIAL\";" << std::endl;
    } else {
        stream << metadata_var << ".element_type = \"\";" << std::endl;
        stream << metadata_var << ".target_type = \"\";" << std::endl;
    }
    if (!this->loopnest_indices_.empty()) {
        stream << metadata_var << ".loopnest_index = " << this->loopnest_indices_.at(&node) << ";" << std::endl;
    } else {
        stream << metadata_var << ".loopnest_index = -1;" << std::endl;
    }
    stream << metadata_var << ".region_uuid = \"" << region_uuid << "\";" << std::endl;

    // Initialize region
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
               << ", __DAISY_EVENT_SET_CPU);" << std::endl;
    } else {
        stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
               << ", __DAISY_EVENT_SET_CUDA);" << std::endl;
    }

    // Enter region
    stream << "__daisy_instrumentation_enter(" << region_id_var << ");" << std::endl;
}

void InstrumentationPlan::end_instrumentation(const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream)
    const {
    std::string region_id_var = sdfg_.name() + "_" + std::to_string(node.element_id()) + "_id";

    // Exit region
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    } else {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    }

    // Finalize region
    stream << "__daisy_instrumentation_finalize(" << region_id_var << ");" << std::endl;
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
        loopnest_indices[loop] = i;
        if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            if (map_node->schedule_type().value() == "CUDA") {
                nodes.insert({loop, InstrumentationEventType::CUDA});
                continue;
            }
        }
        nodes.insert({loop, InstrumentationEventType::CPU}); // Default to CPU if not CUDA
    }
    return std::make_unique<InstrumentationPlan>(sdfg, nodes, loopnest_indices);
}

} // namespace codegen
} // namespace sdfg

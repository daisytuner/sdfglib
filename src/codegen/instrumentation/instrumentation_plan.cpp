#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include <memory>
#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/element.h"

namespace sdfg {
namespace codegen {

void InstrumentationPlan::update(const structured_control_flow::ControlFlowNode& node, InstrumentationEventType event_type) {
    this->nodes_[&node] = event_type;
}

bool InstrumentationPlan::should_instrument(const Element& node) const { return this->nodes_.count(&node); }

void InstrumentationPlan::begin_instrumentation(
    const Element& node, PrettyPrinter& stream, LanguageExtension& language_extension, const InstrumentationInfo& info
) const {
    auto& metadata = sdfg_.metadata();
    std::string sdfg_name = sdfg_.name();
    std::string sdfg_file = metadata.at("sdfg_file");
    std::string arg_capture_path = metadata.at("arg_capture_path");
    std::string features_file = metadata.at("features_file");
    std::string opt_report_file = metadata.at("opt_report_file");

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
    stream << metadata_var << ".opt_report_file = \"" << opt_report_file << "\";" << std::endl;
    stream << metadata_var << ".element_id = " << info.element_id() << ";" << std::endl;
    stream << metadata_var << ".element_type = \"" << info.element_type().value() << "\";" << std::endl;
    stream << metadata_var << ".target_type = \"" << info.target_type().value() << "\";" << std::endl;
    stream << metadata_var << ".loopnest_index = " << info.loopnest_index() << ";" << std::endl;
    stream << metadata_var << ".region_uuid = \"" << region_uuid << "\";" << std::endl;

    // Initialize region
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
               << ", __DAISY_EVENT_SET_CPU);" << std::endl;
    } else if (this->nodes_.at(&node) == InstrumentationEventType::CUDA) {
        stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
               << ", __DAISY_EVENT_SET_CUDA);" << std::endl;
    } else {
        stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
               << ", __DAISY_EVENT_SET_NONE);" << std::endl;
    }

    // Enter region
    stream << "__daisy_instrumentation_enter(" << region_id_var << ");" << std::endl;
}

void InstrumentationPlan::end_instrumentation(
    const Element& node, PrettyPrinter& stream, LanguageExtension& language_extension, const InstrumentationInfo& info
) const {
    std::string region_id_var = sdfg_.name() + "_" + std::to_string(node.element_id()) + "_id";

    // Exit region
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    } else {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    }

    // Perform FlopAnalysis
    if (auto structured_node = dynamic_cast<const structured_control_flow::ControlFlowNode*>(&node)) {
        if (this->flops_.contains(structured_node)) {
            auto flop = this->flops_.at(structured_node);
            if (!flop.is_null()) {
                std::string flop_str = language_extension.expression(flop);
                stream << "__daisy_instrumentation_increment(" << region_id_var << ", \"flop\", " << flop_str << ");"
                       << std::endl;
            }
        }
    }

    for (auto entry : info.metrics()) {
        stream << "__daisy_instrumentation_increment(" << region_id_var << ", \"" << entry.first << "\", "
               << entry.second << ");" << std::endl;
    }

    // Finalize region
    stream << "__daisy_instrumentation_finalize(" << region_id_var << ");" << std::endl;
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<InstrumentationPlan>(sdfg, std::unordered_map<const Element*, InstrumentationEventType>{});
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_map<const Element*, InstrumentationEventType> nodes;
    for (size_t i = 0; i < ols.size(); i++) {
        auto& loop = ols[i];
        if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            if (map_node->schedule_type().value() == "CUDA") {
                nodes.insert({loop, InstrumentationEventType::CUDA});
                continue;
            }
        }
        nodes.insert({loop, InstrumentationEventType::CPU}); // Default to CPU if not CUDA
    }

    auto sdfg_clone = sdfg.clone();
    auto builder = builder::StructuredSDFGBuilder(sdfg_clone);
    LibNodeFinder lib_node_finder(builder, analysis_manager);
    lib_node_finder.visit();
    for (auto& lib_node : lib_node_finder.get_lib_nodes_D2H()) {
        nodes.insert({lib_node, InstrumentationEventType::D2H});
    }
    for (auto& lib_node : lib_node_finder.get_lib_nodes_H2D()) {
        nodes.insert({lib_node, InstrumentationEventType::H2D});
    }

    return std::make_unique<InstrumentationPlan>(sdfg, nodes);
}

bool LibNodeFinder::accept(structured_control_flow::Block& node) {
    for (auto libnode : node.dataflow().library_nodes()) {
        if (libnode->code().value() == "CUDAD2HTransfer" || libnode->code().value() == "TTEnqueueRead") {
            lib_nodes_D2H.push_back(libnode);
        }
        if (libnode->code().value() == "CUDAH2DTransfer" || libnode->code().value() == "TTEnqueueWrite") {
            lib_nodes_H2D.push_back(libnode);
        }
    }
}

bool LibNodeFinder::accept(structured_control_flow::Sequence& node) { return true; }

bool LibNodeFinder::accept(structured_control_flow::IfElse& node) { return true; }

bool LibNodeFinder::accept(structured_control_flow::For& node) { return true; }

bool LibNodeFinder::accept(structured_control_flow::While& node) { return true; }

bool LibNodeFinder::accept(structured_control_flow::Map& node) { return true; }

} // namespace codegen
} // namespace sdfg

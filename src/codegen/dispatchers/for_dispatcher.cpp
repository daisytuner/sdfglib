#include "sdfg/codegen/dispatchers/for_dispatcher.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"

namespace sdfg {
namespace codegen {

ForDispatcher::ForDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::For& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void ForDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

InstrumentationInfo ForDispatcher::instrumentation_info() const {
    long long loopnest_index = -1;

    // Check if this is an outermost loop
    auto& loop_tree_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    auto outermost_loops = loop_tree_analysis.outermost_loops();
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        if (outermost_loops[i] == &node_) {
            loopnest_index = i;
            break;
        }
    }

    // Query LoopInfo
    analysis::LoopInfo loop_info;
    if (loopnest_index != -1) {
        loop_info = loop_tree_analysis.loop_info(&node_);
    }

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        std::string flop_str = language_extension_.expression(flop);
        metrics.insert({"flop", flop_str});
    }

    return InstrumentationInfo(
        node_.element_id(), ElementType_For, TargetType_SEQUENTIAL, loop_info, loopnest_index, metrics
    );
};

} // namespace codegen
} // namespace sdfg

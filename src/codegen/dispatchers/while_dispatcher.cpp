#include "sdfg/codegen/dispatchers/while_dispatcher.h"

namespace sdfg {
namespace codegen {

WhileDispatcher::WhileDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void WhileDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "while (1)" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

BreakDispatcher::BreakDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Break& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void BreakDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "break;" << std::endl;
};

ContinueDispatcher::ContinueDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Continue& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void ContinueDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "continue;" << std::endl;
};

ReturnDispatcher::ReturnDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Return& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan), node_(node) {

      };

void ReturnDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    // Free heap allocations
    for (auto& container : sdfg_.containers()) {
        if (sdfg_.is_external(container)) {
            continue;
        }
        auto& type = sdfg_.type(container);

        // Free if needed
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            if (type.storage_type().is_cpu_heap()) {
                main_stream << language_extension_.external_prefix() << "free(" << container << ");" << std::endl;
            } else if (type.storage_type().is_nv_generic()) {
                main_stream << "cudaSetDevice(0);" << std::endl;
                main_stream << "cudaFree(" << container << ");" << std::endl;
            }
        }
    }

    if (node_.unreachable()) {
        main_stream << "/* unreachable return */" << std::endl;
    } else if (node_.is_data()) {
        std::string return_str = node_.data();
        if (sdfg_.is_external(node_.data())) {
            return_str = "&" + this->language_extension_.external_prefix() + return_str;
        }
        main_stream << "return " << return_str << ";" << std::endl;
    } else if (node_.is_constant()) {
        if (symbolic::is_nullptr(symbolic::symbol(node_.data()))) {
            main_stream << "return " << this->language_extension_.expression(symbolic::symbol(node_.data())) << ";"
                        << std::endl;
        } else {
            main_stream << "return " << node_.data() << ";" << std::endl;
        }
    }
};

} // namespace codegen
} // namespace sdfg

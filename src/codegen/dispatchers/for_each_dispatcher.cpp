#include "sdfg/codegen/dispatchers/for_each_dispatcher.h"

namespace sdfg {
namespace codegen {

ForEachDispatcher::ForEachDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::ForEach& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void ForEachDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    types::Pointer ptr_type;
    types::Pointer ptr_ptr_type(static_cast<const types::IType&>(ptr_type));

    std::string iterator = language_extension_.expression(node_.iterator());

    main_stream << "for";
    main_stream << "(";
    if (node_.has_init()) {
        main_stream << iterator;
        main_stream << " = ";
        main_stream << "*(" << language_extension_.type_cast(language_extension_.expression(node_.init()), ptr_ptr_type) << ")";
    }
    main_stream << ";";
    main_stream << iterator;
    main_stream << " != ";
    main_stream << language_extension_.expression(node_.end());
    main_stream << ";";
    main_stream << iterator;
    main_stream << " = ";
    main_stream << "*(" << language_extension_.type_cast(language_extension_.expression(node_.update()), ptr_ptr_type) << ")";
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, sdfg_, node_.root(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

} // namespace codegen
} // namespace sdfg

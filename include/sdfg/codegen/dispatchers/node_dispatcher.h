#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/instrumentation/instrumentation.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace codegen {

class NodeDispatcher {
private:
    structured_control_flow::ControlFlowNode& node_;

protected:
    LanguageExtension& language_extension_;

    StructuredSDFG& sdfg_;

    Instrumentation& instrumentation_;

    virtual bool begin_node(PrettyPrinter& stream);

    virtual void end_node(PrettyPrinter& stream, bool has_declaration);

public:
    NodeDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        structured_control_flow::ControlFlowNode& node,
        Instrumentation& instrumentation
    );

    virtual ~NodeDispatcher() = default;

    virtual void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) = 0;

    virtual void
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
};

} // namespace codegen
} // namespace sdfg

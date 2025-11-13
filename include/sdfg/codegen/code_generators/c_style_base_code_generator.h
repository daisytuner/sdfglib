#pragma once

#include <vector>

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

class CStyleBaseCodeGenerator : public CodeGenerator {
protected:
    virtual LanguageExtension& language_extension() = 0;

    virtual void dispatch_includes() = 0;

    virtual void dispatch_structures() = 0;

    virtual void dispatch_globals() = 0;

    virtual void dispatch_schedule() = 0;

public:
    CStyleBaseCodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<CodeSnippetFactory> library_snippet_factory = std::make_shared<CodeSnippetFactory>(),
        const std::string& externals_prefix = ""
    );

    bool generate() override;

    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ofstream& ofs_source) override;

    virtual void emit_capture_context_init(std::ostream& ofs_source) const = 0;
};

} // namespace codegen
} // namespace sdfg

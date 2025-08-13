#pragma once

#include "c_style_base_code_generator.h"
#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

namespace sdfg {
namespace codegen {

class CPPCodeGenerator : public CStyleBaseCodeGenerator {
private:
    CPPLanguageExtension language_extension_;

protected:
    void dispatch_includes() override;

    void dispatch_structures() override;

    void dispatch_globals() override;

    void dispatch_schedule() override;

    LanguageExtension& language_extension() override { return language_extension_; }

public:
    explicit CPPCodeGenerator(
        StructuredSDFG& sdfg,
        InstrumentationPlan& instrumentation_plan,
        bool capture_args_results = false,
        const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths = nullptr
    )
        : CStyleBaseCodeGenerator(sdfg, instrumentation_plan, capture_args_results, output_and_header_paths),
          language_extension_(sdfg.externals()) {}

    std::string function_definition() override;

    void emit_capture_context_init(std::ostream& ofs_source) const override;
};

} // namespace codegen
} // namespace sdfg

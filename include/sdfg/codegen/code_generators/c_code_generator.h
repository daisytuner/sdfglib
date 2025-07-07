#pragma once

#include <memory>
#include <vector>

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"

namespace sdfg {
namespace codegen {

class CCodeGenerator : public CodeGenerator {
private:
    CLanguageExtension language_extension_;

protected:
    void dispatch_includes();

    void dispatch_structures();

    void dispatch_globals();

    void dispatch_schedule();

public:
    CCodeGenerator(
        StructuredSDFG& sdfg,
        InstrumentationStrategy instrumentation_strategy = InstrumentationStrategy::NONE,
        bool capture_args_results = false,
        const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths = nullptr
    );

    bool generate() override;

    std::string function_definition() override;

    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void emit_capture_context_init(std::ostream& ofs_source) const;

    void emit_arg_captures(std::ostream& ofs_source, const std::vector<CaptureVarPlan>& plan, bool after);
};

} // namespace codegen
} // namespace sdfg

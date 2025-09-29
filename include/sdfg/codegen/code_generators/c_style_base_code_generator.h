#pragma once

#include <memory>
#include <vector>

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
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
        InstrumentationPlan& instrumentation_plan,
        ArgCaptureType arg_capture_type = ARG_CAPTURE_NONE,
        const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths = nullptr
    );

    bool generate() override;

    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ofstream& ofs_source) override;

    virtual void emit_capture_context_init(std::ostream& ofs_source) const = 0;

    void emit_arg_captures(std::ostream& ofs_source, const std::vector<CaptureVarPlan>& plan, bool after);
};

} // namespace codegen
} // namespace sdfg

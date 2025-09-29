#pragma once

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

namespace sdfg {
namespace codegen {

class CUDACodeGenerator : public CodeGenerator {
private:
    CUDALanguageExtension language_extension_;

protected:
    void dispatch_includes();

    void dispatch_structures();

    void dispatch_globals();

    void dispatch_schedule();

public:
    CUDACodeGenerator(
        StructuredSDFG& sdfg,
        InstrumentationPlan& instrumentation_plan,
        ArgCaptureType arg_capture_type = ARG_CAPTURE_NONE,
        const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths = nullptr
    );

    bool generate() override;

    std::string function_definition() override;

    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ofstream& ofs_source) override;
};

} // namespace codegen
} // namespace sdfg

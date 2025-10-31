
#include "sdfg/codegen/code_generators/c_style_base_code_generator.h"

#include "sdfg/helpers/helpers.h"

namespace sdfg::codegen {

CStyleBaseCodeGenerator::CStyleBaseCodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan,
    const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths,
    const std::string& externals_prefix
)
    : CodeGenerator(
          sdfg, analysis_manager, instrumentation_plan, arg_capture_plan, output_and_header_paths, externals_prefix
      ) {
    if (sdfg.type() != FunctionType_CPU) {
        throw std::runtime_error("CStyleBaseCodeGenerator can only be used for CPU SDFGs");
    }
};

bool CStyleBaseCodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
};

bool CStyleBaseCodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_header(header_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_header.is_open()) {
        return false;
    }

    std::ofstream ofs_source(source_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_source.is_open()) {
        return false;
    }

    ofs_header << "#pragma once" << std::endl;
    ofs_header << this->includes_stream_.str() << std::endl;
    ofs_header << this->classes_stream_.str() << std::endl;
    ofs_header.close();

    ofs_source << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    ofs_source << this->globals_stream_.str() << std::endl;

    append_function_source(ofs_source);

    ofs_source.close();

    return true;
}

void CStyleBaseCodeGenerator::append_function_source(std::ofstream& ofs_source) {
    std::unique_ptr<std::vector<CaptureVarPlan>> capturePlan;
    if (!arg_capture_plan_.is_empty()) {
        this->emit_capture_context_init(ofs_source);
    }

    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;

    auto init_once = library_snippet_factory_.find(CODE_SNIPPET_INIT_ONCE);
    if (init_once != library_snippet_factory_.snippets().end()) {
        ofs_source << init_once->second.stream().str() << std::endl;
    }

    ofs_source << this->main_stream_.str() << std::endl;

    auto deinit_once = library_snippet_factory_.find(CODE_SNIPPET_DEINIT_ONCE);
    if (deinit_once != library_snippet_factory_.snippets().end()) {
        ofs_source << deinit_once->second.stream().str() << std::endl;
    }

    ofs_source << "}" << std::endl;
}

} // namespace sdfg::codegen


#include "sdfg/codegen/code_generators/c_style_base_code_generator.h"

#include "sdfg/helpers/helpers.h"

namespace sdfg::codegen {

CStyleBaseCodeGenerator::CStyleBaseCodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    InstrumentationPlan& instrumentation_plan,
    bool capture_args_results,
    const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths,
    const std::string& externals_prefix
)
    : CodeGenerator(
          sdfg, analysis_manager, instrumentation_plan, capture_args_results, output_and_header_paths, externals_prefix
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
    if (capture_args_results_) {
        capturePlan = create_capture_plans();
        if (capturePlan) {
            this->emit_capture_context_init(ofs_source);
        } else {
            DEBUG_PRINTLN("Cannot capture all args for SDFG '" << sdfg_.name() << "'. Skipping capture instrumentation!");
        }
    }

    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;

    if (capturePlan) {
        this->emit_arg_captures(ofs_source, *capturePlan, false);
    }

    auto init_once = library_snippet_factory_.find(CODE_SNIPPET_INIT_ONCE);
    if (init_once != library_snippet_factory_.snippets().end()) {
        ofs_source << init_once->second.stream().str() << std::endl;
    }

    ofs_source << this->main_stream_.str() << std::endl;

    auto deinit_once = library_snippet_factory_.find(CODE_SNIPPET_DEINIT_ONCE);
    if (deinit_once != library_snippet_factory_.snippets().end()) {
        ofs_source << deinit_once->second.stream().str() << std::endl;
    }

    if (capturePlan) {
        this->emit_arg_captures(ofs_source, *capturePlan, true);
    }

    ofs_source << "}" << std::endl;
}

void CStyleBaseCodeGenerator::
    emit_arg_captures(std::ostream& ofs_source, const std::vector<CaptureVarPlan>& plan, bool after) {
    std::string name = sdfg_.name();

    if (!after) {
        ofs_source << "const bool __daisy_cap_en = __daisy_capture_enter(__capture_ctx);" << std::endl;
    }

    const auto& args = sdfg_.arguments();
    const auto& exts = sdfg_.externals();

    // filter externals by removing functions
    std::vector<std::string> filtered_exts;
    for (const auto& ext_name : exts) {
        if (sdfg_.type(ext_name).type_id() != types::TypeID::Function) {
            filtered_exts.push_back(ext_name);
        }
    }

    ofs_source << "if (__daisy_cap_en) {" << std::endl;

    auto afterBoolStr = after ? "true" : "false";

    for (auto& varPlan : plan) {
        auto argIdx = varPlan.arg_idx;
        std::string argName;
        if (varPlan.is_external) {
            argName = filtered_exts[argIdx - args.size()];
            argName = this->externals_prefix_ + argName;
        } else {
            argName = args[argIdx];
        }

        if ((!after && varPlan.capture_input) || (after && varPlan.capture_output)) {
            switch (varPlan.type) {
                case CaptureVarType::CapRaw: {
                    ofs_source << "\t__daisy_capture_raw(" << "__capture_ctx, " << argIdx << ", " << "&" << argName
                               << ", " << "sizeof(" << argName << "), " << varPlan.inner_type << ", " << afterBoolStr
                               << ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap1D: {
                    ofs_source << "\t__daisy_capture_1d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                               << "sizeof(" << language_extension().primitive_type(varPlan.inner_type) << "), "
                               << varPlan.inner_type << ", " << language_extension().expression(varPlan.dim1) << ", "
                               << afterBoolStr << ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap2D: {
                    ofs_source << "\t__daisy_capture_2d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                               << "sizeof(" << language_extension().primitive_type(varPlan.inner_type) << "), "
                               << varPlan.inner_type << ", " << language_extension().expression(varPlan.dim1) << ", "
                               << language_extension().expression(varPlan.dim2) << ", " << afterBoolStr << ");"
                               << std::endl;
                    break;
                }
                case CaptureVarType::Cap3D: {
                    ofs_source << "\t__daisy_capture_3d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                               << "sizeof(" << language_extension().primitive_type(varPlan.inner_type) << "), "
                               << varPlan.inner_type << ", " << language_extension().expression(varPlan.dim1) << ", "
                               << language_extension().expression(varPlan.dim2) << ", "
                               << language_extension().expression(varPlan.dim3) << ", " << afterBoolStr << ");"
                               << std::endl;
                    break;
                }
                default: {
                    DEBUG_PRINTLN(
                        "Unknown capture type " << static_cast<int>(varPlan.type) << " for arg " << argIdx << " at "
                                                << (after ? "result" : "input") << " time"
                    );
                    break;
                }
            }
        }
    }

    if (after) {
        ofs_source << "\t__daisy_capture_end(__capture_ctx);" << std::endl;
    }

    ofs_source << "}" << std::endl;
}

} // namespace sdfg::codegen

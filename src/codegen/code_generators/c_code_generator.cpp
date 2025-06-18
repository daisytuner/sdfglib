#include "sdfg/codegen/code_generators/c_code_generator.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation.h"
#include "sdfg/codegen/instrumentation/outermost_loops_instrumentation.h"

namespace sdfg {
namespace codegen {

CCodeGenerator::CCodeGenerator(StructuredSDFG& sdfg,
                               InstrumentationStrategy instrumentation_strategy, bool capture_args_results)
    : CodeGenerator(sdfg, instrumentation_strategy, capture_args_results) {
    if (sdfg.type() != FunctionType_CPU) {
        throw std::runtime_error("CCodeGenerator can only be used for CPU SDFGs");
    }
};

bool CCodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
};

std::string CCodeGenerator::function_definition() {
    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : sdfg_.arguments()) {
        args.push_back(language_extension_.declaration(container, sdfg_.type(container)));
    }
    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");

    return "extern void " + sdfg_.name() + "(" + arglist.str() + ")";
};

void CCodeGenerator::emit_capture_context_init(std::ostream& ofs_source) const {
    std::string name = sdfg_.name();

    ofs_source << "static void* __capture_ctx;" << std::endl;
    ofs_source << "static void __attribute__((constructor(1000))) __capture_ctx_init(void) {" << std::endl;
    ofs_source << "\t__capture_ctx = __daisy_capture_init(\"" << name << "\");" << std::endl;
    ofs_source << "}" << std::endl;
    ofs_source << std::endl;
}

void CCodeGenerator::emit_arg_captures(std::ostream& ofs_source, const std::vector<CaptureVarPlan>& plan, bool after) {
    std::string name = sdfg_.name();

    if (!after) {
        ofs_source << "const bool __daisy_cap_en = __daisy_capture_enter(__capture_ctx);" << std::endl;
    }

    const auto& args = sdfg_.arguments();

    ofs_source << "if (__daisy_cap_en) {" << std::endl;

    auto afterBoolStr = after ? "true" : "false";

    for (auto& varPlan : plan) {
        auto argIdx = varPlan.arg_idx;

        if ((!after && varPlan.capture_input) || (after && varPlan.capture_output)) {
            switch (varPlan.type) {
                case CaptureVarType::CapRaw: {
                    ofs_source << "\t__daisy_capture_raw(" <<
                        "__capture_ctx, " <<
                        argIdx << ", " <<
                        "&" << args[argIdx] << ", " <<
                        "sizeof(" << args[argIdx] << "), " <<
                        varPlan.inner_type << ", " <<
                        afterBoolStr <<
                        ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap1D: {
                    ofs_source << "\t__daisy_capture_1d(" <<
                        "__capture_ctx, " <<
                        argIdx << ", " <<
                        args[argIdx] << ", " <<
                        "sizeof(" << language_extension_.primitive_type(varPlan.inner_type) << "), " <<
                        varPlan.inner_type << ", " <<
                        language_extension_.expression(varPlan.dim1) << ", " <<
                        afterBoolStr <<
                        ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap2D: {
                    ofs_source << "\t__daisy_capture_2d(" <<
                        "__capture_ctx, " <<
                        argIdx << ", " <<
                        args[argIdx] << ", " <<
                        "sizeof(" << language_extension_.primitive_type(varPlan.inner_type) <<"), " <<
                        varPlan.inner_type << ", " <<
                        language_extension_.expression(varPlan.dim1) << ", " <<
                        language_extension_.expression(varPlan.dim2) << ", " <<
                        afterBoolStr <<
                        ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap3D: {
                    ofs_source << "\t__daisy_capture_3d(" <<
                        "__capture_ctx, " <<
                        argIdx << ", " <<
                        args[argIdx] << ", " <<
                        "sizeof(" << language_extension_.primitive_type(varPlan.inner_type) << "), " <<
                        varPlan.inner_type << ", " <<
                        language_extension_.expression(varPlan.dim1) << ", " <<
                        language_extension_.expression(varPlan.dim2) << ", " <<
                        language_extension_.expression(varPlan.dim3) << ", " <<
                        afterBoolStr <<
                        ");" << std::endl;
                    break;
                }
                default:
                    std::cerr << "Unknown capture type " << static_cast<int>(varPlan.type) << " for arg " << argIdx << " at " << (after? "result" : "input") << " time" << std::endl;
                    break;
            }
        }
    }

    if (after) {
        ofs_source << "\t__daisy_capture_end(__capture_ctx);" << std::endl;
    }

    ofs_source << "}" << std::endl;
};

bool CCodeGenerator::as_source(const std::filesystem::path& header_path,
                               const std::filesystem::path& source_path,
                               const std::filesystem::path& library_path) {
    std::ofstream ofs_header(header_path, std::ofstream::out);
    if (!ofs_header.is_open()) {
        return false;
    }

    std::ofstream ofs_source(source_path, std::ofstream::out);
    if (!ofs_source.is_open()) {
        return false;
    }

    std::ofstream ofs_library(library_path, std::ofstream::out);
    if (!ofs_library.is_open()) {
        return false;
    }

    ofs_header << "#pragma once" << std::endl;
    ofs_header << this->includes_stream_.str() << std::endl;
    ofs_header << this->classes_stream_.str() << std::endl;
    ofs_header.close();

    ofs_source << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    ofs_source << this->globals_stream_.str() << std::endl;

    std::unique_ptr<std::vector<CaptureVarPlan>> capturePlan;
    if (capture_args_results_) {
        capturePlan = create_capture_plans();
        if (capturePlan) {
            this->emit_capture_context_init(ofs_source);
        } else {
            std::cerr << "Cannot capture all args for SDFG '" << sdfg_.name() << "'. Skpping capture instrumentation!" << std::endl;
        }
    }

    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;

    if (instrumentation_strategy_ != InstrumentationStrategy::NONE) {
        ofs_source << "__daisy_instrument_init();" << std::endl;
    }

    if (capturePlan) {
        this->emit_arg_captures(ofs_source, *capturePlan, false);
    }

    ofs_source << this->main_stream_.str() << std::endl;

    if (capturePlan) {
        this->emit_arg_captures(ofs_source, *capturePlan, true);
    }

    if (instrumentation_strategy_ != InstrumentationStrategy::NONE) {
        ofs_source << "__daisy_instrument_finalize();" << std::endl;
    }

    ofs_source << "}" << std::endl;
    ofs_source.close();

    auto library_content = this->library_stream_.str();
    if (library_content.empty()) {
        ofs_library.close();
        return true;
    }

    ofs_library << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    ofs_library << std::endl;

    ofs_library << library_content << std::endl;
    ofs_library.close();

    return true;
};

void CCodeGenerator::dispatch_includes() {
    this->includes_stream_ << "#include <math.h>" << std::endl;
    this->includes_stream_ << "#include <stdbool.h>" << std::endl;
    this->includes_stream_ << "#include <stdlib.h>" << std::endl;
    if (this->instrumentation_strategy_ != InstrumentationStrategy::NONE)
        this->includes_stream_ << "#include <daisy_rtl.h>" << std::endl;

    this->includes_stream_ << "#define __daisy_min(a,b) ((a)<(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_max(a,b) ((a)>(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_fma(a,b,c) a * b + c" << std::endl;
};

void CCodeGenerator::dispatch_structures() {
    // Forward declarations
    for (auto& structure : sdfg_.structures()) {
        this->classes_stream_ << "typedef struct " << structure << " " << structure << ";"
                              << std::endl;
    }

    // Generate topology-sorted structure definitions
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> structures_graph;
    typedef boost::graph_traits<structures_graph>::vertex_descriptor Vertex;
    std::vector<std::string> names;
    for (auto& structure : sdfg_.structures()) {
        names.push_back(structure);
    }
    structures_graph graph(names.size());

    for (auto& structure : names) {
        auto& definition = sdfg_.structure(structure);
        for (size_t i = 0; i < definition.num_members(); i++) {
            auto member_type = &definition.member_type(symbolic::integer(i));
            while (dynamic_cast<const types::Array*>(member_type)) {
                auto array_type = static_cast<const types::Array*>(member_type);
                member_type = &array_type->element_type();
            }

            if (auto member_structure = dynamic_cast<const sdfg::types::Structure*>(member_type)) {
                boost::add_edge(
                    std::find(names.begin(), names.end(), member_structure->name()) - names.begin(),
                    std::find(names.begin(), names.end(), structure) - names.begin(), graph);
            }
        }
    }

    std::list<Vertex> order;
    std::unordered_map<Vertex, boost::default_color_type> vertex_colors;
    boost::topological_sort(graph, std::back_inserter(order),
                            boost::color_map(boost::make_assoc_property_map(vertex_colors)));
    order.reverse();

    for (auto& structure_index : order) {
        std::string structure = names.at(structure_index);
        auto& definition = sdfg_.structure(structure);
        this->classes_stream_ << "typedef struct ";
        if (definition.is_packed()) {
            this->classes_stream_ << "__attribute__((packed)) ";
        }
        this->classes_stream_ << structure << std::endl;
        this->classes_stream_ << "{\n";

        for (size_t i = 0; i < definition.num_members(); i++) {
            auto& member_type = definition.member_type(symbolic::integer(i));
            if (auto pointer_type = dynamic_cast<const sdfg::types::Pointer*>(&member_type)) {
                if (dynamic_cast<const sdfg::types::Structure*>(&pointer_type->pointee_type())) {
                    this->classes_stream_ << "struct ";
                }
            }
            this->classes_stream_ << language_extension_.declaration("member_" + std::to_string(i),
                                                                     member_type, false, true);
            this->classes_stream_ << ";" << std::endl;
        }

        this->classes_stream_ << "} " << structure << ";" << std::endl;
    }
};

void CCodeGenerator::dispatch_globals() {
    for (auto& container : sdfg_.externals()) {
        this->globals_stream_ << "extern "
                              << language_extension_.declaration(container, sdfg_.type(container))
                              << ";" << std::endl;
    }
};

void CCodeGenerator::dispatch_schedule() {
    // Map external variables to internal variables
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_internal(container)) {
            continue;
        }
        std::string external_name =
            container.substr(0, container.length() - external_suffix.length());
        this->main_stream_ << language_extension_.declaration(container, sdfg_.type(container));
        this->main_stream_ << " = "
                           << language_extension_.type_cast("&" + external_name,
                                                            sdfg_.type(container));
        this->main_stream_ << ";" << std::endl;
    }

    // Declare transient containers
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_transient(container)) {
            continue;
        }

        std::string val =
            this->language_extension_.declaration(container, sdfg_.type(container), false, true);
        if (!val.empty()) {
            this->main_stream_ << val;
            this->main_stream_ << ";" << std::endl;
        }
    }

    // Add instrumentation
    auto instrumentation = create_instrumentation(instrumentation_strategy_, sdfg_);

    auto dispatcher = create_dispatcher(language_extension_, sdfg_, sdfg_.root(), *instrumentation);
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, this->library_stream_);
};

}  // namespace codegen
}  // namespace sdfg

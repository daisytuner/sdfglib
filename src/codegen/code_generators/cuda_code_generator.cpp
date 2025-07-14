#include "sdfg/codegen/code_generators/cuda_code_generator.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/instrumentation.h"
#include "sdfg/codegen/instrumentation/outermost_loops_instrumentation.h"

namespace sdfg {
namespace codegen {

CUDACodeGenerator::CUDACodeGenerator(
    StructuredSDFG& sdfg,
    InstrumentationStrategy instrumentation_strategy,
    bool capture_args_results,
    const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths
)
    : CodeGenerator(sdfg, instrumentation_strategy, capture_args_results, output_and_header_paths) {
    if (sdfg.type() != FunctionType_NV_GLOBAL) {
        throw std::runtime_error("CUDACodeGenerator can only be used for GPU SDFGs");
    }
    if (capture_args_results) {
        std::cerr << "CUDACodeGenerator does not support capturing args/results!";
    }
};

bool CUDACodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
};

std::string CUDACodeGenerator::function_definition() {
    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : sdfg_.arguments()) {
        args.push_back(language_extension_.declaration(container, sdfg_.type(container)));
    }
    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");

    return "extern \"C\" __global__ void " + sdfg_.name() + "(" + arglist.str() + ")";
};

bool CUDACodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_header(header_path, std::ofstream::out);
    if (!ofs_header.is_open()) {
        return false;
    }

    std::ofstream ofs_source(source_path, std::ofstream::out);
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
};

void CUDACodeGenerator::append_function_source(std::ofstream& ofs_source) {
    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;
    ofs_source << this->main_stream_.str() << std::endl;
    ofs_source << "}" << std::endl;
}

void CUDACodeGenerator::dispatch_includes() {
    this->includes_stream_ << "#define "
                           << "__DAISY_NVVM__" << std::endl;
    this->includes_stream_ << "#include "
                           << "\"daisyrtl.h\"" << std::endl;
    if (instrumentation_strategy_ != InstrumentationStrategy::NONE)
        this->includes_stream_ << "#include <daisy_rtl.h>" << std::endl;

    this->includes_stream_ << "#define __daisy_min(a,b) ((a)<(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_max(a,b) ((a)>(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_fma(a,b,c) a * b + c" << std::endl;
};

void CUDACodeGenerator::dispatch_structures() {
    // Forward declarations
    for (auto& structure : sdfg_.structures()) {
        this->classes_stream_ << "struct " << structure << ";" << std::endl;
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
                    std::find(names.begin(), names.end(), structure) - names.begin(),
                    graph
                );
            }
        }
    }

    std::list<Vertex> order;
    std::unordered_map<Vertex, boost::default_color_type> vertex_colors;
    boost::topological_sort(
        graph, std::back_inserter(order), boost::color_map(boost::make_assoc_property_map(vertex_colors))
    );
    order.reverse();

    for (auto& structure_index : order) {
        std::string structure = names.at(structure_index);
        auto& definition = sdfg_.structure(structure);
        this->classes_stream_ << "struct ";
        if (definition.is_packed()) {
            this->classes_stream_ << "__attribute__((packed)) ";
        }
        this->classes_stream_ << structure << std::endl;
        this->classes_stream_ << "{\n";

        for (size_t i = 0; i < definition.num_members(); i++) {
            auto& member_type = definition.member_type(symbolic::integer(i));
            if (dynamic_cast<const sdfg::types::Structure*>(&member_type)) {
                this->classes_stream_ << "struct ";
            }
            this->classes_stream_
                << language_extension_.declaration("member_" + std::to_string(i), member_type, false, true);
            this->classes_stream_ << ";" << std::endl;
        }

        this->classes_stream_ << "};" << std::endl;
    }
};

void CUDACodeGenerator::dispatch_globals() {
    for (auto& container : sdfg_.externals()) {
        auto& type = sdfg_.type(container);
        if (type.storage_type() == types::StorageType_NV_Global) {
            this->globals_stream_ << "extern " << language_extension_.declaration(container, type) << ";" << std::endl;
        }
        if (type.storage_type() == types::StorageType_NV_Constant) {
            assert(type.initializer().empty());
            this->globals_stream_ << "__constant__ " << language_extension_.declaration(container, type, true) << ";"
                                  << std::endl;
        }
    }
};

void CUDACodeGenerator::dispatch_schedule() {
    // Declare shared memory
    for (auto& container : sdfg_.externals()) {
        auto& type = sdfg_.type(container);
        if (type.storage_type() == types::StorageType_NV_Shared) {
            this->main_stream_ << language_extension_.declaration(container, sdfg_.type(container)) << ";" << std::endl;
        }
    }

    // Map external variables to internal variables
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_internal(container)) {
            continue;
        }

        std::string external_name = container.substr(0, container.length() - external_suffix.length());
        this->main_stream_ << language_extension_.declaration(container, sdfg_.type(container));
        this->main_stream_ << " = "
                           << "&" << external_name;
        this->main_stream_ << ";" << std::endl;
    }

    // Declare transient containers
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_transient(container)) {
            continue;
        }

        std::string val = this->language_extension_.declaration(container, sdfg_.type(container), false, true);
        if (!val.empty()) {
            this->main_stream_ << val;
            this->main_stream_ << ";" << std::endl;
        }
    }

    // Add instrumentation
    auto instrumentation = create_instrumentation(instrumentation_strategy_, sdfg_);

    auto dispatcher = create_dispatcher(language_extension_, sdfg_, sdfg_.root(), *instrumentation);
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, this->library_snippet_factory_);
};

} // namespace codegen
} // namespace sdfg

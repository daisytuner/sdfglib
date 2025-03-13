#include "sdfg/codegen/code_generators/c_code_generator.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_factory.h"

namespace sdfg {
namespace codegen {

CCodeGenerator::CCodeGenerator(ConditionalSchedule& schedule, bool instrumented)
    : CodeGenerator(schedule, instrumented){

      };

bool CCodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
};

std::string CCodeGenerator::function_definition() {
    // Define SDFG as a function
    auto& function = this->schedule_.schedule(0).sdfg();

    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : function.arguments()) {
        args.push_back(language_extension_.declaration(container, function.type(container)));
    }
    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");

    return "extern void " + function.name() + "(" + arglist.str() + ")";
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
    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;

    if (instrumented_) {
        ofs_source << "__daisy_instrument_init();" << std::endl;
    }

    ofs_source << this->main_stream_.str() << std::endl;

    if (instrumented_) {
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

    ofs_library << "#undef HWY_TARGET_INCLUDE" << std::endl;
    ofs_library << "#define HWY_TARGET_INCLUDE " << library_path.filename() << std::endl;
    ofs_library << "#include <hwy/foreach_target.h>" << std::endl;
    ofs_library << "#include <hwy/highway.h>" << std::endl;
    ofs_library << "#include <hwy/contrib/math/math-inl.h>" << std::endl;
    ofs_library << std::endl;

    ofs_library << library_content << std::endl;
    ofs_library.close();

    return true;
};

void CCodeGenerator::dispatch_includes() {
    this->includes_stream_ << "#include <math.h>" << std::endl;
    this->includes_stream_ << "#include <stdbool.h>" << std::endl;
    this->includes_stream_ << "#include <stdlib.h>" << std::endl;

    this->includes_stream_ << "#define __daisy_min(a,b) ((a)<(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_max(a,b) ((a)>(b)?(a):(b))" << std::endl;
    this->includes_stream_ << "#define __daisy_fma(a,b,c) a * b + c" << std::endl;
};

void CCodeGenerator::dispatch_structures() {
    auto& function = this->schedule_.schedule(0).sdfg();

    // Forward declarations
    for (auto& structure : function.structures()) {
        this->classes_stream_ << "typedef struct " << structure << " " << structure << ";"
                              << std::endl;
    }

    // Generate topology-sorted structure definitions
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> structures_graph;
    typedef boost::graph_traits<structures_graph>::vertex_descriptor Vertex;
    std::vector<std::string> names;
    for (auto& structure : function.structures()) {
        names.push_back(structure);
    }
    structures_graph graph(names.size());

    for (auto& structure : names) {
        auto& definition = function.structure(structure);
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
        auto& definition = function.structure(structure);
        this->classes_stream_ << "typedef struct " << structure << std::endl;
        this->classes_stream_ << "{\n";

        for (size_t i = 0; i < definition.num_members(); i++) {
            auto& member_type = definition.member_type(symbolic::integer(i));
            if (auto pointer_type = dynamic_cast<const sdfg::types::Pointer*>(&member_type)) {
                if (auto struct_type = dynamic_cast<const sdfg::types::Structure*>(
                        &pointer_type->pointee_type())) {
                    this->classes_stream_ << "struct ";
                }
            }
            this->classes_stream_ << language_extension_.declaration("member_" + std::to_string(i),
                                                                     member_type);
            this->classes_stream_ << ";" << std::endl;
        }

        this->classes_stream_ << "} " << structure << ";" << std::endl;
    }
};

void CCodeGenerator::dispatch_globals() {
    auto& function = this->schedule_.schedule(0).sdfg();
    for (auto& container : function.externals()) {
        this->globals_stream_ << "extern "
                              << language_extension_.declaration(container,
                                                                 function.type(container))
                              << ";" << std::endl;
    }
};

void CCodeGenerator::dispatch_schedule() {
    // Map external variables to internal variables
    auto& function = this->schedule_.schedule(0).sdfg();
    for (auto& container : function.containers()) {
        if (!function.is_internal(container)) {
            continue;
        }
        std::string external_name =
            container.substr(0, container.length() - external_suffix.length());
        this->main_stream_ << language_extension_.declaration(container, function.type(container));
        this->main_stream_ << " = "
                           << "&" << external_name;
        this->main_stream_ << ";" << std::endl;
    }

    for (size_t i = 0; i < schedule_.size(); i++) {
        if (i == 0) {
            this->main_stream_ << "if (" << language_extension_.expression(schedule_.condition(i))
                               << ") {\n";

            auto& function_i = schedule_.schedule(i).builder().subject();
            auto dispatcher = create_dispatcher(language_extension_, schedule_.schedule(i),
                                                function_i.root(), this->instrumented_);
            dispatcher->dispatch(this->main_stream_, this->globals_stream_, this->library_stream_);

            this->main_stream_ << "}\n";
        } else {
            this->main_stream_ << "else if ("
                               << language_extension_.expression(schedule_.condition(i)) << ") {\n";

            auto& function_i = schedule_.schedule(i).builder().subject();
            auto dispatcher = create_dispatcher(language_extension_, schedule_.schedule(i),
                                                function_i.root(), this->instrumented_);
            dispatcher->dispatch(this->main_stream_, this->globals_stream_, this->library_stream_);

            this->main_stream_ << "}\n";
        }
    }
};

}  // namespace codegen
}  // namespace sdfg

#include "sdfg/codegen/code_generators/c_code_generator.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"

namespace sdfg {
namespace codegen {

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

void CCodeGenerator::dispatch_includes() {
    this->includes_stream_ << "#include <math.h>" << std::endl;
    this->includes_stream_ << "#include <cblas.h>" << std::endl;
    this->includes_stream_ << "#include <stdbool.h>" << std::endl;
    this->includes_stream_ << "#include <stdlib.h>" << std::endl;
    this->includes_stream_ << "#include <daisy_rtl/daisy_rtl.h>" << std::endl;
};

void CCodeGenerator::dispatch_structures() {
    // Forward declarations
    for (auto& structure : sdfg_.structures()) {
        this->classes_stream_ << "typedef struct " << structure << " " << structure << ";" << std::endl;
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
            this->classes_stream_
                << language_extension_.declaration("member_" + std::to_string(i), member_type, false, true);
            this->classes_stream_ << ";" << std::endl;
        }

        this->classes_stream_ << "} " << structure << ";" << std::endl;
    }
};

void CCodeGenerator::dispatch_globals() {
    for (auto& container : sdfg_.externals()) {
        this->globals_stream_ << "extern " << language_extension_.declaration(container, sdfg_.type(container)) << ";"
                              << std::endl;
    }
};

void CCodeGenerator::dispatch_schedule() {
    // Map external variables to internal variables
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_internal(container)) {
            continue;
        }
        std::string external_name = container.substr(0, container.length() - external_suffix.length());
        this->main_stream_ << language_extension_.declaration(container, sdfg_.type(container));
        this->main_stream_ << " = " << language_extension_.type_cast("&" + external_name, sdfg_.type(container));
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

    auto dispatcher = create_dispatcher(language_extension_, sdfg_, sdfg_.root(), instrumentation_plan_);
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, this->library_snippet_factory_);
};

} // namespace codegen
} // namespace sdfg

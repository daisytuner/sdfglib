#include "sdfg/codegen/code_generators/cpp_code_generator.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"

namespace sdfg {
namespace codegen {

std::string CPPCodeGenerator::function_definition() {
    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : sdfg_.arguments()) {
        args.push_back(language_extension_.declaration(container, sdfg_.type(container)));
    }
    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");
    std::string arglist_str = arglist.str();
    if (arglist_str.empty()) {
        arglist_str = "void";
    }

    return "extern \"C\" " + this->language_extension_.declaration("", sdfg_.return_type()) + sdfg_.name() + "(" +
           arglist_str + ")";
};

void CPPCodeGenerator::emit_capture_context_init(std::ostream& ofs_source) const {
    std::string name = sdfg_.name();
    std::string arg_capture_path = sdfg_.metadata().at("arg_capture_path");

    ofs_source << "static __daisy_capture_t* __capture_ctx;" << std::endl;
    ofs_source << "static void __attribute__((constructor(1000))) __capture_ctx_init(void) {" << std::endl;
    ofs_source << "\t__capture_ctx = __daisy_capture_init(\"" << name << "\", \"" << arg_capture_path << "\");" << std::endl;
    ofs_source << "}" << std::endl;
    ofs_source << std::endl;
}

void CPPCodeGenerator::dispatch_includes() {
    this->includes_stream_ << "#include <alloca.h>" << std::endl;
    this->includes_stream_ << "#include <cmath>" << std::endl;
    this->includes_stream_ << "#include <cstdio>" << std::endl;
    this->includes_stream_ << "#include <cstdlib>" << std::endl;
    this->includes_stream_ << "#include <cstring>" << std::endl;
    this->includes_stream_ << "#include <cblas.h>" << std::endl;
    this->includes_stream_ << "#include <daisy_rtl/daisy_rtl.h>" << std::endl;
};

void CPPCodeGenerator::dispatch_structures() {
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

void CPPCodeGenerator::dispatch_globals() {
    // Declare globals
    const std::unordered_set<std::string> reserved_symbols = {"stderr", "stdin", "stdout"};
    for (auto& container : sdfg_.externals()) {
        // Function declarations
        if (dynamic_cast<const types::Function*>(&sdfg_.type(container))) {
            continue;
        }
        // Reserved symbols
        if (reserved_symbols.find(container) != reserved_symbols.end()) {
            continue;
        }

        // Other types must be pointers
        auto& type = dynamic_cast<const types::Pointer&>(sdfg_.type(container));
        assert(type.has_pointee_type() && "Externals must have a pointee type");
        auto& base_type = type.pointee_type();

        if (sdfg_.linkage_type(container) == LinkageType_External) {
            this->globals_stream_ << "extern " << language_extension_.declaration(container, base_type) << ";"
                                  << std::endl;
        } else {
            this->globals_stream_ << "static " << language_extension_.declaration(container, base_type);
            if (!type.initializer().empty()) {
                this->globals_stream_ << " = " << type.initializer();
            }
            this->globals_stream_ << ";" << std::endl;
        }
    }
};

void CPPCodeGenerator::dispatch_schedule() {
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

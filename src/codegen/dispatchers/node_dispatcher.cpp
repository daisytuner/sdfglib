#include "sdfg/codegen/dispatchers/node_dispatcher.h"

namespace sdfg {
namespace codegen {

NodeDispatcher::NodeDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                               structured_control_flow::ControlFlowNode& node,
                               Instrumentation& instrumentation)
    : node_(node),
      language_extension_(language_extension),
      schedule_(schedule),
      instrumentation_(instrumentation) {};

bool NodeDispatcher::begin_node(PrettyPrinter& stream) {
    auto& sdfg = schedule_.sdfg();

    // Declare the transient variables
    bool applied = false;
    for (auto& container : sdfg.containers()) {
        if (!sdfg.is_transient(container)) {
            continue;
        }

        if (schedule_.allocation_lifetime(container) == &node_) {
            if (!applied) {
                applied = true;
                stream << "{" << std::endl;
                stream.setIndent(stream.indent() + 4);
            }
            if (schedule_.allocation_type(container) == AllocationType::DECLARE) {
                std::string val =
                    this->language_extension_.declaration(container, sdfg.type(container));
                if (!val.empty()) {
                    stream << val;
                    stream << ";" << std::endl;
                }
            } else {
                std::string val =
                    this->language_extension_.allocation(container, sdfg.type(container));
                if (!val.empty()) {
                    stream << val;
                    stream << ";" << std::endl;
                }
            }
        }
    }

    return applied;
};

void NodeDispatcher::end_node(PrettyPrinter& stream, bool applied) {
    auto& sdfg = schedule_.sdfg();

    if (applied) {
        for (auto& container : sdfg.containers()) {
            if (!sdfg.is_transient(container)) {
                continue;
            }
            if (schedule_.allocation_lifetime(container) == &node_) {
                if (schedule_.allocation_type(container) == AllocationType::ALLOCATE) {
                    auto& type = sdfg.type(container);
                    std::string val = this->language_extension_.deallocation(container, type);
                    if (!val.empty()) {
                        stream << val;
                        stream << ";" << std::endl;
                    }
                }
            }
        }

        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }
};

void NodeDispatcher::dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                              PrettyPrinter& library_stream) {
    bool applied = begin_node(main_stream);

    if (this->instrumentation_.should_instrument(node_)) {
        this->instrumentation_.begin_instrumentation(node_, main_stream);
    }

    dispatch_node(main_stream, globals_stream, library_stream);

    if (this->instrumentation_.should_instrument(node_)) {
        this->instrumentation_.end_instrumentation(node_, main_stream);
    }

    end_node(main_stream, applied);
};

}  // namespace codegen
}  // namespace sdfg

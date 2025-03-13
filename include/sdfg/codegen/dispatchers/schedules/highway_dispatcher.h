#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/data_flow/tasklet.h"

namespace sdfg {
namespace codegen {

static size_t constant_counter = 0;

class HighwayDispatcher : public NodeDispatcher {
   private:
    structured_control_flow::For& node_;

    // Iterators
    const symbolic::Symbol& indvar_;
    std::unordered_set<std::string> moving_symbols_;
    std::unordered_set<std::string> declared_vectors_;

    // Vector types
    size_t bit_width_;

    void dispatch_declarations(const structured_control_flow::ControlFlowNode& node,
                               PrettyPrinter& stream);

    void dispatch_declarations_vector(const structured_control_flow::ControlFlowNode& node,
                                      PrettyPrinter& stream);

    /**** Dispatchers for SDFG elements ****/

    void dispatch(structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream);

    void dispatch(structured_control_flow::Sequence& node, PrettyPrinter& stream);

    void dispatch(structured_control_flow::Block& node, PrettyPrinter& stream);

    void dispatch_tasklet(const data_flow::DataFlowGraph& graph, const data_flow::Tasklet& tasklet,
                          PrettyPrinter& stream);

    std::string indirection_to_cpp(const symbolic::Expression& access, PrettyPrinter& stream);

    static std::string function_name(const structured_control_flow::ControlFlowNode& node);

    static std::vector<std::string> function_arguments(
        Schedule& schedule, structured_control_flow::ControlFlowNode& node);

    std::string condition_to_simd_instruction(const symbolic::Expression& condition);

   public:
    HighwayDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                      structured_control_flow::For& node, bool instrumented);

    void dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                  PrettyPrinter& library_stream) override;

    void dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                       PrettyPrinter& library_stream) override;

    static std::string tasklet_to_simd_instruction(data_flow::TaskletCode c, std::string vec_type);
};

}  // namespace codegen
}  // namespace sdfg

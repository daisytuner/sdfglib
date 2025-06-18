#pragma once

#include <memory>
#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/sdfg.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/scalar.h"

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

class StructuredSDFGBuilder : public FunctionBuilder {
   private:
    std::unique_ptr<StructuredSDFG> structured_sdfg_;

    std::unordered_set<const control_flow::State*> determine_loop_nodes(
        const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const;

    const control_flow::State* find_end_of_if_else(
        const SDFG& sdfg, const State* current, std::vector<const InterstateEdge*>& out_edges,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>&
            pdom_tree);

    void traverse(const SDFG& sdfg);

    void traverse_with_loop_detection(
        const SDFG& sdfg, Sequence& scope, const State* current, const State* end,
        const std::unordered_set<const InterstateEdge*>& continues,
        const std::unordered_set<const InterstateEdge*>& breaks,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
        std::unordered_set<const control_flow::State*>& visited);

    void traverse_without_loop_detection(
        const SDFG& sdfg, Sequence& scope, const State* current, const State* end,
        const std::unordered_set<const InterstateEdge*>& continues,
        const std::unordered_set<const InterstateEdge*>& breaks,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
        std::unordered_set<const control_flow::State*>& visited);

    void add_dataflow(const data_flow::DataFlowGraph& from, Block& to);

   protected:
    Function& function() const override;

   public:
    StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg);

    StructuredSDFGBuilder(const std::string& name, FunctionType type);

    StructuredSDFGBuilder(const SDFG& sdfg);

    StructuredSDFG& subject() const;

    std::unique_ptr<StructuredSDFG> move();

    Element* find_element_by_id(const size_t& element_id) const;

    Sequence& add_sequence(Sequence& parent,
                           const sdfg::control_flow::Assignments& assignments = {},
                           const DebugInfo& debug_info = DebugInfo());

    std::pair<Sequence&, Transition&> add_sequence_before(
        Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info = DebugInfo());

    void remove_child(Sequence& parent, size_t i);

    void remove_child(Sequence& parent, ControlFlowNode& child);

    void insert_children(Sequence& parent, Sequence& other, size_t i);

    Block& add_block(Sequence& parent, const sdfg::control_flow::Assignments& assignments = {},
                     const DebugInfo& debug_info = DebugInfo());

    Block& add_block(Sequence& parent, const data_flow::DataFlowGraph& data_flow_graph,
                     const sdfg::control_flow::Assignments& assignments = {},
                     const DebugInfo& debug_info = DebugInfo());

    std::pair<Block&, Transition&> add_block_before(Sequence& parent, ControlFlowNode& block,
                                                    const DebugInfo& debug_info = DebugInfo());

    std::pair<Block&, Transition&> add_block_before(Sequence& parent, ControlFlowNode& block,
                                                    data_flow::DataFlowGraph& data_flow_graph,
                                                    const DebugInfo& debug_info = DebugInfo());

    std::pair<Block&, Transition&> add_block_after(Sequence& parent, ControlFlowNode& block,
                                                   const DebugInfo& debug_info = DebugInfo());

    std::pair<Block&, Transition&> add_block_after(Sequence& parent, ControlFlowNode& block,
                                                   data_flow::DataFlowGraph& data_flow_graph,
                                                   const DebugInfo& debug_info = DebugInfo());

    For& add_for(Sequence& parent, const symbolic::Symbol& indvar,
                 const symbolic::Condition& condition, const symbolic::Expression& init,
                 const symbolic::Expression& update,
                 const sdfg::control_flow::Assignments& assignments = {},
                 const DebugInfo& debug_info = DebugInfo());

    std::pair<For&, Transition&> add_for_before(Sequence& parent, ControlFlowNode& block,
                                                const symbolic::Symbol& indvar,
                                                const symbolic::Condition& condition,
                                                const symbolic::Expression& init,
                                                const symbolic::Expression& update,
                                                const DebugInfo& debug_info = DebugInfo());

    std::pair<For&, Transition&> add_for_after(Sequence& parent, ControlFlowNode& block,
                                               const symbolic::Symbol& indvar,
                                               const symbolic::Condition& condition,
                                               const symbolic::Expression& init,
                                               const symbolic::Expression& update,
                                               const DebugInfo& debug_info = DebugInfo());

    IfElse& add_if_else(Sequence& parent, const DebugInfo& debug_info = DebugInfo());

    IfElse& add_if_else(Sequence& parent, const sdfg::control_flow::Assignments& assignments,
                        const DebugInfo& debug_info = DebugInfo());

    std::pair<IfElse&, Transition&> add_if_else_before(Sequence& parent, ControlFlowNode& block,
                                                       const DebugInfo& debug_info = DebugInfo());

    Sequence& add_case(IfElse& scope, const sdfg::symbolic::Condition cond,
                       const DebugInfo& debug_info = DebugInfo());

    void remove_case(IfElse& scope, size_t i, const DebugInfo& debug_info = DebugInfo());

    While& add_while(Sequence& parent, const sdfg::control_flow::Assignments& assignments = {},
                     const DebugInfo& debug_info = DebugInfo());

    Continue& add_continue(Sequence& parent, const DebugInfo& debug_info = DebugInfo());

    Continue& add_continue(Sequence& parent, const sdfg::control_flow::Assignments& assignments,
                           const DebugInfo& debug_info = DebugInfo());

    Break& add_break(Sequence& parent, const DebugInfo& debug_info = DebugInfo());

    Break& add_break(Sequence& parent, const sdfg::control_flow::Assignments& assignments,
                     const DebugInfo& debug_info = DebugInfo());

    Map& add_map(Sequence& parent, const symbolic::Symbol& indvar,
                 const symbolic::Expression& num_iterations, const ScheduleType& schedule_type,
                 const sdfg::control_flow::Assignments& assignments = {},
                 const DebugInfo& debug_info = DebugInfo());

    Return& add_return(Sequence& parent, const sdfg::control_flow::Assignments& assignments = {},
                       const DebugInfo& debug_info = DebugInfo());

    For& convert_while(Sequence& parent, While& loop, const symbolic::Symbol& indvar,
                       const symbolic::Condition& condition, const symbolic::Expression& init,
                       const symbolic::Expression& update);

    Map& convert_for(Sequence& parent, For& loop, const symbolic::Expression& num_iterations);

    void clear_sequence(Sequence& parent);

    Sequence& parent(const ControlFlowNode& node);

    /***** Section: Dataflow Graph *****/

    data_flow::AccessNode& add_access(structured_control_flow::Block& block,
                                      const std::string& data,
                                      const DebugInfo& debug_info = DebugInfo());

    data_flow::Tasklet& add_tasklet(
        structured_control_flow::Block& block, const data_flow::TaskletCode code,
        const std::pair<std::string, sdfg::types::Scalar>& output,
        const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
        const DebugInfo& debug_info = DebugInfo());

    data_flow::Memlet& add_memlet(structured_control_flow::Block& block,
                                  data_flow::DataFlowNode& src, const std::string& src_conn,
                                  data_flow::DataFlowNode& dst, const std::string& dst_conn,
                                  const data_flow::Subset& subset,
                                  const DebugInfo& debug_info = DebugInfo());

    template <typename T, typename... Args>
    data_flow::LibraryNode& add_library_node(structured_control_flow::Block& block,
                                             const data_flow::LibraryNodeCode code,
                                             const std::vector<std::string>& outputs,
                                             const std::vector<std::string>& inputs,
                                             const bool side_effect = true,
                                             const DebugInfo& debug_info = DebugInfo(),
                                             Args... arguments) {
        static_assert(std::is_base_of<data_flow::LibraryNode, T>::value,
                      "T must be a subclass of data_flow::LibraryNode");

        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node = std::unique_ptr<T>(new T(this->new_element_id(), debug_info, vertex, dataflow,
                                             code, outputs, inputs, side_effect, arguments...));
        auto res = dataflow.nodes_.insert({vertex, std::move(node)});

        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    };

    void remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge);

    void remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::Tasklet& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node);

    data_flow::AccessNode& symbolic_expression_to_dataflow(structured_control_flow::Block& parent,
                                                           const symbolic::Expression& expr);
};

}  // namespace builder
}  // namespace sdfg

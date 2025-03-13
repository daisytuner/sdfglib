#pragma once

#include <list>
#include <memory>
#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/sdfg.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
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

    void traverse(const SDFG& sdfg);

    void traverse(const SDFG& sdfg, std::unordered_map<const State*, const State*>& pdom_tree,
                  std::list<const InterstateEdge*>& back_edges, Sequence& scope, const State* begin,
                  const State* end,
                  std::unordered_map<const InterstateEdge*, const While*>& active_continues,
                  std::unordered_map<const InterstateEdge*, const While*>& active_breaks,
                  bool skip_loop_detection);

   protected:
    Function& function() const override;

   public:
    StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg);

    StructuredSDFGBuilder(const std::string& name);

    StructuredSDFGBuilder(const SDFG& sdfg);

    StructuredSDFG& subject() const;

    std::unique_ptr<StructuredSDFG> move();

    Sequence& add_sequence(Sequence& parent, const sdfg::symbolic::Assignments& assignments = {},
                           const DebugInfo& debug_info = DebugInfo());

    std::pair<Sequence&, Transition&> add_sequence_before(
        Sequence& parent, ControlFlowNode& block, const DebugInfo& debug_info = DebugInfo());

    void remove_child(Sequence& parent, size_t i);

    void remove_child(Sequence& parent, ControlFlowNode& child);

    void insert_children(Sequence& parent, Sequence& other, size_t i);

    Block& add_block(Sequence& parent, const sdfg::symbolic::Assignments& assignments = {},
                     const DebugInfo& debug_info = DebugInfo());

    Block& add_block(Sequence& parent, const data_flow::DataFlowGraph& data_flow_graph,
                     const sdfg::symbolic::Assignments& assignments = {},
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
                 const sdfg::symbolic::Assignments& assignments = {},
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

    IfElse& add_if_else(Sequence& parent, const sdfg::symbolic::Assignments& assignments,
                        const DebugInfo& debug_info = DebugInfo());

    std::pair<IfElse&, Transition&> add_if_else_before(Sequence& parent, ControlFlowNode& block,
                                                       const DebugInfo& debug_info = DebugInfo());

    Sequence& add_case(IfElse& scope, const sdfg::symbolic::Condition cond,
                       const DebugInfo& debug_info = DebugInfo());

    void remove_case(IfElse& scope, size_t i, const DebugInfo& debug_info = DebugInfo());

    While& add_while(Sequence& parent, const sdfg::symbolic::Assignments& assignments = {},
                     const DebugInfo& debug_info = DebugInfo());

    Kernel& add_kernel(
        Sequence& parent, const std::string& suffix, const DebugInfo& debug_info = DebugInfo(),
        const symbolic::Expression& gridDim_x_init = symbolic::symbol("gridDim.x"),
        const symbolic::Expression& gridDim_y_init = symbolic::symbol("gridDim.y"),
        const symbolic::Expression& gridDim_z_init = symbolic::symbol("gridDim.z"),
        const symbolic::Expression& blockDim_x_init = symbolic::symbol("blockDim.x"),
        const symbolic::Expression& blockDim_y_init = symbolic::symbol("blockDim.y"),
        const symbolic::Expression& blockDim_z_init = symbolic::symbol("blockDim.z"),
        const symbolic::Expression& blockIdx_x_init = symbolic::symbol("blockIdx.x"),
        const symbolic::Expression& blockIdx_y_init = symbolic::symbol("blockIdx.y"),
        const symbolic::Expression& blockIdx_z_init = symbolic::symbol("blockIdx.z"),
        const symbolic::Expression& threadIdx_x_init = symbolic::symbol("threadIdx.x"),
        const symbolic::Expression& threadIdx_y_init = symbolic::symbol("threadIdx.y"),
        const symbolic::Expression& threadIdx_z_init = symbolic::symbol("threadIdx.z"));

    Continue& add_continue(Sequence& parent, const While& loop,
                           const DebugInfo& debug_info = DebugInfo());

    Continue& add_continue(Sequence& parent, const While& loop,
                           const sdfg::symbolic::Assignments& assignments,
                           const DebugInfo& debug_info = DebugInfo());

    Break& add_break(Sequence& parent, const While& loop,
                     const DebugInfo& debug_info = DebugInfo());

    Break& add_break(Sequence& parent, const While& loop,
                     const sdfg::symbolic::Assignments& assignments,
                     const DebugInfo& debug_info = DebugInfo());

    Return& add_return(Sequence& parent, const sdfg::symbolic::Assignments& assignments = {},
                       const DebugInfo& debug_info = DebugInfo());

    For& convert_while(Sequence& parent, While& loop, const symbolic::Symbol& indvar,
                       const symbolic::Condition& condition, const symbolic::Expression& init,
                       const symbolic::Expression& update);

    void clear_sequence(Sequence& parent);

    Sequence& parent(const ControlFlowNode& node);

    Kernel& convert_into_kernel();

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

    data_flow::LibraryNode& add_library_node(
        structured_control_flow::Block& block, const data_flow::LibraryNodeType& call,
        const std::vector<std::pair<std::string, sdfg::types::Scalar>>& outputs,
        const std::vector<std::pair<std::string, sdfg::types::Scalar>>& inputs,
        const bool has_side_effect = true, const DebugInfo& debug_info = DebugInfo());

    void remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge);

    void remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::Tasklet& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node);

    data_flow::AccessNode& symbolic_expression_to_dataflow(structured_control_flow::Block& parent,
                                                           const symbolic::Expression& expr);
};

}  // namespace builder
}  // namespace sdfg

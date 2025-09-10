#pragma once

#include <memory>
#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/debug_info.h"
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

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

class StructuredSDFGBuilder : public FunctionBuilder {
private:
    std::unique_ptr<StructuredSDFG> structured_sdfg_;

    std::unordered_set<const control_flow::State*>
    determine_loop_nodes(const SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const;

    const control_flow::State* find_end_of_if_else(
        const SDFG& sdfg,
        const State* current,
        std::vector<const InterstateEdge*>& out_edges,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree
    );

    void traverse(const SDFG& sdfg);

    void traverse_with_loop_detection(
        const SDFG& sdfg,
        Sequence& scope,
        const State* current,
        const State* end,
        const std::unordered_set<const InterstateEdge*>& continues,
        const std::unordered_set<const InterstateEdge*>& breaks,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
        std::unordered_set<const control_flow::State*>& visited
    );

    void traverse_without_loop_detection(
        const SDFG& sdfg,
        Sequence& scope,
        const State* current,
        const State* end,
        const std::unordered_set<const InterstateEdge*>& continues,
        const std::unordered_set<const InterstateEdge*>& breaks,
        const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
        std::unordered_set<const control_flow::State*>& visited
    );

    void add_dataflow(const data_flow::DataFlowGraph& from, Block& to);

    DebugInfoRegion fill_debug_info(const DebugInfos& debug_info_elements);

protected:
    Function& function() const override;

public:
    StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg);

    StructuredSDFGBuilder(const std::string& name, FunctionType type);

    StructuredSDFGBuilder(const SDFG& sdfg);

    StructuredSDFG& subject() const;

    std::unique_ptr<StructuredSDFG> move();

    Element* find_element_by_id(const size_t& element_id) const;

    Sequence& add_sequence(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<Sequence&, Transition&>
    add_sequence_before(Sequence& parent, ControlFlowNode& block, const DebugInfos& debug_info_elements = {});

    void remove_child(Sequence& parent, size_t i);

    void remove_child(Sequence& parent, ControlFlowNode& child);

    void insert_children(Sequence& parent, Sequence& other, size_t i);

    void insert(ControlFlowNode& node, Sequence& source, Sequence& target, const DebugInfos& debug_info_elements = {});

    Block& add_block(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    Block& add_block(
        Sequence& parent,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<Block&, Transition&>
    add_block_before(Sequence& parent, ControlFlowNode& block, const DebugInfos& debug_info_elements = {});

    std::pair<Block&, Transition&> add_block_before(
        Sequence& parent,
        ControlFlowNode& block,
        data_flow::DataFlowGraph& data_flow_graph,
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<Block&, Transition&>
    add_block_after(Sequence& parent, ControlFlowNode& block, const DebugInfos& debug_info_elements = {});

    std::pair<Block&, Transition&> add_block_after(
        Sequence& parent,
        ControlFlowNode& block,
        data_flow::DataFlowGraph& data_flow_graph,
        const DebugInfos& debug_info_elements = {}
    );

    For& add_for(
        Sequence& parent,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<For&, Transition&> add_for_before(
        Sequence& parent,
        ControlFlowNode& block,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<For&, Transition&> add_for_after(
        Sequence& parent,
        ControlFlowNode& block,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const DebugInfos& debug_info_elements = {}
    );

    IfElse& add_if_else(Sequence& parent, const DebugInfos& debug_info_elements = {});

    IfElse& add_if_else(
        Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfos& debug_info_elements = {}
    );

    std::pair<IfElse&, Transition&>
    add_if_else_before(Sequence& parent, ControlFlowNode& block, const DebugInfos& debug_info_elements = {});

    Sequence& add_case(IfElse& scope, const sdfg::symbolic::Condition cond, const DebugInfos& debug_info_elements = {});

    void remove_case(IfElse& scope, size_t i, const DebugInfos& debug_info_elements = {});

    While& add_while(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    Continue& add_continue(Sequence& parent, const DebugInfos& debug_info_elements = {});

    Continue& add_continue(
        Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfos& debug_info_elements = {}
    );

    Break& add_break(Sequence& parent, const DebugInfos& debug_info_elements = {});

    Break& add_break(
        Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfos& debug_info_elements = {}
    );

    Map& add_map(
        Sequence& parent,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<Map&, Transition&> add_map_after(
        Sequence& parent,
        ControlFlowNode& block,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    std::pair<Map&, Transition&> add_map_before(
        Sequence& parent,
        ControlFlowNode& block,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update,
        const ScheduleType& schedule_type,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    Return& add_return(
        Sequence& parent,
        const sdfg::control_flow::Assignments& assignments = {},
        const DebugInfos& debug_info_elements = {}
    );

    For& convert_while(
        Sequence& parent,
        While& loop,
        const symbolic::Symbol& indvar,
        const symbolic::Condition& condition,
        const symbolic::Expression& init,
        const symbolic::Expression& update
    );

    Map& convert_for(Sequence& parent, For& loop);

    void clear_sequence(Sequence& parent);

    [[deprecated("use ScopeAnalysis instead")]]
    Sequence& parent(const ControlFlowNode& node);

    /***** Section: Dataflow Graph *****/

    data_flow::AccessNode& add_access(
        structured_control_flow::Block& block, const std::string& data, const DebugInfos& debug_info_elements = {}
    );

    data_flow::Tasklet& add_tasklet(
        structured_control_flow::Block& block,
        const data_flow::TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_memlet(
        structured_control_flow::Block& block,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_memlet(
        structured_control_flow::Block& block,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::LibraryNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        structured_control_flow::Block& block,
        data_flow::LibraryNode& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_reference_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    data_flow::Memlet& add_dereference_memlet(
        structured_control_flow::Block& block,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        bool derefs_src,
        const types::IType& base_type,
        const DebugInfos& debug_info_elements = {}
    );

    template<typename T, typename... Args>
    data_flow::LibraryNode&
    add_library_node(structured_control_flow::Block& block, const DebugInfoRegion& debug_info, Args... arguments) {
        static_assert(std::is_base_of<data_flow::LibraryNode, T>::value, "T must be a subclass of data_flow::LibraryNode");

        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node = std::unique_ptr<T>(new T(this->new_element_id(), debug_info, vertex, dataflow, arguments...));
        auto res = dataflow.nodes_.insert({vertex, std::move(node)});

        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    };

    data_flow::LibraryNode& copy_library_node(structured_control_flow::Block& block, const data_flow::LibraryNode& node) {
        auto& dataflow = block.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node_clone = node.clone(this->new_element_id(), vertex, dataflow);
        auto res = dataflow.nodes_.insert({vertex, std::move(node_clone)});
        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    };

    void remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge);

    void remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::CodeNode& node);

    void clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node);

    size_t add_debug_info_element(const DebugInfo& element);

    const DebugTable& debug_info() const;
};

} // namespace builder
} // namespace sdfg

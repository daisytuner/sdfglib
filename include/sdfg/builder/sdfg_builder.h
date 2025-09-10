#pragma once

#include <utility>

#include "sdfg/builder/function_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/debug_info.h"
#include "sdfg/sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace builder {

class SDFGBuilder : public FunctionBuilder {
private:
    std::unique_ptr<SDFG> sdfg_;

    DebugInfoRegion fill_debug_info(const std::vector<DebugInfo>& debug_info_elements);

protected:
    Function& function() const override;

public:
    SDFGBuilder(std::unique_ptr<SDFG>& sdfg);

    SDFGBuilder(const std::string& name, FunctionType type);

    SDFG& subject() const;

    sdfg::control_flow::State* get_non_const_state(const sdfg::control_flow::State* state) const {
        for (auto& entry : sdfg_->states_) {
            if (entry.second.get() == state) {
                return entry.second.get();
            }
        }
        return nullptr;
    }

    std::unique_ptr<SDFG> move();

    /***** Section: Control-Flow Graph *****/

    control_flow::State& add_state(bool is_start_state = false, const std::vector<DebugInfo>& debug_info_elements = {});

    control_flow::State& add_state_before(
        const control_flow::State& state,
        bool is_start_state = false,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    control_flow::State& add_state_after(
        const control_flow::State& state,
        bool connect_states = true,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const symbolic::Condition condition,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const control_flow::Assignments& assignments,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    control_flow::InterstateEdge& add_edge(
        const control_flow::State& src,
        const control_flow::State& dst,
        const control_flow::Assignments& assignments,
        const symbolic::Condition condition,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    void remove_edge(const control_flow::InterstateEdge& edge);

    std::tuple<control_flow::State&, control_flow::State&, control_flow::State&> add_loop(
        const control_flow::State& state,
        sdfg::symbolic::Symbol iterator,
        sdfg::symbolic::Expression init,
        sdfg::symbolic::Condition cond,
        sdfg::symbolic::Expression update,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    /***** Section: Dataflow Graph *****/

    data_flow::AccessNode& add_access(
        control_flow::State& state, const std::string& data, const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Tasklet& add_tasklet(
        control_flow::State& state,
        const data_flow::TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_memlet(
        control_flow::State& state,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_memlet(
        control_flow::State& state,
        data_flow::DataFlowNode& src,
        const std::string& src_conn,
        data_flow::DataFlowNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::Tasklet& dst,
        const std::string& dst_conn,
        const data_flow::Subset& subset,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::Tasklet& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::LibraryNode& dst,
        const std::string& dst_conn,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_computational_memlet(
        control_flow::State& state,
        data_flow::LibraryNode& src,
        const std::string& src_conn,
        data_flow::AccessNode& dst,
        const data_flow::Subset& begin_subset,
        const data_flow::Subset& end_subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_reference_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        const data_flow::Subset& subset,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    data_flow::Memlet& add_dereference_memlet(
        control_flow::State& state,
        data_flow::AccessNode& src,
        data_flow::AccessNode& dst,
        bool derefs_src,
        const types::IType& base_type,
        const std::vector<DebugInfo>& debug_info_elements = {}
    );

    template<typename T, typename... Args>
    data_flow::LibraryNode&
    add_library_node(control_flow::State& state, const DebugInfoRegion& debug_info_region, Args... arguments) {
        static_assert(std::is_base_of<data_flow::LibraryNode, T>::value, "T must be a subclass of data_flow::LibraryNode");

        auto& dataflow = state.dataflow();
        auto vertex = boost::add_vertex(dataflow.graph_);
        auto node = std::unique_ptr<T>(new T(this->new_element_id(), debug_info_region, vertex, dataflow, arguments...)
        );
        auto res = dataflow.nodes_.insert({vertex, std::move(node)});

        return static_cast<data_flow::LibraryNode&>(*(res.first->second));
    };

    size_t add_debug_info_element(const DebugInfo& element);

    const DebugTable& debug_info() const;
};

} // namespace builder
} // namespace sdfg

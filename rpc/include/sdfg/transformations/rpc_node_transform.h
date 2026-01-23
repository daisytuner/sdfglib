#pragma once
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include "sdfg/analysis/users.h"
#include "sdfg/passes/rpc/rpc_responses.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/transformations/utils.h"

namespace sdfg {
namespace transformations {

class RPCNodeTransform : public sdfg::transformations::Transformation {
private:
    structured_control_flow::ControlFlowNode& node_;
    std::string target_;
    std::string category_;

    sdfg::passes::rpc::RpcContext& rpc_context_;

    std::unique_ptr<passes::rpc::RpcOptResponse> applied_opt_;

    bool dump_steps_;

    std::string get_node_id_str() const;

public:
    RPCNodeTransform(
        structured_control_flow::ControlFlowNode& node,
        const std::string& target,
        const std::string& category,
        sdfg::passes::rpc::RpcContext& rpc_context,
        bool print_steps = false
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(
        sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager
    ) override;

    virtual void apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;

    passes::rpc::RpcOptResponse& applied_recipe() const { return *applied_opt_; }

    void to_json(nlohmann::json& j) const override;
};


} // namespace transformations
} // namespace sdfg

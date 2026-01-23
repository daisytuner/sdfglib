#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg::passes::rpc {

struct RpcOptimizationMetadata {
    std::optional<std::string> region_id;
    double speedup = NAN;
    std::optional<double> vector_distance;
};

/**
 * Placeholder for native recipes
 */
struct RpcLocalReplayRecipe {
    nlohmann::json sequence;
};

struct RpcSdfgResult {
    std::unique_ptr<sdfg::StructuredSDFG> sdfg = nullptr;
};

struct RpcOptResponse {
    std::optional<RpcSdfgResult> sdfg_result;
    std::optional<RpcLocalReplayRecipe> local_replay;
    RpcOptimizationMetadata metadata;
    std::optional<std::string> error;
};

struct RpcOptRequest {
    StructuredSDFG& sdfg;
    std::optional<std::string> category;
    std::optional<std::string> target;
    std::optional<analysis::LoopInfo> loop_info;
};

} // namespace sdfg::passes::rpc

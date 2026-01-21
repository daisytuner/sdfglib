#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "sdfg/structured_sdfg.h"

struct RpcOptimizationMetadata {
    std::string region_id;
    double speedup;
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
};

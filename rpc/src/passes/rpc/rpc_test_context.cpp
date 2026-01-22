#include "sdfg/passes/rpc/rpc_test_context.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace sdfg::passes::rpc {

std::unique_ptr<RpcTestContext> RpcTestContext::build_context() {
    const char* cfg_path_c = std::getenv("SDFG_RPC_CONFIG");

    // If a config is explicitly provided, require all expected keys.
    if (cfg_path_c && *cfg_path_c) {
        std::filesystem::path cfg_path(cfg_path_c);
        std::ifstream in(cfg_path);
        if (!in) {
            throw std::runtime_error("SDFG_RPC_CONFIG points to unreadable file: " + cfg_path.string());
        }

        nlohmann::json j;
        in >> j;

        if (!j.contains("SERVER") || !j["SERVER"].is_string()) {
            throw std::runtime_error("SDFG_RPC_CONFIG JSON is missing string key 'SERVER'");
        }
        if (!j.contains("ENDPOINT") || !j["ENDPOINT"].is_string()) {
            throw std::runtime_error("SDFG_RPC_CONFIG JSON is missing string key 'ENDPOINT'");
        }
        if (!j.contains("AUTH_HEADER") || !j["AUTH_HEADER"].is_string()) {
            throw std::runtime_error("SDFG_RPC_CONFIG JSON is missing string key 'AUTH_HEADER'");
        }

        std::string server = j["SERVER"].get<std::string>();
        std::string endpoint = j["ENDPOINT"].get<std::string>();
        std::string auth_header = j["AUTH_HEADER"].get<std::string>();

        return std::make_unique<RpcTestContext>(std::move(server), std::move(endpoint), std::move(auth_header));
    }

    // Default to local test server if no config is provided.
    return std::make_unique<RpcTestContext>("http://localhost:3000/docc", "/get-recipe", "");
}

const RpcTestContext& RpcTestContext::default_context() {
    static std::unique_ptr<RpcTestContext> ctx = RpcTestContext::build_context();
    return *ctx;
}

} // namespace sdfg::passes::rpc

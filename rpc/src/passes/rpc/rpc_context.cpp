#include "sdfg/passes/rpc/rpc_context.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

namespace sdfg::passes::rpc {

std::unique_ptr<RpcContext> build_rpc_context_local() {
    return std::make_unique<SimpleRpcContext>("http://localhost:3000/docc", "/transfertuning");
}

std::unique_ptr<RpcContext> build_rpc_context_from_file(std::optional<std::filesystem::path> config_file) {
    std::filesystem::path cfg_path;

    if (!config_file) {
        auto envVar = std::getenv("SDFG_RPC_CONFIG");
        if (envVar && *envVar) {
            cfg_path = std::filesystem::path(envVar);
        } else {
            std::cerr << "[WARNING] Using local RpcContext" << std::endl;
            return build_rpc_context_local();
        }
    } else {
        cfg_path = *config_file;
    }


    std::ifstream in(cfg_path);
    if (!in) {
        throw std::runtime_error("Config file not readable: " + cfg_path.string());
    }

    nlohmann::json j;
    in >> j;


    std::string server = "http://localhost:3000/docc";
    std::string endpoint = "/transfertuning";
    std::unordered_map<std::string, std::string> headers;

    auto serverJ = j.find("SERVER");
    if (serverJ != j.end() && serverJ->is_string()) {
        server = serverJ->get<std::string>();
    }
    auto endpointJ = j.find("ENDPOINT");
    if (endpointJ != j.end() && endpointJ->is_string()) {
        endpoint = endpointJ->get<std::string>();
    }

    auto headersJ = j.find("HEADERS");
    if (headersJ != j.end() && headersJ->is_object()) {
        for (auto& [key, value] : headersJ->items()) {
            if (value.is_string()) {
                headers[key] = value.get<std::string>();
            }
        }
    }

    auto headerOverrideVar = std::getenv("RPC_HEADER");
    if (headerOverrideVar && *headerOverrideVar) {
        std::string headerOverride = headerOverrideVar;
        auto idx = headerOverride.find_first_of(':');
        if (idx != std::string::npos) {
            std::string key = headerOverride.substr(0, idx);
            std::string value = headerOverride.substr(idx + 1);
            headers[key] = value;
        } else {
            headers["RPC-Hint"] = headerOverride;
        }
    }

    std::cerr << "[INFO] Using RPC target " << server << endpoint << ", headers: ";
    for (const auto& [key, value] : headers) {
        std::cerr << key << ", ";
    }
    std::cerr << ")" << std::endl;

    return std::make_unique<SimpleRpcContext>(server, endpoint, headers);
}

// // Optional testing headers: allow pointing the local server at JSON fixture files.
//         if (const char* sdfg_path = std::getenv("SDFG_TEST_SDFG_PATH")) {
//             if (*sdfg_path) {
//                 headers["sdfg-path"] = sdfg_path;
//             }
//         }
//         if (const char* seq_path = std::getenv("SDFG_TEST_SEQUENCE_PATH")) {
//             if (*seq_path) {
//                 headers["sequence-path"] = seq_path;
//             }
//         }

} // namespace sdfg::passes::rpc

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "sdfg/passes/rpc/rpc_context.h"

namespace sdfg::passes::rpc {

class RpcTestContext : public RpcContext {
private:
    std::string server_;
    std::string endpoint_;
    std::string auth_str_;

public:
    RpcTestContext(std::string server, std::string endpoint, std::string auth_str)
        : server_(server), endpoint_(endpoint), auth_str_(auth_str) {}

    std::string get_remote_address() const override { return server_ + "/" + endpoint_; }

    std::unordered_map<std::string, std::string> get_auth_headers() const override {
        std::unordered_map<std::string, std::string> headers;
        if (!auth_str_.empty()) {
            headers["Authorization"] = auth_str_;
        }

        // Optional testing headers: allow pointing the local server at JSON fixture files.
        if (const char* sdfg_path = std::getenv("SDFG_TEST_SDFG_PATH")) {
            if (*sdfg_path) {
                headers["sdfg-path"] = sdfg_path;
            }
        }
        if (const char* seq_path = std::getenv("SDFG_TEST_SEQUENCE_PATH")) {
            if (*seq_path) {
                headers["sequence-path"] = seq_path;
            }
        }
        return headers;
    }

    static std::unique_ptr<RpcTestContext> build_context();

    // Default singleton context used when no explicit RpcContext is provided.
    static const RpcTestContext& default_context();
};

} // namespace sdfg::passes::rpc

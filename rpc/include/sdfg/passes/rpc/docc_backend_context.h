#pragma once

#include <memory>
#include <optional>
#include "sdfg/passes/rpc/rpc_context.h"

namespace sdfg {
namespace passes {
namespace rpc {

class DoccBackendContext : public RpcContext {
private:
    std::string server_;
    std::string endpoint_;
    std::string auth_str_;
    bool job_only_;

    static std::optional<std::pair<std::string, bool>> find_docc_auth();

public:
    DoccBackendContext(std::string server, std::string endpoint, std::string auth_str, bool job_only)
        : server_(server), endpoint_(endpoint), auth_str_(auth_str), job_only_(job_only) {}

    std::string get_remote_address() const override { return server_ + "/" + endpoint_; }

    std::unordered_map<std::string, std::string> get_auth_headers() const override {
        std::unordered_map<std::string, std::string> headers;

        if (!auth_str_.empty()) {
            headers["Authorization"] = auth_str_;
        }
        return headers;
    }

    static std::unique_ptr<DoccBackendContext> build_context();
};

} // namespace rpc
} // namespace passes
} // namespace sdfg

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace sdfg::passes::rpc {

class RpcContext {
public:
    virtual std::string get_remote_address() const = 0;
    virtual std::unordered_map<std::string, std::string> get_auth_headers() const = 0;

    virtual ~RpcContext() = default;
};

class SimpleRpcContext : public RpcContext {
private:
    std::string server_;
    std::string endpoint_;
    std::unordered_map<std::string, std::string> headers_;

public:
    SimpleRpcContext(std::string server, std::string endpoint, std::unordered_map<std::string, std::string> headers = {})
        : server_(server), endpoint_(endpoint), headers_(headers) {}

    std::string get_remote_address() const override { return server_ + "/" + endpoint_; }

    std::unordered_map<std::string, std::string> get_auth_headers() const override { return headers_; }
};

std::unique_ptr<RpcContext> build_rpc_context_local();

std::unique_ptr<RpcContext> build_rpc_context_from_file(std::optional<std::filesystem::path> config_file = std::nullopt);

} // namespace sdfg::passes::rpc

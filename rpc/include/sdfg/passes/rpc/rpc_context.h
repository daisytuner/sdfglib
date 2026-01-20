#pragma once

#include <string>
#include <unordered_map>

namespace sdfg::passes::rpc {

class RpcContext {
public:
    virtual std::string get_remote_address() const = 0;
    virtual std::unordered_map<std::string, std::string> get_auth_headers() const = 0;

    virtual ~RpcContext() = default;
};

} // namespace sdfg::rpc

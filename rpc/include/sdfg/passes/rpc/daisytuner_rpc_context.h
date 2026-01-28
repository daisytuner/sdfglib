#pragma once

#include "rpc_context.h"

namespace sdfg::passes::rpc {

class DaisytunerTransfertuningRpcContext : public SimpleRpcContext {
public:
    DaisytunerTransfertuningRpcContext(std::string license_token, std::string token_prefix = "Token");


    static std::unique_ptr<DaisytunerTransfertuningRpcContext> from_docc_config();
};

} // namespace sdfg::passes::rpc

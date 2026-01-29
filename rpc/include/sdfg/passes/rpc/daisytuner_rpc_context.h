#pragma once

#include "rpc_context.h"

namespace sdfg::passes::rpc {

class DaisytunerTransfertuningRpcContext : public SimpleRpcContext {
public:
    inline static constexpr auto DEFAULT_ENDPOINT = "transfertune";
    inline static constexpr auto DEFAULT_AUTH_HEADER = "Authorization";
    inline static constexpr auto DEFAULT_SERVER = "https://docc-backend-1080482399950.europe-west1.run.app/docc";

    DaisytunerTransfertuningRpcContext(std::string license_token, bool job_specific_token = false);

    static std::unique_ptr<DaisytunerTransfertuningRpcContext> from_docc_config();

    static std::string build_auth_header_content(std::pair<std::string, bool> docc_auth);

    static std::optional<std::pair<std::string, bool>> find_docc_auth();
};

} // namespace sdfg::passes::rpc

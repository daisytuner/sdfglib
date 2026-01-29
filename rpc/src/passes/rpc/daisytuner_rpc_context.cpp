#include "sdfg/passes/rpc/daisytuner_rpc_context.h"

#include <fstream>
#include <iostream>

namespace sdfg::passes::rpc {

static std::optional<std::pair<std::string, bool>> find_docc_auth() {
    // Check $DOCC_ACCESS_TOKEN_ENV
    const char* env_token = std::getenv("DOCC_ACCESS_TOKEN");
    if (env_token && *env_token) {
        return std::make_pair(env_token, false);
    }

    // Check $HOME/.config/docc/token
    const char* home = std::getenv("HOME");
    if (home && *home) {
        std::filesystem::path config_dir = std::filesystem::path(home) / ".config" / "docc";
        std::string path = (config_dir / "token").string();
        std::ifstream infile(path);
        if (infile) {
            std::ostringstream ss;
            ss << infile.rdbuf();
            std::string token = ss.str();
            // Remove trailing newlines/spaces
            token.erase(token.find_last_not_of(" \n\r\t") + 1);
            return std::make_pair(token, false);
        }
    }

    // Check /var/lib/daisytuner/session
    std::filesystem::path session_path = "/var/lib/daisytuner/session";
    std::ifstream session_file(session_path);
    if (session_file) {
        std::ostringstream ss;
        ss << session_file.rdbuf();
        std::string token = ss.str();
        // Remove trailing newlines/spaces
        token.erase(token.find_last_not_of(" \n\r\t") + 1);
        return std::make_pair(token, false);
    }

    return std::nullopt;
}

DaisytunerTransfertuningRpcContext::DaisytunerTransfertuningRpcContext(std::string license_token, std::string token_prefix)
    : SimpleRpcContext(
          "https://docc-backend-1080482399950.europe-west1.run.app/docc",
          "transfertune",
          {{"Authorization", token_prefix + " " + license_token}}
      ) {}


std::unique_ptr<DaisytunerTransfertuningRpcContext> DaisytunerTransfertuningRpcContext::from_docc_config() {
    auto auth = find_docc_auth();
    if (!auth.has_value()) {
        throw std::runtime_error("DOCC access token not found in DOCC_ACCESS_TOKEN or $HOME/.config/docc/token");
    }
    std::cerr << "[INFO] Using Daisytuner DOCC Backend" << std::endl;

    return std::make_unique<DaisytunerTransfertuningRpcContext>(auth->first);
}

} // namespace sdfg::passes::rpc

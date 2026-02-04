#include "sdfg/passes/rpc/docc_backend_context.h"
#include <fstream>

namespace sdfg {
namespace passes {
namespace rpc {

std::optional<std::pair<std::string, bool>> DoccBackendContext::find_docc_auth() {
    // Job specific token $DOCC_JOB_TOKEN wins against all. Only on runners
    const char* job_token = std::getenv("DOCC_JOB_TOKEN");
    if (job_token && *job_token) {
        std::string auth_str = "Job " + std::string(job_token);
        return std::make_pair(auth_str, true);
    }

    // Check $DOCC_ACCESS_TOKEN_ENV
    const char* env_token = std::getenv("DOCC_ACCESS_TOKEN");
    if (env_token && *env_token) {
        std::string auth_str = "Token " + std::string(env_token);
        return std::make_pair(auth_str, false);
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
            std::string auth_str = "Token " + token;
            return std::make_pair(auth_str, false);
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
        std::string auth_str = "Token " + token;
        return std::make_pair(auth_str, false);
    }

    return std::nullopt;
}

std::shared_ptr<DoccBackendContext> DoccBackendContext::build_context() {
    auto auth = find_docc_auth();
    if (auth) {
        // TODO lookup server overrides
        return std::make_shared<DoccBackendContext>(
            "https://europe-west1-daisy-367210.cloudfunctions.net/docc",
            "transfertuning/get_closest_neighbors",
            auth->first,
            auth->second
        );
    } else {
        return nullptr;
    }
}

} // namespace rpc
} // namespace passes
} // namespace sdfg

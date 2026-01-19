#include "sdfg/targets/offloading/utils.h"

namespace cuda_offloading_codegen {


bool write_library_snippets_to_files(
    std::filesystem::path build_path,
    std::unordered_set<std::string> lib_files,
    const std::unordered_map<std::string, sdfg::codegen::CodeSnippet>& snippets,
    std::unordered_map<std::string, std::vector<std::filesystem::path>>& files_for_post_processing,
    const std::string& file_ending
) {
    for (auto& [name, snippet] : snippets) {
        if (snippet.is_as_file()) {
            auto p = build_path / (name + "." + snippet.extension());
            std::ofstream outfile_lib;
            if (lib_files.insert(p.string()).second) {
                outfile_lib.open(p, std::ios_base::out);
            } else {
                outfile_lib.open(p, std::ios_base::app);
            }
            if (!outfile_lib.is_open()) {
                throw std::runtime_error("Failed to open library file: " + p.string());
            }
            outfile_lib << snippet.stream().str() << std::endl;
            outfile_lib.close();

            if (snippet.extension() == file_ending) {
                auto it = files_for_post_processing.find(file_ending);
                std::vector<std::filesystem::path>* cu_files;
                if (it != files_for_post_processing.end()) {
                    cu_files = &(it->second);
                } else {
                    cu_files = &files_for_post_processing.emplace(file_ending, std::vector<std::filesystem::path>())
                                    .first->second;
                }
                cu_files->emplace_back(p);
            }
        }
    }

    return true;
}

} // namespace cuda_offloading_codegen


#include <sdfg/codegen/code_generator.h>
#include <sdfg/structured_control_flow/map.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuda_offloading_codegen {

bool write_library_snippets_to_files(
    std::filesystem::path,
    std::unordered_set<std::string>,
    const std::unordered_map<std::string, sdfg::codegen::CodeSnippet>&,
    std::unordered_map<std::string, std::vector<std::filesystem::path>>&,
    const std::string& file_ending
);

} // namespace cuda_offloading_codegen

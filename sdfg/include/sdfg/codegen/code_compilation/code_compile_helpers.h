#pragma once

#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/structured_control_flow/map.h"
namespace sdfg {
namespace codegen {

bool base_include_paths();

bool write_library_snippets_to_files(
    std::filesystem::path build_path,
    std::unordered_set<std::string> lib_files,
    const std::unordered_map<std::string, sdfg::codegen::CodeSnippet>& snippets,
    std::unordered_map<std::string, std::vector<std::filesystem::path>>& files_for_post_processing
);

bool compile_additional_files(
    const std::filesystem::path& build_path,
    const std::vector<std::string>& compile_args,
    std::set<std::string>& link_2nd_args,
    std::unordered_map<std::string, std::vector<std::filesystem::path>> files_for_post_processing
);

bool compile_to_object_file(
    const std::filesystem::path& source_file,
    const std::filesystem::path& object_file,
    const std::vector<std::string>& compile_args
);

void add_schedule_type_specific_linker_args(
    const structured_control_flow::ScheduleType& schedule_type, std::set<std::string>& linker_args
);

void add_schedule_type_specific_compile_args(
    const structured_control_flow::ScheduleType& schedule_type, std::vector<std::string>& compile_args
);

} // namespace codegen
} // namespace sdfg

#include "sdfg/codegen/code_snippet_factory.h"

namespace sdfg::codegen {

CodeSnippetFactory::CodeSnippetFactory(const std::pair<std::filesystem::path, std::filesystem::path>* config)
    : output_path_(config ? config->first : "."), header_path_(config ? config->second : "") {}


CodeSnippet& CodeSnippetFactory::require(const std::string& name, const std::string& extension, bool as_file) {
    auto [snippet, newly_created] = snippets_.try_emplace(name, name, extension, as_file);

    if (!newly_created && extension != snippet->second.extension()) {
        throw std::runtime_error(
            "Code snippet " + name + " already exists with '." + snippet->second.extension() +
            "', but was required with '." + extension + "'"
        );
    }

    return snippet->second;
}

std::unordered_map<std::string, CodeSnippet>::iterator CodeSnippetFactory::find(const std::string& name) {
    return snippets_.find(name);
}
const std::unordered_map<std::string, CodeSnippet>& CodeSnippetFactory::snippets() const { return snippets_; }

} // namespace sdfg::codegen

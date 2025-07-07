#include "sdfg/codegen/utils.h"

namespace sdfg {
namespace codegen {

// Constructor
PrettyPrinter::PrettyPrinter(int indent, bool frozen)
    : indentSize(indent), frozen_(frozen) {

      };

// Set the indentation level
void PrettyPrinter::setIndent(int indent) { indentSize = indent; };

int PrettyPrinter::indent() const { return indentSize; };

// Get the underlying string
std::string PrettyPrinter::str() const { return stream.str(); };

// Clear the stringstream content
void PrettyPrinter::clear() {
    stream.str("");
    stream.clear();
};

// Overload for manipulators (like std::endl)
PrettyPrinter& PrettyPrinter::operator<<(std::ostream& (*manip)(std::ostream&) ) {
    if (frozen_) {
        throw std::runtime_error("PrettyPrinter is frozen");
    }
    stream << manip;
    // Reset indent application on new lines
    if (manip == static_cast<std::ostream& (*) (std::ostream&)>(std::endl)) {
        isNewLine = true;
    }
    return *this;
};

// Apply indentation only at the beginning of a new line
void PrettyPrinter::applyIndent() {
    if (isNewLine && indentSize > 0) {
        stream << std::setw(indentSize) << "";
        isNewLine = false;
    }
};

Reference::Reference(const types::IType& reference_) : reference_(reference_.clone()) {};

Reference::Reference(
    types::StorageType storage_type, size_t alignment, const std::string& initializer, const types::IType& reference_
)
    : IType(storage_type, alignment, initializer), reference_(reference_.clone()) {};

std::unique_ptr<types::IType> Reference::clone() const {
    return std::make_unique<Reference>(this->storage_type(), this->alignment(), this->initializer(), *this->reference_);
};

types::TypeID Reference::type_id() const { return types::TypeID::Reference; };

types::PrimitiveType Reference::primitive_type() const { return this->reference_->primitive_type(); };

bool Reference::is_symbol() const { return false; };

const types::IType& Reference::reference_type() const { return *this->reference_; };

bool Reference::operator==(const types::IType& other) const {
    if (auto reference = dynamic_cast<const Reference*>(&other)) {
        return *(this->reference_) == *reference->reference_ && this->alignment_ == reference->alignment_;
    } else {
        return false;
    }
};

std::string Reference::print() const { return "Reference(" + this->reference_->print() + ")"; };

CodeSnippetFactory::CodeSnippetFactory(const std::pair<std::filesystem::path, std::filesystem::path>* config)
    : output_path_(config ? config->first : "."), header_path_(config ? config->second : "") {}


CodeSnippet& CodeSnippetFactory::require(const std::string& name, const std::string& extension, bool as_file) {
    auto [snippet, newly_created] = snippets_.try_emplace(name, extension, as_file);

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

} // namespace codegen
} // namespace sdfg

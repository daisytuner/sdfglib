#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>

#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

class PrettyPrinter {
public:
    // Constructor
    PrettyPrinter(int indent = 0, bool frozen = false);

    // Set the indentation level
    void setIndent(int indent);

    int indent() const;

    // Get the underlying string
    std::string str() const;

    // Clear the stringstream content
    void clear();

    // Overload the insertion operator
    template<typename T>
    PrettyPrinter& operator<<(const T& value) {
        if (frozen_) {
            throw std::runtime_error("PrettyPrinter is frozen");
        }
        applyIndent();
        stream << value;
        return *this;
    }

    // Overload for manipulators (like std::endl)
    PrettyPrinter& operator<<(std::ostream& (*manip)(std::ostream&) );

private:
    std::stringstream stream;
    int indentSize;
    bool isNewLine = true;
    bool frozen_;

    // Apply indentation only at the beginning of a new line
    void applyIndent();
};

class Reference : public types::IType {
private:
    std::unique_ptr<types::IType> reference_;

public:
    Reference(const types::IType& reference_);

    Reference(
        types::StorageType storage_type, size_t alignment, const std::string& initializer, const types::IType& reference_
    );

    std::unique_ptr<types::IType> clone() const override;

    virtual types::TypeID type_id() const override;

    types::PrimitiveType primitive_type() const override;

    bool is_symbol() const override;

    const types::IType& reference_type() const;

    bool operator==(const types::IType& other) const override;

    std::string print() const override;
};

class CodeSnippet {
protected:
    PrettyPrinter stream_;
    std::string extension_;
    bool as_file_;

public:
    CodeSnippet(const std::string& extension, bool as_file) : extension_(extension), as_file_(as_file) {};

    PrettyPrinter& stream() { return stream_; }

    const PrettyPrinter& stream() const { return stream_; }

    const std::string& extension() const { return extension_; }

    bool is_as_file() const { return as_file_; }
};

class CodeSnippetFactory {
protected:
    std::unordered_map<std::string, CodeSnippet> snippets_;
    const std::filesystem::path output_path_;
    const std::filesystem::path header_path_;

public:
    CodeSnippetFactory(const std::pair<std::filesystem::path, std::filesystem::path>* config = nullptr);

    virtual ~CodeSnippetFactory() = default;

    CodeSnippet& require(const std::string& name, const std::string& extension, bool as_file = true);

    std::unordered_map<std::string, CodeSnippet>::iterator find(const std::string& name);

    const std::unordered_map<std::string, CodeSnippet>& snippets() const;

    const std::filesystem::path& output_path() const { return output_path_; }
    const std::filesystem::path& header_path() const { return header_path_; }
};

inline std::string CODE_SNIPPET_INIT_ONCE = "init_once";
inline std::string CODE_SNIPPET_DEINIT_ONCE = "deinit_once";

} // namespace codegen
} // namespace sdfg

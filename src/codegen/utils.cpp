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
PrettyPrinter& PrettyPrinter::operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (frozen_) {
        throw std::runtime_error("PrettyPrinter is frozen");
    }
    stream << manip;
    // Reset indent application on new lines
    if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
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

Reference::Reference(const types::IType& reference_)
    : reference_(reference_.clone()) {

      };

std::unique_ptr<types::IType> Reference::clone() const {
    return std::make_unique<Reference>(*this->reference_);
};

types::PrimitiveType Reference::primitive_type() const {
    return this->reference_->primitive_type();
};

bool Reference::is_symbol() const { return false; };

const types::IType& Reference::reference_type() const { return *this->reference_; };

bool Reference::operator==(const types::IType& other) const {
    if (auto reference = dynamic_cast<const Reference*>(&other)) {
        return *(this->reference_) == *reference->reference_;
    } else {
        return false;
    }
};

uint Reference::address_space() const { return this->reference_->address_space(); };

sdfg::types::DeviceLocation Reference::device_location() const {
    return this->reference_->device_location();
};

std::string Reference::initializer() const { return this->reference_->initializer(); };

}  // namespace codegen
}  // namespace sdfg

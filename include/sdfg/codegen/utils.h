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
    template <typename T>
    PrettyPrinter& operator<<(const T& value) {
        if (frozen_) {
            throw std::runtime_error("PrettyPrinter is frozen");
        }
        applyIndent();
        stream << value;
        return *this;
    }

    // Overload for manipulators (like std::endl)
    PrettyPrinter& operator<<(std::ostream& (*manip)(std::ostream&));

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

    std::unique_ptr<types::IType> clone() const override;

    types::PrimitiveType primitive_type() const override;

    bool is_symbol() const override;

    const types::IType& reference_type() const;

    bool operator==(const types::IType& other) const override;

    uint address_space() const override;

    sdfg::types::DeviceLocation device_location() const override;

    std::string initializer() const override;
};

}  // namespace codegen
}  // namespace sdfg
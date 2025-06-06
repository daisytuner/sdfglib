#pragma once

#include <exception>
#include <string>
#include <string_view>

namespace sdfg {

class InvalidSDFGException : public std::exception {
   private:
    std::string message_;

   public:
    InvalidSDFGException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

class UnstructuredControlFlowException : public std::exception {
   public:
    const char* what() const noexcept override { return "Unstructured control flow detected"; }
};

class StringEnum {
   public:
    constexpr explicit StringEnum(std::string_view value) : value_(value) {}

    constexpr std::string_view value() const { return value_; }

    constexpr bool operator==(const StringEnum& other) const { return value_ == other.value_; }

    constexpr bool operator!=(const StringEnum& other) const { return !(*this == other); }

   private:
    std::string_view value_;
};

}  // namespace sdfg

#pragma once

#include <exception>
#include <string>
#include <utility>

namespace sdfg {

class InvalidSDFGException : public std::exception {
private:
    std::string message;

public:
    explicit InvalidSDFGException(std::string message) : message(std::move(message)) {}

    auto what() const noexcept -> const char* override { return message.c_str(); }
};

class UnstructuredControlFlowException : public std::exception {
public:
    auto what() const noexcept -> const char* override { return "Unstructured control flow detected"; }
};

class StringEnum {
public:
    explicit StringEnum(std::string value) : value_(std::move(value)) {}
    StringEnum(const StringEnum& other) = default;
    StringEnum(StringEnum&& other) noexcept : value_(std::move(other.value_)) {}

    auto operator=(const StringEnum& other) -> StringEnum& = default;

    auto operator=(StringEnum&& other) noexcept -> StringEnum& {
        value_ = std::move(other.value_);
        return *this;
    }

    auto value() const -> std::string { return value_; }

    auto operator==(const StringEnum& other) const -> bool { return value_ == other.value_; }

    auto operator!=(const StringEnum& other) const -> bool { return !(*this == other); }

private:
    std::string value_;
};

} // namespace sdfg

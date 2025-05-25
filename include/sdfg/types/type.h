#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <type_traits>

#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

namespace types {

enum PrimitiveType {
    Void,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Half,
    BFloat,
    Float,
    Double,
    X86_FP80,
    FP128,
    PPC_FP128
};

enum DeviceLocation { x86, nvptx };

constexpr const char* primitive_type_to_string(PrimitiveType e) {
    switch (e) {
        case PrimitiveType::Void:
            return "Void";
        case PrimitiveType::Bool:
            return "Bool";
        case PrimitiveType::Int8:
            return "Int8";
        case PrimitiveType::Int16:
            return "Int16";
        case PrimitiveType::Int32:
            return "Int32";
        case PrimitiveType::Int64:
            return "Int64";
        case PrimitiveType::UInt8:
            return "UInt8";
        case PrimitiveType::UInt16:
            return "UInt16";
        case PrimitiveType::UInt32:
            return "UInt32";
        case PrimitiveType::UInt64:
            return "UInt64";
        case PrimitiveType::Half:
            return "Half";
        case PrimitiveType::BFloat:
            return "BFloat";
        case PrimitiveType::Float:
            return "Float";
        case PrimitiveType::Double:
            return "Double";
        case PrimitiveType::X86_FP80:
            return "X86_FP80";
        case PrimitiveType::FP128:
            return "FP128";
        case PrimitiveType::PPC_FP128:
            return "PPC_FP128";
    }
    throw std::invalid_argument("Invalid primitive type");
};

constexpr PrimitiveType primitive_type_from_string(std::string_view e) {
    if (e == "Void") {
        return PrimitiveType::Void;
    } else if (e == "Bool") {
        return PrimitiveType::Bool;
    } else if (e == "Int8") {
        return PrimitiveType::Int8;
    } else if (e == "Int16") {
        return PrimitiveType::Int16;
    } else if (e == "Int32") {
        return PrimitiveType::Int32;
    } else if (e == "Int64") {
        return PrimitiveType::Int64;
    } else if (e == "UInt8") {
        return PrimitiveType::UInt8;
    } else if (e == "UInt16") {
        return PrimitiveType::UInt16;
    } else if (e == "UInt32") {
        return PrimitiveType::UInt32;
    } else if (e == "UInt64") {
        return PrimitiveType::UInt64;
    } else if (e == "Half") {
        return PrimitiveType::Half;
    } else if (e == "BFloat") {
        return PrimitiveType::BFloat;
    } else if (e == "Float") {
        return PrimitiveType::Float;
    } else if (e == "Double") {
        return PrimitiveType::Double;
    } else if (e == "X86_FP80") {
        return PrimitiveType::X86_FP80;
    } else if (e == "FP128") {
        return PrimitiveType::FP128;
    } else if (e == "PPC_FP128") {
        return PrimitiveType::PPC_FP128;
    }
    throw std::invalid_argument("Invalid primitive type");
};

constexpr size_t bit_width(PrimitiveType e) {
    switch (e) {
        case PrimitiveType::Void:
            return 0;
        case PrimitiveType::Bool:
            return 1;
        case PrimitiveType::Int8:
            return 8;
        case PrimitiveType::Int16:
            return 16;
        case PrimitiveType::Int32:
            return 32;
        case PrimitiveType::Int64:
            return 64;
        case PrimitiveType::UInt8:
            return 8;
        case PrimitiveType::UInt16:
            return 16;
        case PrimitiveType::UInt32:
            return 32;
        case PrimitiveType::UInt64:
            return 64;
        case PrimitiveType::Half:
            return 16;
        case PrimitiveType::BFloat:
            return 16;
        case PrimitiveType::Float:
            return 32;
        case PrimitiveType::Double:
            return 64;
        case PrimitiveType::X86_FP80:
            return 80;
        case PrimitiveType::FP128:
            return 128;
        case PrimitiveType::PPC_FP128:
            return 128;
    }
    throw std::invalid_argument("Invalid primitive type");
};

constexpr bool is_floating_point(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Half:
        case PrimitiveType::BFloat:
        case PrimitiveType::Float:
        case PrimitiveType::Double:
        case PrimitiveType::X86_FP80:
        case PrimitiveType::FP128:
        case PrimitiveType::PPC_FP128:
            return true;
        default:
            return false;
    }
};

constexpr bool is_integer(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Bool:
        case PrimitiveType::Int8:
        case PrimitiveType::Int16:
        case PrimitiveType::Int32:
        case PrimitiveType::Int64:
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
            return true;
        default:
            return false;
    }
};

constexpr bool is_signed(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Int8:
        case PrimitiveType::Int16:
        case PrimitiveType::Int32:
        case PrimitiveType::Int64:
            return true;
        default:
            return false;
    }
};

constexpr bool is_unsigned(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Bool:
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
            return true;
        default:
            return false;
    }
};

constexpr PrimitiveType as_signed(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::UInt8:
            return PrimitiveType::Int8;
        case PrimitiveType::UInt16:
            return PrimitiveType::Int16;
        case PrimitiveType::UInt32:
            return PrimitiveType::Int32;
        case PrimitiveType::UInt64:
            return PrimitiveType::Int64;
        default:
            return e;
    }
};

constexpr PrimitiveType as_unsigned(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Int8:
            return PrimitiveType::UInt8;
        case PrimitiveType::Int16:
            return PrimitiveType::UInt16;
        case PrimitiveType::Int32:
            return PrimitiveType::UInt32;
        case PrimitiveType::Int64:
            return PrimitiveType::UInt64;
        default:
            return e;
    }
};

class IType {
   public:
    virtual ~IType() = default;

    virtual PrimitiveType primitive_type() const = 0;

    virtual DeviceLocation device_location() const = 0;

    virtual uint address_space() const = 0;

    virtual bool is_symbol() const = 0;

    virtual bool operator==(const IType& other) const = 0;

    virtual std::unique_ptr<IType> clone() const = 0;

    virtual std::string initializer() const = 0;
};

}  // namespace types
}  // namespace sdfg

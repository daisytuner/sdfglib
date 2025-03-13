#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
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
    Float,
    Double
};

enum DeviceLocation { x86, nvptx };

constexpr const char* primitive_type_to_string(PrimitiveType e) noexcept {
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
        case PrimitiveType::Float:
            return "Float";
        case PrimitiveType::Double:
            return "Double";
    }
    assert(false);
};

constexpr PrimitiveType primitive_type_from_string(const char* e) noexcept {
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
    } else if (e == "Float") {
        return PrimitiveType::Float;
    } else if (e == "Double") {
        return PrimitiveType::Double;
    }
    assert(false);
};

constexpr size_t bit_width(PrimitiveType e) noexcept {
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
        case PrimitiveType::Float:
            return 32;
        case PrimitiveType::Double:
            return 64;
    }
    assert(false);
};

constexpr bool is_floating_point(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::Float:
        case PrimitiveType::Double:
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

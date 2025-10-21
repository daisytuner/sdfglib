#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <type_traits>

#include "sdfg/exceptions.h"
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
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Half,
    BFloat,
    Float,
    Double,
    X86_FP80,
    FP128,
    PPC_FP128
};

class StorageType {
public:
    enum AllocationLifetime {
        Lifetime_Default,
        Lifetime_SDFG,
    };

private:
    std::string value_;
    symbolic::Expression allocation_size_;
    AllocationLifetime allocation_lifetime_;

public:
    StorageType(const std::string& value)
        : value_(value), allocation_size_(SymEngine::null), allocation_lifetime_(Lifetime_Default) {}

    StorageType(
        const std::string& value, const symbolic::Expression& allocation_size, AllocationLifetime allocation_lifetime
    )
        : value_(value), allocation_size_(allocation_size), allocation_lifetime_(allocation_lifetime) {}

    std::string value() const { return value_; }

    symbolic::Expression allocation_size() const { return allocation_size_; }

    AllocationLifetime allocation_lifetime() const { return allocation_lifetime_; }

    bool operator==(const StorageType& other) const {
        if (value_ != other.value_) {
            return false;
        }
        if (allocation_lifetime_ != other.allocation_lifetime_) {
            return false;
        }
        if (allocation_size_.is_null() && other.allocation_size_.is_null()) {
            return true;
        }
        if (!allocation_size_.is_null() && !other.allocation_size_.is_null()) {
            return symbolic::eq(allocation_size_, other.allocation_size_);
        }
        return false;
    }

    bool is_cpu_stack() const { return value_ == "CPU_Stack"; }

    bool is_cpu_heap() const { return value_ == "CPU_Heap"; }

    bool is_nv_generic() const { return value_ == "NV_Generic"; }

    bool is_nv_global() const { return value_ == "NV_Global"; }

    bool is_nv_shared() const { return value_ == "NV_Shared"; }

    bool is_nv_constant() const { return value_ == "NV_Constant"; }

    static StorageType CPU_Stack() { return StorageType("CPU_Stack"); }

    static StorageType CPU_Heap(symbolic::Expression allocation_size, StorageType::AllocationLifetime allocation_lifetime) {
        return StorageType("CPU_Heap", allocation_size, allocation_lifetime);
    }

    static StorageType NV_Generic() { return StorageType("NV_Generic"); }

    static StorageType NV_Global() { return StorageType("NV_Global"); }

    static StorageType NV_Shared() { return StorageType("NV_Shared"); }

    static StorageType NV_Constant() { return StorageType("NV_Constant"); }
};

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
        case PrimitiveType::Int128:
            return "Int128";
        case PrimitiveType::UInt8:
            return "UInt8";
        case PrimitiveType::UInt16:
            return "UInt16";
        case PrimitiveType::UInt32:
            return "UInt32";
        case PrimitiveType::UInt64:
            return "UInt64";
        case PrimitiveType::UInt128:
            return "UInt128";
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
    } else if (e == "Int128") {
        return PrimitiveType::Int128;
    } else if (e == "UInt8") {
        return PrimitiveType::UInt8;
    } else if (e == "UInt16") {
        return PrimitiveType::UInt16;
    } else if (e == "UInt32") {
        return PrimitiveType::UInt32;
    } else if (e == "UInt64") {
        return PrimitiveType::UInt64;
    } else if (e == "UInt128") {
        return PrimitiveType::UInt128;
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
        case PrimitiveType::Int128:
            return 128;
        case PrimitiveType::UInt8:
            return 8;
        case PrimitiveType::UInt16:
            return 16;
        case PrimitiveType::UInt32:
            return 32;
        case PrimitiveType::UInt64:
            return 64;
        case PrimitiveType::UInt128:
            return 128;
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
        case PrimitiveType::Int128:
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
        case PrimitiveType::UInt128:
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
        case PrimitiveType::Int128:
            return true;
        default:
            return false;
    }
};

constexpr bool is_unsigned(PrimitiveType e) noexcept {
    switch (e) {
        case PrimitiveType::UInt8:
        case PrimitiveType::UInt16:
        case PrimitiveType::UInt32:
        case PrimitiveType::UInt64:
        case PrimitiveType::UInt128:
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
        case PrimitiveType::UInt128:
            return PrimitiveType::Int128;
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
        case PrimitiveType::Int128:
            return PrimitiveType::UInt128;
        default:
            return e;
    }
};

enum class TypeID {
    Scalar,
    Array,
    Structure,
    Pointer,
    Reference,
    Function,
};

class IType {
protected:
    StorageType storage_type_;
    size_t alignment_;
    std::string initializer_;

public:
    IType(StorageType storage_type = StorageType::CPU_Stack(), size_t alignment = 0, const std::string& initializer = "")
        : storage_type_(storage_type), alignment_(alignment), initializer_(initializer) {};

    virtual ~IType() = default;

    virtual TypeID type_id() const = 0;

    StorageType storage_type() const { return storage_type_; };

    size_t alignment() const { return alignment_; };

    std::string initializer() const { return initializer_; };

    virtual PrimitiveType primitive_type() const = 0;

    virtual bool is_symbol() const = 0;

    virtual bool operator==(const IType& other) const = 0;

    virtual std::unique_ptr<IType> clone() const = 0;

    virtual std::string print() const = 0;

    friend std::ostream& operator<<(std::ostream& os, const IType& type) {
        os << type.print();
        return os;
    };
};

} // namespace types
} // namespace sdfg

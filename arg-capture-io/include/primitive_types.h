#pragma once

#include <cstdint>

namespace arg_capture {

enum class PrimitiveType {
    Void = 0,
    Bool = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    Int128 = 6,
    UInt8 = 7,
    UInt16 = 8,
    UInt32 = 9,
    UInt64 = 10,
    UInt128 = 11,
    Half = 12,
    BFloat = 13,
    Float = 14,
    Double = 15,
    X86_FP80 = 16,
    FP128 = 17,
    PPC_FP128 = 18,
    PRIMITIVE_TYPE_COUNT = 19
};

static const char* primitive_type_names[] = {
    "Void",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "Half",
    "BFloat",
    "Float",
    "Double",
    "X86_FP80",
    "FP128",
    "PPC_FP128"
};

constexpr const char* to_string(PrimitiveType e) {
    if (e < PrimitiveType::Void || e >= PrimitiveType::PRIMITIVE_TYPE_COUNT) {
        return "[unknown]";
    } else {
        return primitive_type_names[static_cast<int32_t>(e)];
    }

}

}
#pragma once
#include <string>

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
    PPC_FP128,
    PRIMITIVE_TYPE_COUNT
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

constexpr const char* to_string(enum PrimitiveType e) {
    if (e < 0 || e >= PRIMITIVE_TYPE_COUNT) {
        return "[unknown]";
    } else {
        return primitive_type_names[e];
    }

}

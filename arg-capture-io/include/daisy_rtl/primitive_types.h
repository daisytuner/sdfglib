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

static const char* primitive_type_names[] = {"Void",  "Bool",   "Int8",     "Int16",  "Int32",     "Int64", "Int128",
                                             "UInt8", "UInt16", "UInt32",   "UInt64", "UInt128",   "Half",  "BFloat",
                                             "Float", "Double", "X86_FP80", "FP128",  "PPC_FP128", ""};

static const char* primitive_type_cppspellings[] = {
    "void", // Void
    "bool", // Bool
    "int8_t", // Int8
    "int16_t", // Int16
    "int32_t", // Int32
    "int64_t", // Int64
    "__int128", // Int128
    "uint8_t", // UInt8
    "uint16_t", // UInt16
    "uint32_t", // UInt32
    "uint64_t", // UInt64
    "unsigned __int128", // UInt128
    "__fp16", // Half (C++ spellings vary: could also be "_Float16")
    "bfloat16_t", // BFloat (requires <bfloat16>, e.g. x86/ARM extensions)
    "float", // Float
    "double", // Double
    "long double", // X86_FP80  (x86 80-bit extended precision)
    "__float128", // FP128     (software quad precision via libquadmath)
    "__ibm128", // PPC_FP128 (IBM double-double)
    ""
};

static const double primitive_type_machine_epsilons[] = {
    0.0, // void
    0.0, // bool
    0.0, // int8_t
    0.0, // int16_t
    0.0, // int32_t
    0.0, // int64_t
    0.0, // __int128
    0.0, // uint8_t
    0.0, // uint16_t
    0.0, // uint32_t
    0.0, // uint64_t
    0.0, // unsigned __int128
    9.765625e-04, // Half (__fp16)  ≈ 2^-10
    7.8125e-03, // BFloat16 (bfloat16_t) ≈ 2^-7
    1.1920929e-07, // Float (float, 32-bit)  ≈ 2^-23
    2.220446049250313e-16, // Double (double, 64-bit) ≈ 2^-52
    1.0842021724855044e-19, // long double (x86_80-bit) ≈ 2^-63
    1.9259299443872354e-34, // __float128 (quad-precision) ≈ 2^-112
    1.0842021724855044e-19, // __ibm128 (double-double) ≈ 2^-63
    0.0
};


constexpr const char* to_string(PrimitiveType e) {
    if (e < PrimitiveType::Void || e >= PrimitiveType::PRIMITIVE_TYPE_COUNT) {
        return "[unknown]";
    } else {
        return primitive_type_names[static_cast<int32_t>(e)];
    }
}

constexpr const char* to_cpp(PrimitiveType e) {
    if (e < PrimitiveType::Void || e >= PrimitiveType::PRIMITIVE_TYPE_COUNT) {
        return "[unknown]";
    } else {
        return primitive_type_cppspellings[static_cast<int32_t>(e)];
    }
}

constexpr double machine_epsilon(PrimitiveType e) {
    if (e < PrimitiveType::Void || e >= PrimitiveType::PRIMITIVE_TYPE_COUNT) {
        return 0.0;
    } else {
        return primitive_type_machine_epsilons[static_cast<int32_t>(e)];
    }
}

} // namespace arg_capture

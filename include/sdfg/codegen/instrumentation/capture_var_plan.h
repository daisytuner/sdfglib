#pragma once
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

enum class CaptureVarType { None, CapRaw, Cap1D, Cap2D };

class CaptureVarPlan {
   public:
    const bool capture_input;
    const bool capture_output;
    const CaptureVarType type;
    const int arg_idx;
    const bool is_external;

    const sdfg::types::PrimitiveType inner_type;
    const sdfg::symbolic::Expression dim1;
    const sdfg::symbolic::Expression dim2;

    CaptureVarPlan(bool capture_input, bool capture_output, CaptureVarType type, int arg_idx, bool is_external,
                   sdfg::types::PrimitiveType inner_type, const sdfg::symbolic::Expression dim1,
                   const sdfg::symbolic::Expression dim2);
};






#pragma once
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

enum CaptureVarType { CAPTURE_NONE, CAPTURE_RAW, CAPTURE_1D, CAPTURE_2D };

class CaptureVarPlan {
   public:
    const bool capture_input;
    const bool capture_output;
    const CaptureVarType type;
    const int argIdx;
    const bool isExternal;

    const sdfg::types::PrimitiveType innerType;
    const sdfg::symbolic::Expression dim1;
    const sdfg::symbolic::Expression dim2;

    CaptureVarPlan(bool capture_input, bool capture_output, CaptureVarType type, int argIdx, bool isExternal,
                   sdfg::types::PrimitiveType innerType, const sdfg::symbolic::Expression dim1,
                   const sdfg::symbolic::Expression dim2);
};






#pragma once
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg{
namespace codegen {


enum class CaptureVarType { None, CapRaw, Cap1D, Cap2D, Cap3D };

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
    const sdfg::symbolic::Expression dim3;

    CaptureVarPlan(bool capture_input, bool capture_output, CaptureVarType type, int arg_idx, bool is_external,
                   sdfg::types::PrimitiveType inner_type, const sdfg::symbolic::Expression dim1 = sdfg::symbolic::Expression(),
                   const sdfg::symbolic::Expression dim2 = sdfg::symbolic::Expression(), const sdfg::symbolic::Expression dim3 = sdfg::symbolic::Expression());
};




}  // namespace codegen
}  // namespace sdfg

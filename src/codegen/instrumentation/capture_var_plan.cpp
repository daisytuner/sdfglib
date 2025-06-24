#include "sdfg/codegen/instrumentation/capture_var_plan.h"

namespace sdfg{
namespace codegen {

CaptureVarPlan::CaptureVarPlan(bool capture_input, bool capture_output, CaptureVarType type, int argIdx, bool isExternal,
                               sdfg::types::PrimitiveType innerType, const sdfg::symbolic::Expression dim1,
                               const sdfg::symbolic::Expression dim2, const sdfg::symbolic::Expression dim3):
    capture_input(capture_input), capture_output(capture_output), type(type), arg_idx(argIdx), is_external(isExternal),
    inner_type(innerType), dim1(dim1), dim2(dim2), dim3(dim3)
{
}

}  // namespace codegen
}  // namespace sdfg
#include "sdfg/codegen/code_generator.h"
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/structure.h"

namespace sdfg {
namespace codegen {

std::tuple<int, types::PrimitiveType> CodeGenerator::analyze_type_rec(
    symbolic::Expression* dims,
    int maxDim,
    int dimIdx,
    const types::IType& type,
    int argIdx,
    const analysis::MemAccessRange* range
) {
    if (dimIdx > maxDim) {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << ": data nesting deeper than " << maxDim
                  << ", ignoring" << std::endl;
        return std::make_tuple(-1, types::Void);
    }

    if (auto* scalarType = dynamic_cast<const types::Scalar*>(&type)) {
        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": scalar" << std::endl;
        return std::make_tuple(dimIdx, scalarType->primitive_type());
    } else if (auto structureType = dynamic_cast<const sdfg::types::Structure*>(&type)) {
        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": struct" << std::endl;
        return std::make_tuple(dimIdx, types::Void);
    } else if (auto* arrayType = dynamic_cast<const types::Array*>(&type)) {
        dims[dimIdx] = arrayType->num_elements();
        auto& inner = arrayType->element_type();

        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": " <<
        // language_extension_.expression(dims[dimIdx]) << std::endl;

        return analyze_type_rec(dims, maxDim, dimIdx + 1, inner, argIdx, range);
    } else if (auto* ptrType = dynamic_cast<const types::Pointer*>(&type)) {
        if (range && !range->is_undefined()) {
            const auto& dim = range->dims()[dimIdx];

            if (symbolic::eq(symbolic::zero(), dim.first)) {
                dims[dimIdx] = symbolic::add(dim.second, symbolic::one());

                auto& inner = ptrType->pointee_type();

                return analyze_type_rec(dims, maxDim, dimIdx + 1, inner, argIdx, range);
            } else {
                std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": has upper bound "
                          << dim.second->__str__() << ", but does not start at 0, cannot capture" << std::endl;
                return std::make_tuple(-2, types::Void);
            }
        } else {
            std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx
                      << ": missing range, cannot capture!" << std::endl;
            return std::make_tuple(-2, types::Void);
        }
    } else {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << ": unsupported type " << type.print()
                  << ", cannot capture!" << std::endl;
        return std::make_tuple(-1, types::Void);
    }
}

bool CodeGenerator::add_capture_plan(
    const std::string& varName,
    int argIdx,
    bool isExternal,
    std::vector<CaptureVarPlan>& plan,
    const analysis::MemAccessRanges& ranges
) {
    auto& type = sdfg_.type(varName);

    const auto* range = ranges.get(varName);

    symbolic::Expression dims[3];

    int dimCount;
    types::PrimitiveType innerPrim;
    std::tie(dimCount, innerPrim) = analyze_type_rec(dims, 3, 0, type, argIdx, range);

    bool isRead = range ? range->saw_read() : true;
    bool isWritten = range ? range->saw_write() : true;

    if (dimCount == 0) {
        plan.emplace_back(isRead, false, CaptureVarType::CapRaw, argIdx, isExternal, innerPrim);
    } else if (dimCount == 1) {
        plan.emplace_back(isRead, isWritten, CaptureVarType::Cap1D, argIdx, isExternal, innerPrim, dims[0]);
    } else if (dimCount == 2) {
        plan.emplace_back(isRead, isWritten, CaptureVarType::Cap2D, argIdx, isExternal, innerPrim, dims[0], dims[1]);
    } else if (dimCount == 3) {
        plan.emplace_back(
            isRead, isWritten, CaptureVarType::Cap3D, argIdx, isExternal, innerPrim, dims[0], dims[1], dims[2]
        );
    } else {
        return false;
    }

    return true;
}

std::unique_ptr<std::vector<CaptureVarPlan>> CodeGenerator::create_capture_plans() {
    auto name = sdfg_.name();

    analysis::AnalysisManager analysis_manager(sdfg_);
    const auto& args = sdfg_.arguments();
    auto& exts = sdfg_.externals();
    const auto& ranges = analysis_manager.get<analysis::MemAccessRanges>();

    auto plan = std::make_unique<std::vector<CaptureVarPlan>>();

    bool working = true;

    int argIdx = -1;
    for (auto& argName : args) {
        ++argIdx;

        working &= add_capture_plan(argName, argIdx, false, *plan.get(), ranges);
    }

    for (auto& argName : exts) {
        ++argIdx;

        working &= add_capture_plan(argName, argIdx, true, *plan.get(), ranges);
    }

    return plan;
}

} // namespace codegen
} // namespace sdfg

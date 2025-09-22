#include "sdfg/codegen/code_generator.h"
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/type_analysis.h"
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
    const analysis::MemAccessRange* range,
    analysis::AnalysisManager& analysis_manager,
    const StructuredSDFG& sdfg,
    std::string var_name
) {
    if (dimIdx > maxDim) {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << ": data nesting deeper than " << maxDim
                  << ", ignoring" << std::endl;
        return std::make_tuple(-1, types::Void);
    }

    if (auto* scalarType = dynamic_cast<const types::Scalar*>(&type)) {
        return std::make_tuple(dimIdx, scalarType->primitive_type());
    } else if (auto structureType = dynamic_cast<const sdfg::types::Structure*>(&type)) {
        return std::make_tuple(dimIdx, types::Void);
    } else if (auto* arrayType = dynamic_cast<const types::Array*>(&type)) {
        dims[dimIdx] = arrayType->num_elements();
        auto& inner = arrayType->element_type();

        return analyze_type_rec(dims, maxDim, dimIdx + 1, inner, argIdx, range, analysis_manager, sdfg, var_name);
    } else if (auto* ptrType = dynamic_cast<const types::Pointer*>(&type)) {
        if (range && !range->is_undefined()) {
            const auto& dim = range->dims()[dimIdx];

            if (symbolic::eq(symbolic::zero(), dim.first)) {
                dims[dimIdx] = symbolic::add(dim.second, symbolic::one());
                const types::IType* inner = nullptr;
                if (ptrType->has_pointee_type()) {
                    inner = &(ptrType->pointee_type());
                } else {
                    if (dimIdx > 0) {
                        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx
                                  << ": missing pointee type for dim > 0, cannot capture!" << std::endl;
                        return std::make_tuple(-2, types::Void);
                    } else {
                        auto& type_analysis = analysis_manager.get<analysis::TypeAnalysis>();
                        auto outer = type_analysis.get_outer_type(var_name);
                        if (outer != nullptr) {
                            if (auto* ptrType_new = dynamic_cast<const types::Pointer*>(outer)) {
                                if (ptrType_new->has_pointee_type()) {
                                    inner = &(ptrType_new->pointee_type());
                                } else {
                                    std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx
                                              << ": missing pointee type, cannot capture!" << std::endl;
                                    return std::make_tuple(-2, types::Void);
                                }
                            }
                        } else {
                            std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx
                                      << ": could not infer type from container, cannot capture!" << std::endl;
                            return std::make_tuple(-2, types::Void);
                        }
                    }
                    if (inner == nullptr) {
                        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx
                                  << ": could not infer type from container, cannot capture!" << std::endl;
                        return std::make_tuple(-2, types::Void);
                    }
                }

                return analyze_type_rec(dims, maxDim, dimIdx + 1, *inner, argIdx, range, analysis_manager, sdfg, var_name);
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
    const types::IType* type = nullptr;
    if (isExternal) {
        auto& pointer_type = dynamic_cast<const types::Pointer&>(sdfg_.type(varName));
        assert(pointer_type.has_pointee_type() && "Externals must have a pointee type");
        type = &pointer_type.pointee_type();
    } else {
        type = &sdfg_.type(varName);
    }

    const auto* range = ranges.get(varName);

    analysis::AnalysisManager analysis_manager(sdfg_);

    symbolic::Expression dims[3];

    int dimCount = 0;
    types::PrimitiveType innerPrim;

    std::tie(dimCount, innerPrim) =
        analyze_type_rec(dims, 3, 0, *type, argIdx, range, analysis_manager, sdfg_, varName);

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
        if (sdfg_.type(argName).type_id() == types::TypeID::Function) {
            continue;
        }
        ++argIdx;

        working &= add_capture_plan(argName, argIdx, true, *plan.get(), ranges);
    }

    if (!working) {
        std::cerr << "In '" << name << "': could not create capture plan, returning empty plan" << std::endl;
        return std::make_unique<std::vector<CaptureVarPlan>>();
    }

    return plan;
}

} // namespace codegen
} // namespace sdfg

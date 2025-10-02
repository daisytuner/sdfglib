#include "sdfg/codegen/code_generator.h"
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/number.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/structure.h"

namespace sdfg {
namespace codegen {

std::tuple<int, types::PrimitiveType> CodeGenerator::analyze_type_rec(
    symbolic::Expression* dims,
    int max_dim,
    int dim_idx,
    const types::IType& type,
    int arg_idx,
    const analysis::MemAccessRange* range,
    analysis::AnalysisManager& analysis_manager,
    const StructuredSDFG& sdfg,
    std::string var_name
) {
    if (dim_idx > max_dim) {
        DEBUG_PRINTLN(
            "In '" << sdfg_.name() << "', arg" << arg_idx << ": data nesting deeper than " << max_dim << ", ignoring"
        );
        return std::make_tuple(-1, types::Void);
    }

    if (auto* scalarType = dynamic_cast<const types::Scalar*>(&type)) {
        return std::make_tuple(dim_idx, scalarType->primitive_type());
    } else if (auto structureType = dynamic_cast<const sdfg::types::Structure*>(&type)) {
        return std::make_tuple(dim_idx, types::Void);
    } else if (auto* arrayType = dynamic_cast<const types::Array*>(&type)) {
        dims[dim_idx] = arrayType->num_elements();
        auto& inner = arrayType->element_type();

        return analyze_type_rec(dims, max_dim, dim_idx + 1, inner, arg_idx, range, analysis_manager, sdfg, var_name);
    } else if (auto* ptrType = dynamic_cast<const types::Pointer*>(&type)) {
        if (!range || range->is_undefined()) {
            DEBUG_PRINTLN(
                "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx << ": missing range, cannot capture!"
            );
            return std::make_tuple(-2, types::Void);
        }
        if (range->dims().size() <= dim_idx) {
            DEBUG_PRINTLN(
                "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx
                       << ": missing dimension in range, cannot capture!"
            );
            return std::make_tuple(-2, types::Void);
        }
        const auto& dim = range->dims().at(dim_idx);
        if (!symbolic::eq(dim.first, symbolic::zero())) {
            DEBUG_PRINTLN(
                "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx << ": has upper bound "
                       << dim.second->__str__() << ", but does not start at 0, cannot capture"
            );
            return std::make_tuple(-2, types::Void);
        }

        dims[dim_idx] = symbolic::add(dim.second, symbolic::one());
        const types::IType* inner = nullptr;
        if (ptrType->has_pointee_type()) {
            inner = &(ptrType->pointee_type());
        } else {
            if (dim_idx > 0) {
                DEBUG_PRINTLN(
                    "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx
                           << ": missing pointee type for dim > 0, cannot capture!"
                );
                return std::make_tuple(-2, types::Void);
            } else {
                auto& type_analysis = analysis_manager.get<analysis::TypeAnalysis>();
                auto outer = type_analysis.get_outer_type(var_name);
                if (outer != nullptr) {
                    if (auto* ptrType_new = dynamic_cast<const types::Pointer*>(outer)) {
                        if (ptrType_new->has_pointee_type()) {
                            inner = &(ptrType_new->pointee_type());
                        } else {
                            DEBUG_PRINTLN(
                                "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx
                                       << ": missing pointee type, cannot capture!"
                            );
                            return std::make_tuple(-2, types::Void);
                        }
                    }
                } else {
                    DEBUG_PRINTLN(
                        "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx
                               << ": could not infer type from container, cannot capture!"
                    );
                    return std::make_tuple(-2, types::Void);
                }
            }
            if (inner == nullptr) {
                DEBUG_PRINTLN(
                    "In '" << sdfg_.name() << "', arg" << arg_idx << " dim" << dim_idx
                           << ": could not infer type from container, cannot capture!"
                );
                return std::make_tuple(-2, types::Void);
            }
        }

        return analyze_type_rec(dims, max_dim, dim_idx + 1, *inner, arg_idx, range, analysis_manager, sdfg, var_name);
    }

    DEBUG_PRINTLN(
        "In '" << sdfg_.name() << "', arg" << arg_idx << ": unsupported type " << type.print() << ", cannot capture!"
    );
    return std::make_tuple(-1, types::Void);
}

bool CodeGenerator::add_capture_plan(
    const std::string& var_name,
    int arg_idx,
    bool is_external,
    std::vector<CaptureVarPlan>& plan,
    const analysis::MemAccessRanges& ranges
) {
    const types::IType* type = nullptr;
    if (is_external) {
        auto& pointer_type = dynamic_cast<const types::Pointer&>(sdfg_.type(var_name));
        assert(pointer_type.has_pointee_type() && "Externals must have a pointee type");
        type = &pointer_type.pointee_type();
    } else {
        type = &sdfg_.type(var_name);
    }

    const auto* range = ranges.get(var_name);

    analysis::AnalysisManager analysis_manager(sdfg_);

    symbolic::Expression dims[3];

    int dim_count = 0;
    types::PrimitiveType inner_type;

    std::tie(dim_count, inner_type) =
        analyze_type_rec(dims, 3, 0, *type, arg_idx, range, analysis_manager, sdfg_, var_name);

    bool is_read = range ? range->saw_read() : true;
    bool is_written = range ? range->saw_write() : true;

    if (dim_count == 0) {
        plan.emplace_back(
            is_read || is_written, is_written && is_external, CaptureVarType::CapRaw, arg_idx, is_external, inner_type
        );
    } else if (dim_count == 1) {
        plan.emplace_back(is_read, is_written, CaptureVarType::Cap1D, arg_idx, is_external, inner_type, dims[0]);
    } else if (dim_count == 2) {
        plan.emplace_back(is_read, is_written, CaptureVarType::Cap2D, arg_idx, is_external, inner_type, dims[0], dims[1]);
    } else if (dim_count == 3) {
        plan.emplace_back(
            is_read, is_written, CaptureVarType::Cap3D, arg_idx, is_external, inner_type, dims[0], dims[1], dims[2]
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

    int arg_idx = -1;
    for (auto& arg_name : args) {
        ++arg_idx;

        working &= add_capture_plan(arg_name, arg_idx, false, *plan.get(), ranges);
    }

    for (auto& arg_name : exts) {
        if (sdfg_.type(arg_name).type_id() == types::TypeID::Function) {
            continue;
        }
        ++arg_idx;

        working &= add_capture_plan(arg_name, arg_idx, true, *plan.get(), ranges);
    }

    if (!working) {
        DEBUG_PRINTLN("In '" << name << "': could not create capture plan, returning empty plan");
        return std::make_unique<std::vector<CaptureVarPlan>>();
    }

    return plan;
}

} // namespace codegen
} // namespace sdfg

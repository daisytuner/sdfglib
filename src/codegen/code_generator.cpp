#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"

namespace sdfg {
namespace codegen {

std::tuple<int, types::PrimitiveType> CodeGenerator::analyze_type_rec(symbolic::Expression* dims, int maxDim, int dimIdx, const types::IType& type, int argIdx) {
    if (dimIdx >= maxDim) {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << ": data nesting deeper than " << maxDim << ", ignoring" << std::endl;
        return std::make_tuple(-1, types::Void);
    }

    if (auto* scalarType = dynamic_cast<const types::Scalar*>(&type)) {
        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": scalar" << std::endl;
        return std::make_tuple(dimIdx, scalarType->primitive_type());
    } else if (auto structureType = dynamic_cast<const sdfg::types::StructureDefinition*>(&type)) {
        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": struct" << std::endl;
        return std::make_tuple(dimIdx, types::Void);
    } else if (auto* arrayType = dynamic_cast<const types::Array*>(&type)) {
        dims[dimIdx] = arrayType->num_elements();
        auto& inner = arrayType->element_type();

        // std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": " << language_extension_.expression(dims[dimIdx]) << std::endl;

        return analyze_type_rec(dims, maxDim, dimIdx+1, inner, argIdx);
    } else if (auto* ptrType = dynamic_cast<const types::Pointer*>(&type)) {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " dim" << dimIdx << ": missing range, assuming 1!" << std::endl;
        dims[dimIdx] = symbolic::one();
        auto& inner = ptrType->pointee_type();

        return analyze_type_rec(dims, maxDim, dimIdx+1, inner, argIdx);
    } else {
        std::cerr << "In '" << sdfg_.name() << "', arg" << argIdx << " unsupported type " << type.print() << " for capture. Skipping!" << std::endl;
        return std::make_tuple(-1, types::Void);
    }
}

void CodeGenerator::add_capture_plan(const std::string& varName, int argIdx, bool isExternal, std::vector<CaptureVarPlan>& plan) {

        bool isRead = true;
        bool isWritten = true;

        auto& type = sdfg_.type(varName);

        symbolic::Expression dims[3];

        int dimCount;
        types::PrimitiveType innerPrim;
        std::tie(dimCount, innerPrim) = analyze_type_rec(dims, 3, 0, type, argIdx);

        if (dimCount == 0) {
            plan.emplace_back(isRead, isWritten, CAPTURE_RAW, argIdx, isExternal, innerPrim, SymEngine::RCP<const SymEngine::Basic>(), SymEngine::RCP<const SymEngine::Basic>());
        } else if (dimCount == 1) {
            plan.emplace_back(isRead, isWritten, CAPTURE_1D, argIdx, isExternal, innerPrim, dims[0], SymEngine::RCP<const SymEngine::Basic>());
        } else if (dimCount == 2) {
            plan.emplace_back(isRead, isWritten, CAPTURE_2D, argIdx, isExternal, innerPrim, dims[0], dims[1]);
        } else {
            std::cerr << "In '" << varName << "', arg" << argIdx << " unsupported type " << type.print() << " for capture. Skipping!" << std::endl;

        }
}

std::unique_ptr<std::vector<CaptureVarPlan>> CodeGenerator::create_capture_plans() {
    auto name = sdfg_.name();

    analysis::AnalysisManager analysis_manager(sdfg_);
    const auto& args = sdfg_.arguments();
    auto& exts = sdfg_.externals();

    auto plan = std::make_unique<std::vector<CaptureVarPlan>>();
    

    int argIdx = -1;
    for (auto& argName : args) {
        ++argIdx;

        add_capture_plan(argName, argIdx, false, *plan.get());
    }

    for (auto& argName : exts) {
        ++argIdx;

        add_capture_plan(argName, argIdx, true, *plan.get());
    }

    return plan;
}

}
}

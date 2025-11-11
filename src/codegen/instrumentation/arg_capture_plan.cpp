#include "sdfg/codegen/instrumentation/arg_capture_plan.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

CaptureVarPlan::CaptureVarPlan(
    bool capture_input,
    bool capture_output,
    CaptureVarType type,
    int argIdx,
    bool isExternal,
    sdfg::types::PrimitiveType innerType,
    const sdfg::symbolic::Expression dim1,
    const sdfg::symbolic::Expression dim2,
    const sdfg::symbolic::Expression dim3
)
    : capture_input(capture_input), capture_output(capture_output), type(type), arg_idx(argIdx),
      is_external(isExternal), inner_type(innerType), dim1(dim1), dim2(dim2), dim3(dim3) {}

bool ArgCapturePlan::should_instrument(const structured_control_flow::ControlFlowNode& node) const {
    return this->nodes_.count(&node);
}

void ArgCapturePlan::begin_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream, LanguageExtension& language_extension
) const {
    stream << "const bool __daisy_cap_en_" << node.element_id() << " = __daisy_capture_enter(__capture_ctx, "
           << node.element_id() << ");" << std::endl;

    stream << "if (__daisy_cap_en_" << node.element_id() << ")";
    stream << "{" << std::endl;

    auto& node_plan = this->nodes_.at(&node);
    this->emit_arg_captures(stream, language_extension, node_plan, false, std::to_string(node.element_id()));

    stream << "}" << std::endl;
}

void ArgCapturePlan::end_instrumentation(
    const structured_control_flow::ControlFlowNode& node, PrettyPrinter& stream, LanguageExtension& language_extension
) const {
    stream << "if (__daisy_cap_en_" << node.element_id() << ")";
    stream << "{" << std::endl;

    auto& node_plan = this->nodes_.at(&node);
    this->emit_arg_captures(stream, language_extension, node_plan, true, std::to_string(node.element_id()));

    stream << "}" << std::endl;

    stream << "__daisy_capture_end(__capture_ctx);" << std::endl;
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<ArgCapturePlan>(
        sdfg,
        std::unordered_map<
            const structured_control_flow::ControlFlowNode*,
            std::unordered_map<std::string, CaptureVarPlan>>{}
    );
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::root(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes;
    auto root_plan = create_capture_plan(sdfg, analysis_manager, sdfg.root());
    nodes.insert({&sdfg.root(), root_plan});
    return std::make_unique<ArgCapturePlan>(sdfg, nodes);
}

std::unique_ptr<ArgCapturePlan> ArgCapturePlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes;
    for (size_t i = 0; i < ols.size(); i++) {
        auto& loop = ols[i];
        auto loop_plan = create_capture_plan(sdfg, analysis_manager, *loop);
        nodes.insert({loop, loop_plan});
    }

    DEBUG_PRINTLN("Created arg capture plan for " + std::to_string(nodes.size()) + " nodes.");

    return std::make_unique<ArgCapturePlan>(sdfg, nodes);
}

void ArgCapturePlan::emit_arg_captures(
    PrettyPrinter& stream,
    LanguageExtension& language_extension,
    const std::unordered_map<std::string, CaptureVarPlan>& plan,
    bool after,
    std::string element_id
) const {
    auto afterBoolStr = after ? "true" : "false";
    for (auto& [argName, varPlan] : plan) {
        auto argIdx = varPlan.arg_idx;
        if ((!after && varPlan.capture_input) || (after && varPlan.capture_output)) {
            switch (varPlan.type) {
                case CaptureVarType::CapRaw: {
                    stream << "\t__daisy_capture_raw(" << "__capture_ctx, " << argIdx << ", " << "&" << argName << ", "
                           << "sizeof(" << argName << "), " << varPlan.inner_type << ", " << afterBoolStr << ", "
                           << element_id << ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap1D: {
                    stream << "\t__daisy_capture_1d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                           << "sizeof(" << language_extension.primitive_type(varPlan.inner_type) << "), "
                           << varPlan.inner_type << ", " << language_extension.expression(varPlan.dim1) << ", "
                           << afterBoolStr << ", " << element_id << ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap2D: {
                    stream << "\t__daisy_capture_2d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                           << "sizeof(" << language_extension.primitive_type(varPlan.inner_type) << "), "
                           << varPlan.inner_type << ", " << language_extension.expression(varPlan.dim1) << ", "
                           << language_extension.expression(varPlan.dim2) << ", " << afterBoolStr << ", " << element_id
                           << ");" << std::endl;
                    break;
                }
                case CaptureVarType::Cap3D: {
                    stream << "\t__daisy_capture_3d(" << "__capture_ctx, " << argIdx << ", " << argName << ", "
                           << "sizeof(" << language_extension.primitive_type(varPlan.inner_type) << "), "
                           << varPlan.inner_type << ", " << language_extension.expression(varPlan.dim1) << ", "
                           << language_extension.expression(varPlan.dim2) << ", "
                           << language_extension.expression(varPlan.dim3) << ", " << afterBoolStr << ", " << element_id
                           << ");" << std::endl;
                    break;
                }
                default: {
                    DEBUG_PRINTLN(
                        "Unknown capture type " << static_cast<int>(varPlan.type) << " for arg " << argIdx << " at "
                                                << (after ? "result" : "input") << " time"
                    );
                    break;
                }
            }
        }
    }
}

std::tuple<int, types::PrimitiveType> ArgCapturePlan::analyze_type_rec(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    symbolic::Expression* dims,
    int max_dim,
    int dim_idx,
    const types::IType& type,
    int arg_idx,
    const analysis::MemAccessRange* range,
    std::string var_name
) {
    if (dim_idx > max_dim) {
        DEBUG_PRINTLN("arg" << arg_idx << ": data nesting deeper than " << max_dim << ", ignoring");
        return std::make_tuple(-1, types::Void);
    }

    if (auto* scalarType = dynamic_cast<const types::Scalar*>(&type)) {
        return std::make_tuple(dim_idx, scalarType->primitive_type());
    } else if (auto structureType = dynamic_cast<const sdfg::types::Structure*>(&type)) {
        return std::make_tuple(dim_idx, types::Void);
    } else if (auto* arrayType = dynamic_cast<const types::Array*>(&type)) {
        dims[dim_idx] = arrayType->num_elements();
        auto& inner = arrayType->element_type();

        return analyze_type_rec(sdfg, analysis_manager, dims, max_dim, dim_idx + 1, inner, arg_idx, range, var_name);
    } else if (auto* ptrType = dynamic_cast<const types::Pointer*>(&type)) {
        if (!range || range->is_undefined()) {
            DEBUG_PRINTLN("arg" << arg_idx << " dim" << dim_idx << ": missing range, cannot capture!");
            return std::make_tuple(-2, types::Void);
        }
        if (range->dims().size() <= dim_idx) {
            DEBUG_PRINTLN("arg" << arg_idx << " dim" << dim_idx << ": missing dimension in range, cannot capture!");
            return std::make_tuple(-2, types::Void);
        }
        const auto& dim = range->dims().at(dim_idx);
        if (!symbolic::eq(dim.first, symbolic::zero())) {
            DEBUG_PRINTLN(
                "arg" << arg_idx << " dim" << dim_idx << ": has upper bound " << dim.second->__str__()
                      << ", but does not start at 0, cannot capture"
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
                    "arg" << arg_idx << " dim" << dim_idx << ": missing pointee type for dim > 0, cannot capture!"
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
                                "arg" << arg_idx << " dim" << dim_idx << ": missing pointee type, cannot capture!"
                            );
                            return std::make_tuple(-2, types::Void);
                        }
                    }
                } else {
                    DEBUG_PRINTLN(
                        "arg" << arg_idx << " dim" << dim_idx
                              << ": could not infer type from container, cannot capture!"
                    );
                    return std::make_tuple(-2, types::Void);
                }
            }
            if (inner == nullptr) {
                DEBUG_PRINTLN(
                    "arg" << arg_idx << " dim" << dim_idx << ": could not infer type from container, cannot capture!"
                );
                return std::make_tuple(-2, types::Void);
            }
        }

        return analyze_type_rec(sdfg, analysis_manager, dims, max_dim, dim_idx + 1, *inner, arg_idx, range, var_name);
    }

    DEBUG_PRINTLN("arg" << arg_idx << ": unsupported type " << type.print() << ", cannot capture!");
    return std::make_tuple(-1, types::Void);
}

bool ArgCapturePlan::add_capture_plan(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    const std::string& var_name,
    int arg_idx,
    bool is_external,
    std::unordered_map<std::string, CaptureVarPlan>& plan,
    analysis::MemAccessRanges& ranges
) {
    const types::IType* type = nullptr;
    if (is_external) {
        auto& pointer_type = dynamic_cast<const types::Pointer&>(sdfg.type(var_name));
        assert(pointer_type.has_pointee_type() && "Externals must have a pointee type");
        type = &pointer_type.pointee_type();
    } else {
        type = &sdfg.type(var_name);
    }

    const auto* range = ranges.get(var_name, node, {var_name});

    symbolic::Expression dims[3];

    int dim_count = 0;
    types::PrimitiveType inner_type;

    std::tie(dim_count, inner_type) =
        analyze_type_rec(sdfg, analysis_manager, dims, 3, 0, *type, arg_idx, range, var_name);
    if (inner_type == types::Void || dim_count < 0 || dim_count > 3) {
        return false;
    }

    bool is_read = range ? range->saw_read() : true;
    bool is_written = range ? range->saw_write() : true;
    if (dim_count == 0) {
        plan.insert(
            {var_name,
             CaptureVarPlan(
                 is_read || is_written,
                 is_written && (is_read || is_external),
                 CaptureVarType::CapRaw,
                 arg_idx,
                 is_external,
                 inner_type
             )}
        );
    } else if (dim_count == 1) {
        plan.insert(
            {var_name,
             CaptureVarPlan(is_read, is_written, CaptureVarType::Cap1D, arg_idx, is_external, inner_type, dims[0])}
        );
    } else if (dim_count == 2) {
        plan.insert(
            {var_name,
             CaptureVarPlan(is_read, is_written, CaptureVarType::Cap2D, arg_idx, is_external, inner_type, dims[0], dims[1])
            }
        );
    } else if (dim_count == 3) {
        plan.insert(
            {var_name,
             CaptureVarPlan(
                 is_read, is_written, CaptureVarType::Cap3D, arg_idx, is_external, inner_type, dims[0], dims[1], dims[2]
             )}
        );
    }

    DEBUG_PRINTLN("Successfully added capture plan ");
    return true;
}

std::unordered_map<std::string, CaptureVarPlan> ArgCapturePlan::create_capture_plan(
    StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
) {
    auto arguments = find_arguments(sdfg, analysis_manager, node);

    // Sort arguments to have a deterministic order
    std::vector<std::string> args;
    for (auto& [arg_name, flags] : arguments) {
        args.push_back(arg_name);
    }
    std::sort(args.begin(), args.end());
    DEBUG_PRINTLN("Found " + std::to_string(args.size()) + " arguments for region " + std::to_string(node.element_id()));

    // Determine ranges per arguments
    auto& ranges_analysis = analysis_manager.get<analysis::MemAccessRanges>();
    bool working = true;
    int arg_idx = -1;
    std::unordered_map<std::string, CaptureVarPlan> plan;
    for (auto& arg_name : args) {
        if (sdfg.is_external(arg_name)) {
            continue;
        }

        ++arg_idx;
        working &= add_capture_plan(sdfg, analysis_manager, node, arg_name, arg_idx, false, plan, ranges_analysis);
    }

    for (auto& arg_name : args) {
        if (!sdfg.is_external(arg_name) || sdfg.type(arg_name).type_id() == types::TypeID::Function) {
            continue;
        }

        ++arg_idx;
        working &= add_capture_plan(sdfg, analysis_manager, node, arg_name, arg_idx, false, plan, ranges_analysis);
    }
    if (!working) {
        DEBUG_PRINTLN("could not create capture plan, returning empty plan");
        return std::unordered_map<std::string, CaptureVarPlan>{};
    }

    std::cout << "Created capture plan for " << plan.size() << " arguments." << std::endl;
    return plan;
}

std::unordered_map<std::string, std::pair<bool, bool>> ArgCapturePlan::find_arguments(
    StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
) {
    // Infer arguments of scope
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView scope_users(users, node);

    std::unordered_map<std::string, std::pair<bool, bool>> all_containers;
    for (auto& user : scope_users.uses()) {
        if (symbolic::is_nullptr(symbolic::symbol(user->container()))) {
            continue;
        }

        bool is_read, is_write;
        switch (user->use()) {
            case analysis::READ:
                is_read = true;
                is_write = false;
                break;
            case analysis::WRITE:
                is_read = false;
                is_write = true;
                break;
            case analysis::MOVE:
            case analysis::VIEW:
            default:
                is_read = true;
                is_write = true;
        }
        auto it = all_containers.insert({user->container(), {is_read, is_write}});
        if (!it.second) {
            it.first->second.first |= is_read;
            it.first->second.second |= is_write;
        }
    }

    std::unordered_map<std::string, std::pair<bool, bool>> arguments;
    for (auto& [container, flags] : all_containers) {
        if (sdfg.is_argument(container) || sdfg.is_external(container)) {
            arguments.insert({container, {flags.first, flags.second}});
            continue;
        }

        size_t total_uses = users.uses(container).size();
        size_t scope_uses = scope_users.uses(container).size();

        if (scope_uses < total_uses) {
            arguments.insert({container, {flags.first, flags.second}});
        }
    }

    return arguments;
}

} // namespace codegen
} // namespace sdfg

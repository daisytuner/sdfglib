#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/flop_analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
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

    CaptureVarPlan(
        bool capture_input,
        bool capture_output,
        CaptureVarType type,
        int arg_idx,
        bool is_external,
        sdfg::types::PrimitiveType inner_type,
        const sdfg::symbolic::Expression dim1 = sdfg::symbolic::Expression(),
        const sdfg::symbolic::Expression dim2 = sdfg::symbolic::Expression(),
        const sdfg::symbolic::Expression dim3 = sdfg::symbolic::Expression()
    );
};

class ArgCapturePlan {
private:
    StructuredSDFG& sdfg_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unordered_map<std::string, CaptureVarPlan>>
        nodes_;

    static bool add_capture_plan(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        const std::string& var_name,
        int arg_idx,
        bool is_external,
        std::unordered_map<std::string, CaptureVarPlan>& plan,
        analysis::MemAccessRanges& ranges
    );

    static std::tuple<int, types::PrimitiveType> analyze_type_rec(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        symbolic::Expression* dims,
        int max_dim,
        int dim_idx,
        const types::IType& type,
        int arg_idx,
        const analysis::MemAccessRange* range,
        std::string var_name
    );

public:
    ArgCapturePlan(
        StructuredSDFG& sdfg,
        const std::unordered_map<
            const structured_control_flow::ControlFlowNode*,
            std::unordered_map<std::string, CaptureVarPlan>>& nodes
    )
        : sdfg_(sdfg), nodes_(nodes) {}

    bool is_empty() const { return nodes_.empty(); }

    bool should_instrument(const structured_control_flow::ControlFlowNode& node) const;

    void begin_instrumentation(
        const structured_control_flow::ControlFlowNode& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension
    ) const;

    void end_instrumentation(
        const structured_control_flow::ControlFlowNode& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension
    ) const;

    void emit_arg_captures(
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const std::unordered_map<std::string, CaptureVarPlan>& plan,
        bool after
    ) const;

    static std::unique_ptr<ArgCapturePlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<ArgCapturePlan> root(StructuredSDFG& sdfg);

    static std::unique_ptr<ArgCapturePlan> outermost_loops_plan(StructuredSDFG& sdfg);

    static std::unordered_map<std::string, CaptureVarPlan> create_capture_plan(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node
    );

    static std::unordered_map<std::string, std::pair<bool, bool>> find_arguments(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node
    );
};

} // namespace codegen
} // namespace sdfg

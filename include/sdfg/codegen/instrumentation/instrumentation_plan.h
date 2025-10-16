#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/flop_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

enum InstrumentationEventType {
    CPU = 0,
    CUDA = 1,
};

class InstrumentationPlan {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> flops_;

protected:
    StructuredSDFG& sdfg_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType> nodes_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, size_t> loopnest_indices_;

public:
    InstrumentationPlan(
        StructuredSDFG& sdfg,
        const std::unordered_map<const structured_control_flow::ControlFlowNode*, InstrumentationEventType>& nodes,
        const std::unordered_map<const structured_control_flow::ControlFlowNode*, size_t>& loopnest_indices = {}
    )
        : sdfg_(sdfg), nodes_(nodes), loopnest_indices_(loopnest_indices) {
        analysis::AnalysisManager analysis_manager(this->sdfg_);
        auto& flop_analysis = analysis_manager.get<analysis::FlopAnalysis>();
        this->flops_ = flop_analysis.get();
    }

    InstrumentationPlan(const InstrumentationPlan& other) = delete;
    InstrumentationPlan(InstrumentationPlan&& other) = delete;

    InstrumentationPlan& operator=(const InstrumentationPlan& other) = delete;
    InstrumentationPlan& operator=(InstrumentationPlan&& other) = delete;

    void update(const structured_control_flow::ControlFlowNode& node, InstrumentationEventType event_type);

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

    static std::unique_ptr<InstrumentationPlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<InstrumentationPlan> outermost_loops_plan(StructuredSDFG& sdfg);
};

} // namespace codegen
} // namespace sdfg

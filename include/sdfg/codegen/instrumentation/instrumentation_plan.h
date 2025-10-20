#pragma once

#include <unordered_map>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/flop_analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace codegen {

enum InstrumentationEventType { CPU = 0, CUDA = 1, TENSTORRENT = 2, H2D = 3, D2H = 4 };

class InstrumentationPlan {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Expression> flops_;

protected:
    StructuredSDFG& sdfg_;
    std::unordered_map<const Element*, InstrumentationEventType> nodes_;

public:
    InstrumentationPlan(StructuredSDFG& sdfg, const std::unordered_map<const Element*, InstrumentationEventType>& nodes)
        : sdfg_(sdfg), nodes_(nodes) {
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

    bool should_instrument(const Element& node) const;

    void begin_instrumentation(
        const Element& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const InstrumentationInfo& info
    ) const;

    void end_instrumentation(
        const Element& node,
        PrettyPrinter& stream,
        LanguageExtension& language_extension,
        const InstrumentationInfo& info
    ) const;

    static std::unique_ptr<InstrumentationPlan> none(StructuredSDFG& sdfg);

    static std::unique_ptr<InstrumentationPlan> outermost_loops_plan(StructuredSDFG& sdfg);
};

class LibNodeFinder : public visitor::StructuredSDFGVisitor {
private:
    std::vector<const data_flow::LibraryNode*> lib_nodes_H2D;
    std::vector<const data_flow::LibraryNode*> lib_nodes_D2H;

public:
    const std::vector<const data_flow::LibraryNode*>& get_lib_nodes_H2D() const { return lib_nodes_H2D; }
    const std::vector<const data_flow::LibraryNode*>& get_lib_nodes_D2H() const { return lib_nodes_D2H; }

    virtual bool accept(structured_control_flow::Block& node);

    virtual bool accept(structured_control_flow::Sequence& node);

    virtual bool accept(structured_control_flow::Return& node);

    virtual bool accept(structured_control_flow::IfElse& node);

    virtual bool accept(structured_control_flow::For& node);

    virtual bool accept(structured_control_flow::While& node);

    virtual bool accept(structured_control_flow::Continue& node);

    virtual bool accept(structured_control_flow::Break& node);

    virtual bool accept(structured_control_flow::Map& node);

    LibNodeFinder(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}
};

} // namespace codegen
} // namespace sdfg

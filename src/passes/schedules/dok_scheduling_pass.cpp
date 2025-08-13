#include "sdfg/passes/schedules/dok_scheduling_pass.h"
#include "sdfg/analysis/degrees_of_knowledge_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"


namespace sdfg {
namespace passes {

DOKScheduling::DOKScheduling() {}

std::string DOKScheduling::name() { return "DOKScheduling"; }

bool DOKScheduling::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& dok_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto outermost_maps = loop_analysis.outermost_maps();

    // Schedule outermost maps
    for (const auto& node : outermost_maps) {
        Map* map = static_cast<Map*>(node);
        // test for dynamic behaviour
        auto balance = dok_analysis.balance_of_a_map(*map);
        auto load = dok_analysis.load_of_a_map(*map);
        auto size = dok_analysis.size_of_a_map(*map);
        auto number = dok_analysis.number_of_maps(*map);

        // enable fat bin generation
    }

    return false;
}


} // namespace passes
} // namespace sdfg

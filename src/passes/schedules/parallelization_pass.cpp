#include "sdfg/passes/schedules/parallelization_pass.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/transformations/parallelization.h"

namespace sdfg {
namespace passes {

std::string ParallelizationPass::name() { return "Parallelization"; }

bool ParallelizationPass::run_pass(builder::StructuredSDFGBuilder& builder,
                                   analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto& loop : loop_analysis.outermost_loops()) {
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            transformations::Parallelization transformation(*map);
            if (transformation.can_be_applied(builder, analysis_manager)) {
                transformation.apply(builder, analysis_manager);
                applied = true;
            }
        }
    }

    return applied;
}

}  // namespace passes
}  // namespace sdfg

#include "sdfg/passes/scheduling_pass.h"

#include <gtest/gtest.h>

#include "../fixtures/polybench.h"

#include <sdfg/passes/pipeline.h>

using namespace sdfg;

TEST(SchedulingPassTest, fdtd_2d_OpenMP) {
    auto fixture_ = fdtd_2d();
    builder::StructuredSDFGBuilder builder(fixture_);
    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(builder, analysis_manager);

    passes::SchedulingPass scheduling_pass("openmp", "server", false);
    scheduling_pass.run(builder, analysis_manager);

    sdfg::structured_control_flow::For* ols =
        dynamic_cast<sdfg::structured_control_flow::For*>(&builder.subject().root().at(0).first);
    EXPECT_NE(ols, nullptr);

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto second_level_loops = loop_analysis.children(ols);
    EXPECT_EQ(second_level_loops.size(), 4);
    for (auto& loop : second_level_loops) {
        sdfg::structured_control_flow::Map* sls = dynamic_cast<sdfg::structured_control_flow::Map*>(loop);
        EXPECT_NE(sls, nullptr);
        EXPECT_EQ(sls->schedule_type().value(), "CPU_PARALLEL");
    }
}

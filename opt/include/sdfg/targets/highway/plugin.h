#pragma once

#include <string>

#include "sdfg/targets/highway/codegen/highway_map_dispatcher.h"
#include "sdfg/targets/highway/schedule.h"

namespace sdfg {
namespace highway {

inline void register_highway_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_Highway::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<HighwayMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );
}

} // namespace highway
} // namespace sdfg

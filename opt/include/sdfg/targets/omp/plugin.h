#pragma once

#include <string>

#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"
#include "sdfg/targets/omp/schedule.h"

namespace sdfg {
namespace omp {

inline void register_omp_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_OMP::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<OMPMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );
}

} // namespace omp
} // namespace sdfg

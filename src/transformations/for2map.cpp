#include "sdfg/transformations/for2map.h"

#include "sdfg/analysis/data_parallelism_analysis.h"

namespace sdfg {
namespace transformations {

For2Map::For2Map(structured_control_flow::Sequence& parent, structured_control_flow::For& loop)
    : parent_(parent), loop_(loop) {

      };

std::string For2Map::name() { return "For2Map"; };

bool For2Map::can_be_applied(Schedule& schedule) {
    auto& sdfg = schedule.sdfg();
    auto& analysis_manager = schedule.analysis_manager();

    // Criterion: loop must be data-parallel w.r.t containers
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop_);
    if (dependencies.size() == 0) {
        return false;
    }

    // Criterion: index variable must be normalizable

    // TODO: @Adrian: Check if the loop is a map

    return true;
};

void For2Map::apply(Schedule& schedule) {
    auto& sdfg = schedule.sdfg();

    // TODO: @Adrian: Check if the loop is a map
};

}  // namespace transformations
}  // namespace sdfg

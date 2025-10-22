#include "sdfg/structured_control_flow/map.h"
#include <string>

#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/expression.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace structured_control_flow {

Map::
    Map(size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type)
    : StructuredLoop(element_id, debug_info, indvar, init, update, condition), schedule_type_(schedule_type) {};

void Map::validate(const Function& function) const { StructuredLoop::validate(function); };

const ScheduleType& Map::schedule_type() const { return this->schedule_type_; };

void ScheduleType_CPU_Parallel::num_threads(ScheduleType& schedule, const symbolic::Expression num_threads) {
    serializer::JSONSerializer serializer;
    schedule.set_property("num_threads", serializer.expression(num_threads));
}
const symbolic::Expression ScheduleType_CPU_Parallel::num_threads(const ScheduleType& schedule) {
    if (schedule.properties().find("num_threads") == schedule.properties().end()) {
        return SymEngine::null;
    }
    std::string expr_str = schedule.properties().at("num_threads");
    auto expr = symbolic::parse(expr_str);
    return expr;
}
void ScheduleType_CPU_Parallel::omp_schedule(ScheduleType& schedule, OpenMPSchedule omp_schedule) {
    schedule.set_property("omp_schedule", std::to_string(omp_schedule));
}
OpenMPSchedule ScheduleType_CPU_Parallel::omp_schedule(const ScheduleType& schedule) {
    if (schedule.properties().find("omp_schedule") == schedule.properties().end()) {
        return OpenMPSchedule::Static;
    }
    return static_cast<OpenMPSchedule>(std::stoi(schedule.properties().at("omp_schedule")));
}

void ScheduleType_CPU_Parallel::tasking(ScheduleType& schedule, const bool enable) {
    schedule.set_property("tasking", enable ? "1" : "0");
}
bool ScheduleType_CPU_Parallel::tasking(const ScheduleType& schedule) {
    if (schedule.properties().find("tasking") == schedule.properties().end()) {
        return false;
    }
    return schedule.properties().at("tasking") == "1";
}

void ScheduleType_CPU_Parallel::chunk_size(ScheduleType& schedule, const symbolic::Expression chunk_size) {
    serializer::JSONSerializer serializer;
    schedule.set_property("chunk_size", serializer.expression(chunk_size));
}

const symbolic::Expression ScheduleType_CPU_Parallel::chunk_size(const ScheduleType& schedule) {
    if (schedule.properties().find("chunk_size") == schedule.properties().end()) {
        return SymEngine::null;
    }
    std::string expr_str = schedule.properties().at("chunk_size");
    auto expr = symbolic::parse(expr_str);
    return expr;
}

} // namespace structured_control_flow
} // namespace sdfg

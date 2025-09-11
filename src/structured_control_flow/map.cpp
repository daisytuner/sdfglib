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

void Map::validate(const Function& function) const { this->root_->validate(function); };

ScheduleType& Map::schedule_type() { return this->schedule_type_; };

const ScheduleType& Map::schedule_type() const { return this->schedule_type_; };

void ScheduleType_CPU_Parallel::num_threads(ScheduleType& schedule, const symbolic::Expression& num_threads) {
    serializer::JSONSerializer serializer;
    schedule.set_property("num_threads", serializer.expression(num_threads));
}
const symbolic::Expression ScheduleType_CPU_Parallel::num_threads(const ScheduleType& schedule) {
    serializer::JSONSerializer serializer;
    if (schedule.properties().find("num_threads") == schedule.properties().end()) {
        return SymEngine::null;
    }
    std::string expr_str = schedule.properties().at("num_threads");
    SymEngine::Expression expr(expr_str);
    return expr;
}
void ScheduleType_CPU_Parallel::set_dynamic(ScheduleType& schedule) { schedule.set_property("dynamic", "true"); }
bool ScheduleType_CPU_Parallel::dynamic(const ScheduleType& schedule) {
    if (schedule.properties().find("dynamic") == schedule.properties().end()) {
        return false;
    }
    return schedule.properties().at("dynamic") == "true";
}

} // namespace structured_control_flow
} // namespace sdfg

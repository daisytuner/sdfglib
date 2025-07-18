#include "sdfg/structured_control_flow/map.h"

#include "sdfg/symbolic/symbolic.h"

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

void Map::validate() const { this->root_->validate(); };

ScheduleType& Map::schedule_type() { return this->schedule_type_; };

const ScheduleType& Map::schedule_type() const { return this->schedule_type_; };

} // namespace structured_control_flow
} // namespace sdfg

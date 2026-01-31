#include "printf_target.h"

namespace sdfg {
namespace printf_target {

std::string ScheduleType_Printf::target_stream(const structured_control_flow::ScheduleType& schedule) {
    auto it = schedule.properties().find("target_stream");
    if (it != schedule.properties().end()) {
        return it->second;
    }
    return "stdout"; // Default stream
}

void ScheduleType_Printf::target_stream(structured_control_flow::ScheduleType& schedule, const std::string& stream) {
    schedule.set_property("target_stream", stream);
}

} // namespace printf_target
} // namespace sdfg

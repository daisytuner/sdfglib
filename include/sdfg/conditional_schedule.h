#pragma once

#include "sdfg/schedule.h"

namespace sdfg {

class ConditionalSchedule {
   private:
    std::vector<std::unique_ptr<Schedule>> schedules_;

   public:
    ConditionalSchedule(std::unique_ptr<StructuredSDFG>& sdfg);

    ConditionalSchedule(const ConditionalSchedule& ConditionalSchedule) = delete;
    ConditionalSchedule& operator=(const ConditionalSchedule&) = delete;

    std::string name() const;

    DebugInfo debug_info() const;

    const Schedule& schedule(size_t index) const;

    Schedule& schedule(size_t index);

    symbolic::Condition condition(size_t index) const;

    size_t size() const;

    void push_front(std::unique_ptr<Schedule>& schedule);
};
}  // namespace sdfg

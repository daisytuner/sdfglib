#include "sdfg/conditional_schedule.h"

namespace sdfg {

ConditionalSchedule::ConditionalSchedule(std::unique_ptr<StructuredSDFG>& sdfg) : schedules_() {
    schedules_.push_back(std::make_unique<Schedule>(sdfg, symbolic::Assumptions()));
};

std::string ConditionalSchedule::name() const { return this->schedules_.at(0)->sdfg().name(); };

DebugInfo ConditionalSchedule::debug_info() const {
    return this->schedules_.at(0)->sdfg().debug_info();
};

const Schedule& ConditionalSchedule::schedule(size_t index) const {
    return *this->schedules_.at(index);
};

Schedule& ConditionalSchedule::schedule(size_t index) { return *this->schedules_.at(index); };

symbolic::Condition ConditionalSchedule::condition(size_t index) const {
    auto condition = symbolic::__true__();
    auto assumptions = this->schedules_.at(index)->assumptions();
    for (auto& assum : assumptions) {
        auto sym = assum.first;
        assert(assum.second.map() == SymEngine::null);

        auto lb = assum.second.lower_bound();
        auto ub = assum.second.upper_bound();
        auto cond = symbolic::And(symbolic::Le(sym, ub), symbolic::Ge(sym, lb));

        condition = symbolic::And(condition, cond);
    }
    return condition;
};

size_t ConditionalSchedule::size() const { return this->schedules_.size(); };

void ConditionalSchedule::push_front(std::unique_ptr<Schedule>& schedule) {
    this->schedules_.insert(this->schedules_.begin(), std::move(schedule));
};

}  // namespace sdfg

#pragma once

#include <memory>
#include <unordered_map>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class ScheduleType {
private:
    std::unordered_map<std::string, std::string> properties_;
    std::string value_;

public:
    ScheduleType(std::string value) : value_(value) {}
    const std::string& value() const { return value_; }
    const std::unordered_map<std::string, std::string>& properties() const { return properties_; }
    void set_property(const std::string& key, const std::string& value) {
        if (properties_.find(key) == properties_.end()) {
            properties_.insert({key, value});
            return;
        }
        properties_.at(key) = value;
    }

    void operator=(const ScheduleType& rhs) {
        value_ = rhs.value_;
        properties_.clear();
        for (const auto& entry : rhs.properties_) {
            properties_.insert(entry);
        }
    }
};

class ScheduleType_Sequential {
public:
    static const std::string value() { return "SEQUENTIAL"; }
    static ScheduleType create() { return ScheduleType(value()); }
};

class ScheduleType_CPU_Parallel {
public:
    static void num_threads(ScheduleType& schedule, const symbolic::Expression& num_threads);
    static const symbolic::Expression num_threads(const ScheduleType& schedule);
    static void set_dynamic(ScheduleType& schedule);
    static bool dynamic(const ScheduleType& schedule);
    static const std::string value() { return "CPU_PARALLEL"; }
    static ScheduleType create() { return ScheduleType(value()); }
};

class Map : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    ScheduleType schedule_type_;

    Map(size_t element_id,
        const DebugInfoRegion& debug_info,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type);

public:
    Map(const Map& node) = delete;
    Map& operator=(const Map&) = delete;

    void validate(const Function& function) const override;

    ScheduleType& schedule_type();

    const ScheduleType& schedule_type() const;
};

} // namespace structured_control_flow
} // namespace sdfg

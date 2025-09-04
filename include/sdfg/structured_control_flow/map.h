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

public:
    static const std::string value() { return "BASE"; }
    std::unordered_map<std::string, std::string> properties() const { return properties_; }
};

class ScheduleType_Sequential : public ScheduleType {
public:
    static const std::string value() { return "SEQUENTIAL"; }
};

class ScheduleType_CPU_Parallel : public ScheduleType {
public:
    ScheduleType_CPU_Parallel();
    void num_threads(const symbolic::Expression& num_threads);
    const symbolic::Expression& num_threads() const;
    void set_dynamic();
    bool dynamic() const;
    static const std::string value() { return "CPU_PARALLEL"; }
};

class Map : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    ScheduleType schedule_type_;

    Map(size_t element_id,
        const DebugInfo& debug_info,
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

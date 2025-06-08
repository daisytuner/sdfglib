#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

typedef StringEnum ScheduleType;
inline constexpr ScheduleType ScheduleType_Sequential{"SEQUENTIAL"};
inline constexpr ScheduleType ScheduleType_CPU_Parallel{"CPU_PARALLEL"};

class Map : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Symbol indvar_;
    symbolic::Expression num_iterations_;

    ScheduleType schedule_type_;

    symbolic::Expression init_;
    symbolic::Expression update_;
    symbolic::Condition condition_;

    std::unique_ptr<Sequence> root_;

    Map(const DebugInfo& debug_info, symbolic::Symbol indvar, symbolic::Expression num_iterations,
        ScheduleType schedule_type);

   public:
    Map(const Map& node) = delete;
    Map& operator=(const Map&) = delete;

    const symbolic::Symbol& indvar() const override;

    symbolic::Symbol& indvar();

    const symbolic::Expression& init() const override;

    const symbolic::Expression& update() const override;

    const symbolic::Condition& condition() const override;

    const symbolic::Expression& num_iterations() const;

    symbolic::Expression& num_iterations();

    ScheduleType schedule_type() const;

    Sequence& root() const override;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg
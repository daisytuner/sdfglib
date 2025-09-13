#include "sdfg/structured_control_flow/map.h"
#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::structured_control_flow {

TEST(CPU_PARALLELScheduleTypeTest, ScheduleTypeTest) {
    ScheduleType cpu_parallel_schedule = ScheduleType_CPU_Parallel::create();

    EXPECT_EQ(cpu_parallel_schedule.value(), ScheduleType_CPU_Parallel::value());

    ScheduleType_CPU_Parallel::num_threads(cpu_parallel_schedule, symbolic::integer(256));

    EXPECT_TRUE(symbolic::eq(ScheduleType_CPU_Parallel::num_threads(cpu_parallel_schedule), symbolic::integer(256)));

    EXPECT_EQ(ScheduleType_CPU_Parallel::omp_schedule(cpu_parallel_schedule), OpenMPSchedule::Static);
    ScheduleType_CPU_Parallel::omp_schedule(cpu_parallel_schedule, OpenMPSchedule::Dynamic);
    EXPECT_EQ(ScheduleType_CPU_Parallel::omp_schedule(cpu_parallel_schedule), OpenMPSchedule::Dynamic);
}

} // namespace sdfg::structured_control_flow

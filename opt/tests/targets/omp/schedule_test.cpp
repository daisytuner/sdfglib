#include "sdfg/targets/omp/schedule.h"

#include "sdfg/serializer/json_serializer.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(OMPScheduleTypeTest, OMPAttributes) {
    ScheduleType omp_schedule = omp::ScheduleType_OMP::create();

    EXPECT_EQ(omp_schedule.value(), omp::ScheduleType_OMP::value());

    omp::ScheduleType_OMP::num_threads(omp_schedule, symbolic::integer(256));

    EXPECT_TRUE(symbolic::eq(omp::ScheduleType_OMP::num_threads(omp_schedule), symbolic::integer(256)));

    EXPECT_EQ(omp::ScheduleType_OMP::omp_schedule(omp_schedule), omp::OpenMPSchedule::Static);
    omp::ScheduleType_OMP::omp_schedule(omp_schedule, omp::OpenMPSchedule::Dynamic);
    EXPECT_EQ(omp::ScheduleType_OMP::omp_schedule(omp_schedule), omp::OpenMPSchedule::Dynamic);
}

TEST(OMPScheduleTypeTest, SerializeDeserialize) {
    ScheduleType sched_type = omp::ScheduleType_OMP::create();

    sched_type.set_property("num_threads", "4");

    sdfg::serializer::JSONSerializer serializer;

    // Serialize the schedule type
    nlohmann::json j;
    serializer.schedule_type_to_json(j, sched_type);

    // Deserialize the schedule type
    ScheduleType sched_type_new = serializer.json_to_schedule_type(j);

    EXPECT_EQ(sched_type_new.value(), sched_type.value());
    EXPECT_EQ(sched_type_new.properties().size(), sched_type.properties().size());
    EXPECT_EQ(sched_type_new.properties().at("num_threads"), sched_type.properties().at("num_threads"));
}

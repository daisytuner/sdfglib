#include "sdfg/conditional_schedule.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
using namespace sdfg;

TEST(ConditionalScheduleTest, Default) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType::CPU);

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("N", desc, true);
    builder.add_container("i", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {}, DebugInfo("test.cpp", 10, 2, 20, 5));
    auto sdfg = builder.move();

    auto conditional_schedule = std::make_unique<ConditionalSchedule>(sdfg);

    EXPECT_EQ(conditional_schedule->name(), "sdfg_1");
    EXPECT_EQ(conditional_schedule->size(), 1);
    EXPECT_EQ(conditional_schedule->debug_info().filename(), "test.cpp");
    EXPECT_EQ(conditional_schedule->debug_info().start_line(), 10);
    EXPECT_EQ(conditional_schedule->debug_info().start_column(), 2);
    EXPECT_EQ(conditional_schedule->debug_info().end_line(), 20);
    EXPECT_EQ(conditional_schedule->debug_info().end_column(), 5);

    // Default condition
    auto condition = conditional_schedule->condition(0);
    EXPECT_TRUE(symbolic::eq(condition, symbolic::__true__()));
}

TEST(ConditionalScheduleTest, PushFront) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType::CPU);

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("N", desc, true);
    builder.add_container("i", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto sdfg = builder.move();
    auto sdfg_ = sdfg->clone();

    auto conditional_schedule = std::make_unique<ConditionalSchedule>(sdfg);

    auto sym = symbolic::symbol("N");

    symbolic::Assumptions assumptions;
    assumptions.insert({sym, symbolic::Assumption(sym)});
    assumptions.at(sym).lower_bound(symbolic::integer(0));
    assumptions.at(sym).upper_bound(symbolic::integer(10));
    auto schedule_ = std::make_unique<Schedule>(sdfg_, assumptions);
    conditional_schedule->push_front(schedule_);

    auto& sched = conditional_schedule->schedule(0);
    EXPECT_TRUE(symbolic::eq(sched.assumptions().at(sym).lower_bound(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(sched.assumptions().at(sym).upper_bound(), symbolic::integer(10)));

    auto condition_ = conditional_schedule->condition(0);
    EXPECT_TRUE(symbolic::eq(condition_, symbolic::And(symbolic::Ge(sym, symbolic::integer(0)),
                                                       symbolic::Le(sym, symbolic::integer(10)))));
}

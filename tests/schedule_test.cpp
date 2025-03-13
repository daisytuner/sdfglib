#include "sdfg/schedule.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
using namespace sdfg;

TEST(ScheduleTest, NodeAllocations) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("N", desc, true);
    builder.add_container("M", desc, false, true);
    builder.add_container("i", desc);

    builder.add_structure("struct_1");
    types::Structure desc_struct("struct_1");
    builder.add_container("S", desc_struct);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(
        root, symbolic::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(sdfg);

    schedule->allocation_lifetime("i", &block);
    schedule->allocation_lifetime("S", &root);

    EXPECT_EQ(schedule->allocation_lifetime("i"), &block);
    EXPECT_EQ(schedule->allocation_lifetime("S"), &root);
}

TEST(ScheduleTest, Allocations) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("N", desc, true);
    builder.add_container("M", desc, false, true);
    builder.add_container("i", desc);

    builder.add_structure("struct_1");
    types::Structure desc_struct("struct_1");
    builder.add_container("S", desc_struct);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(
        root, symbolic::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto sdfg = builder.move();
    auto schedule = std::make_unique<Schedule>(sdfg);

    schedule->allocation_lifetime("i", &block);
    schedule->allocation_lifetime("S", &block);

    auto allocations = schedule->allocations(&block);
    EXPECT_EQ(allocations.size(), 2);
    EXPECT_EQ(allocations.count("i"), 1);
    EXPECT_EQ(allocations.count("S"), 1);

    auto root_allocations = schedule->allocations(&root);
    EXPECT_EQ(root_allocations.size(), 2);
    EXPECT_EQ(root_allocations.count("i"), 1);
    EXPECT_EQ(root_allocations.count("S"), 1);
}

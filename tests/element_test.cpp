#include <gtest/gtest.h>
#include "sdfg/debug_info.h"
using namespace sdfg;

TEST(DebugTableTest, Empty) {
    DebugInfoRegion debug_info;
    EXPECT_FALSE(debug_info.has());
    EXPECT_EQ(debug_info.filename(), "");
    EXPECT_EQ(debug_info.function(), "");
    EXPECT_EQ(debug_info.start_line(), 0);
    EXPECT_EQ(debug_info.start_column(), 0);
    EXPECT_EQ(debug_info.end_line(), 0);
    EXPECT_EQ(debug_info.end_column(), 0);
}

TEST(DebugTableTest, Basic) {
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element(debug_loc);
    DebugTable debug_info;

    debug_info.add_element(debug_info_element);

    DebugInfoRegion debug_info_region({0}, debug_info.elements());

    EXPECT_TRUE(debug_info_region.has());
    EXPECT_EQ(debug_info_region.filename(), "test.cpp");
    EXPECT_EQ(debug_info_region.function(), "test_function");
    EXPECT_EQ(debug_info_region.start_line(), 10);
    EXPECT_EQ(debug_info_region.start_column(), 2);
    EXPECT_EQ(debug_info_region.end_line(), 10);
    EXPECT_EQ(debug_info_region.end_column(), 2);
}

TEST(DebugTableTest, Merge_Left) {
    DebugTable debug_info;
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element(debug_loc);

    debug_info.add_element(debug_info_element);

    DebugInfoRegion debug_info_region_left({0}, debug_info.elements());
    DebugInfoRegion debug_info_region_right;

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

TEST(DebugTableTest, Merge_Right) {
    DebugTable debug_info;
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element(debug_loc);

    debug_info.add_element(debug_info_element);

    DebugInfoRegion debug_info_region_right({0}, debug_info.elements());
    DebugInfoRegion debug_info_region_left;

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

TEST(DebugTableTest, Merge_LeftRight) {
    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element_left(debug_loc_left);

    DebugLoc debug_loc_right("test.cpp", "test_function", 25, 5, true);
    DebugInfo debug_info_element_right(debug_loc_right);

    DebugTable debug_info;

    debug_info.add_element(debug_info_element_left);
    debug_info.add_element(debug_info_element_right);

    DebugInfoRegion debug_info_region_right({1}, debug_info.elements());
    DebugInfoRegion debug_info_region_left({0}, debug_info.elements());

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugTableTest, Merge_RightLeft) {
    DebugLoc debug_loc_right("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element_right(debug_loc_right);

    DebugLoc debug_loc_left("test.cpp", "test_function", 25, 5, true);
    DebugInfo debug_info_element_left(debug_loc_left);

    DebugTable debug_info;

    debug_info.add_element(debug_info_element_right);
    debug_info.add_element(debug_info_element_left);

    DebugInfoRegion debug_info_region_right({0}, debug_info.elements());
    DebugInfoRegion debug_info_region_left({1}, debug_info.elements());

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugTableTest, Merge_same_line) {
    DebugLoc debug_loc_right("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element_right(debug_loc_right);

    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 5, true);
    DebugInfo debug_info_element_left(debug_loc_left);

    DebugTable debug_info;

    debug_info.add_element(debug_info_element_right);
    debug_info.add_element(debug_info_element_left);

    DebugInfoRegion debug_info_region_right({0}, debug_info.elements());
    DebugInfoRegion debug_info_region_left({1}, debug_info.elements());

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugTableTest, Merge_multiple_locations) {
    DebugLoc debug_loc_right("test.cpp", "test_call", 200, 2, true);
    DebugLoc debug_loc_right2("test.cpp", "test_function", 8, 5, true);
    DebugInfo debug_info_element_right({debug_loc_right, debug_loc_right2});

    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfo debug_info_element_left(debug_loc_left);

    DebugTable debug_info;

    debug_info.add_element(debug_info_element_right);
    debug_info.add_element(debug_info_element_left);

    DebugInfoRegion debug_info_region_right({0}, debug_info.elements());
    DebugInfoRegion debug_info_region_left({1}, debug_info.elements());

    auto merged = DebugInfoRegion::merge(debug_info_region_left, debug_info_region_right, debug_info.elements());
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 8);
    EXPECT_EQ(merged.start_column(), 5);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

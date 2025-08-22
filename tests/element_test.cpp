#include "sdfg/element.h"

#include <gtest/gtest.h>
using namespace sdfg;

TEST(DebugInfoTest, Empty) {
    DebugInfo debug_info;
    EXPECT_FALSE(debug_info.has());
    EXPECT_EQ(debug_info.filename(), "");
    EXPECT_EQ(debug_info.function(), "");
    EXPECT_EQ(debug_info.start_line(), 0);
    EXPECT_EQ(debug_info.start_column(), 0);
    EXPECT_EQ(debug_info.end_line(), 0);
    EXPECT_EQ(debug_info.end_column(), 0);
}

TEST(DebugInfoTest, Basic) {
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element(debug_loc);
    DebugInfo debug_info(debug_info_element);

    EXPECT_TRUE(debug_info.has());
    EXPECT_EQ(debug_info.filename(), "test.cpp");
    EXPECT_EQ(debug_info.function(), "test_function");
    EXPECT_EQ(debug_info.start_line(), 10);
    EXPECT_EQ(debug_info.start_column(), 2);
    EXPECT_EQ(debug_info.end_line(), 10);
    EXPECT_EQ(debug_info.end_column(), 2);
}

TEST(DebugInfoTest, Merge_Left) {
    DebugInfo debug_info_left;
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element(debug_loc);
    DebugInfo debug_info_right(debug_info_element);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

TEST(DebugInfoTest, Merge_Right) {
    DebugInfo debug_info_right;
    DebugLoc debug_loc("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element(debug_loc);
    DebugInfo debug_info_left(debug_info_element);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

TEST(DebugInfoTest, Merge_LeftRight) {
    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element_left(debug_loc_left);
    DebugInfo debug_info_left(debug_info_element_left);

    DebugLoc debug_loc_right("test.cpp", "test_function", 25, 5, true);
    DebugInfoElement debug_info_element_right(debug_loc_right);
    DebugInfo debug_info_right(debug_info_element_right);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugInfoTest, Merge_RightLeft) {
    DebugLoc debug_loc_right("test.cpp", "test_function", 25, 5, true);
    DebugInfoElement debug_info_element_right(debug_loc_right);
    DebugInfo debug_info_right(debug_info_element_right);

    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element_left(debug_loc_left);
    DebugInfo debug_info_left(debug_info_element_left);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugInfoTest, Merge_same_line) {
    DebugLoc debug_loc_right("test.cpp", "test_function", 10, 5, true);
    DebugInfoElement debug_info_element_right(debug_loc_right);
    DebugInfo debug_info_right(debug_info_element_right);

    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element_left(debug_loc_left);
    DebugInfo debug_info_left(debug_info_element_left);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugInfoTest, Merge_multiple_locations) {
    DebugLoc debug_loc_right("test.cpp", "test_call", 200, 2, true);
    DebugLoc debug_loc_right2("test.cpp", "test_function", 8, 5, true);
    DebugInfoElement debug_info_element_right({debug_loc_right, debug_loc_right2});
    DebugInfo debug_info_right(debug_info_element_right);

    DebugLoc debug_loc_left("test.cpp", "test_function", 10, 2, true);
    DebugInfoElement debug_info_element_left(debug_loc_left);
    DebugInfo debug_info_left(debug_info_element_left);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.function(), "test_function");
    EXPECT_EQ(merged.start_line(), 8);
    EXPECT_EQ(merged.start_column(), 5);
    EXPECT_EQ(merged.end_line(), 10);
    EXPECT_EQ(merged.end_column(), 2);
}

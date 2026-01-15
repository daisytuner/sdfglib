#include "sdfg/element.h"

#include <gtest/gtest.h>
using namespace sdfg;

TEST(DebugInfoTest, Empty) {
    DebugInfo debug_info;
    EXPECT_FALSE(debug_info.has());
    EXPECT_EQ(debug_info.filename(), "");
    EXPECT_EQ(debug_info.start_line(), 0);
    EXPECT_EQ(debug_info.start_column(), 0);
    EXPECT_EQ(debug_info.end_line(), 0);
    EXPECT_EQ(debug_info.end_column(), 0);
}

TEST(DebugInfoTest, Basic) {
    DebugInfo debug_info("test.cpp", 10, 2, 20, 4);

    EXPECT_TRUE(debug_info.has());
    EXPECT_EQ(debug_info.filename(), "test.cpp");
    EXPECT_EQ(debug_info.start_line(), 10);
    EXPECT_EQ(debug_info.start_column(), 2);
    EXPECT_EQ(debug_info.end_line(), 20);
    EXPECT_EQ(debug_info.end_column(), 4);
}

TEST(DebugInfoTest, Merge_Left) {
    DebugInfo debug_info_left;
    DebugInfo debug_info_right("test.cpp", 10, 2, 20, 4);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 20);
    EXPECT_EQ(merged.end_column(), 4);
}

TEST(DebugInfoTest, Merge_Right) {
    DebugInfo debug_info_right;
    DebugInfo debug_info_left("test.cpp", 10, 2, 20, 4);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 20);
    EXPECT_EQ(merged.end_column(), 4);
}

TEST(DebugInfoTest, Merge_LeftRight) {
    DebugInfo debug_info_left("test.cpp", 10, 2, 20, 4);
    DebugInfo debug_info_right("test.cpp", 15, 3, 25, 5);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

TEST(DebugInfoTest, Merge_RightLeft) {
    DebugInfo debug_info_right("test.cpp", 10, 2, 20, 4);
    DebugInfo debug_info_left("test.cpp", 15, 3, 25, 5);

    auto merged = DebugInfo::merge(debug_info_left, debug_info_right);
    EXPECT_TRUE(merged.has());
    EXPECT_EQ(merged.filename(), "test.cpp");
    EXPECT_EQ(merged.start_line(), 10);
    EXPECT_EQ(merged.start_column(), 2);
    EXPECT_EQ(merged.end_line(), 25);
    EXPECT_EQ(merged.end_column(), 5);
}

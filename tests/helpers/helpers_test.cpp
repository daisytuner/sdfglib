#include "sdfg/helpers/helpers.h"

#include <gtest/gtest.h>

#include <unordered_set>

using namespace sdfg;

TEST(HelpersTest, Endswith) {
    std::string s = "Hello, World!";
    std::string ending = "World!";

    EXPECT_TRUE(helpers::endswith(s, ending));

    std::string s2 = "Hello, World!";
    std::string ending2 = "Hello";

    EXPECT_FALSE(helpers::endswith(s2, ending2));
}

TEST(HelpersTest, Join) {
    std::vector<std::string> v = {"1", "2", "3", "4", "5"};
    std::string delim = ", ";
    std::string result = helpers::join(v, delim);

    EXPECT_EQ(result, "1, 2, 3, 4, 5");

    std::vector<int> v2 = {};
    std::string delim2 = ", ";
    std::string result2 = helpers::join(v2, delim2);

    EXPECT_EQ(result2, "");
}

TEST(HelpersTest, SetsIntersect) {
    std::unordered_set<int> s1 = {1, 2, 3, 4, 5};
    std::unordered_set<int> s2 = {4, 5, 6, 7, 8};
    std::unordered_set<int> s3 = {6, 7, 8, 9, 10};

    EXPECT_TRUE(helpers::sets_intersect(s1, s2));
    EXPECT_FALSE(helpers::sets_intersect(s1, s3));
}

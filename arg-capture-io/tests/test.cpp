#include <gtest/gtest.h>

#include "test.h"
#include <filesystem>

static std::filesystem::path outputs_base_path;

const std::filesystem::path& get_outputs_base_path() {
    return outputs_base_path;
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    auto exe_path = std::filesystem::current_path() / std::filesystem::path(argv[0]);

    outputs_base_path = exe_path.parent_path() / "test_outputs";

    return RUN_ALL_TESTS();
}

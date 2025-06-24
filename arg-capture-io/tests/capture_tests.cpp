#include <gtest/gtest.h>

#include "arg_capture_io.h"
#include "primitive_types.h"
#include "test.h"
#include <filesystem>
#include <fstream>

using namespace arg_capture;

TEST(CaptureTests, PrimitiveTypesStrings) {
    auto str = to_string(arg_capture::PrimitiveType::Float);
    EXPECT_STREQ(str, "Float");
}

TEST(CaptureTests, EmptyToStart) {
    ArgCaptureIO capture("some_function");

    EXPECT_TRUE(capture.get_captures().empty());
}

TEST(CaptureTests, Clear) {
    ArgCaptureIO capture("some_function");

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_TRUE(capture.create_and_capture_inline(3, true, static_cast<int>(PrimitiveType::Int64), {8}, data));
    EXPECT_FALSE(capture.get_captures().empty());
    capture.clear();
    EXPECT_TRUE(capture.get_captures().empty());
}

TEST(CaptureTests, CaptureInline) {
    ArgCaptureIO capture("some_function");

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_TRUE(capture.create_and_capture_inline(3, true, static_cast<int>(PrimitiveType::Int64), {8}, data));

    data[0] = 9;
    
    auto& cap = capture.get_captures().at(std::make_pair(3, true));

    EXPECT_EQ(cap.arg_idx, 3);
    EXPECT_TRUE(cap.after);
    EXPECT_EQ(static_cast<PrimitiveType>(cap.primitive_type), PrimitiveType::Int64);
    EXPECT_EQ(cap.dims.size(), 1);
    EXPECT_EQ(cap.dims[0], 8);
    EXPECT_FALSE(cap.ext_file);
    EXPECT_TRUE(cap.data);
    EXPECT_EQ(cap.data->size(), 8);
    uint8_t ref_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(memcmp(cap.data->data(), ref_data, 8), 0);
}

TEST(CaptureTests, Capture3D) {
    ArgCaptureIO capture("some_function");

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    EXPECT_TRUE(capture.create_and_capture_inline(3, true, static_cast<int>(PrimitiveType::UInt8), {1, 2, 2, 3}, data));

    auto& cap = capture.get_captures().at(std::make_pair(3, true));

    EXPECT_EQ(cap.data->size(), 12);
}

TEST(CaptureTests, CaptureToFile) {
    ArgCaptureIO capture("some_function");    

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    auto& base = get_outputs_base_path();

    std::filesystem::path file = base / "CaptureTests" / "test_capture.bin";

    EXPECT_TRUE(capture.create_and_capture_to_file(3, true, static_cast<int>(PrimitiveType::Int64), {8}, file, data));

    auto& cap = capture.get_captures().at(std::make_pair(3, true));

    EXPECT_EQ(cap.arg_idx, 3);
    EXPECT_TRUE(cap.after);
    EXPECT_EQ(static_cast<PrimitiveType>(cap.primitive_type), PrimitiveType::Int64);
    EXPECT_EQ(cap.dims.size(), 1);
    EXPECT_EQ(cap.dims[0], 8);
    EXPECT_FALSE(cap.data);
    EXPECT_TRUE(cap.ext_file);
    EXPECT_EQ(cap.ext_file->string(), file.string());

    EXPECT_TRUE(std::filesystem::exists(file));
    std::ifstream ifs(file, std::ifstream::binary);

    uint8_t read_data[8] = {0};
    ifs.read(reinterpret_cast<char*>(read_data), 8);
    ifs.close();
    EXPECT_EQ(memcmp(read_data, data, 8), 0);
}
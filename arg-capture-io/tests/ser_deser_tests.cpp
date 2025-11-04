#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include "daisy_rtl/arg_capture_io.h"
#include "daisy_rtl/primitive_types.h"
#include "test.h"

using namespace arg_capture;

TEST(SerDeserTests, SerializeAndDeserializeCaptures) {
    ArgCaptureIO capture("some_function");
    capture.invocation("123");
    capture.invocation("123");

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_TRUE(capture.create_and_capture_inline(3, false, static_cast<int>(PrimitiveType::Int64), {8}, data, "123"));
    data[0] = 9;
    EXPECT_TRUE(capture.create_and_capture_inline(3, true, static_cast<int>(PrimitiveType::Double), {8}, data, "123"));
    data[1] = 10;
    auto binFile = get_outputs_base_path() / "SerDeser" / "arg4_in.bin";
    EXPECT_TRUE(capture.create_and_capture_to_file(
        4, false, static_cast<int>(PrimitiveType::Int64), {1, 2, 4}, binFile, data, "123"
    ));

    auto indexFile = get_outputs_base_path() / "SerDeser" / "index.json";
    capture.write_index(indexFile);

    auto deser = ArgCaptureIO::from_index(indexFile);

    EXPECT_EQ(deser->get_name(), capture.get_name());
    EXPECT_EQ(deser->get_current_invocation("123"), capture.get_current_invocation("123"));
    EXPECT_EQ(deser->get_captures().at("123").size(), 3);

    auto& elem1 = deser->get_captures().at("123").at(std::make_pair(3, false));
    EXPECT_EQ(elem1.arg_idx, 3);
    EXPECT_EQ(elem1.after, false);
    EXPECT_EQ(elem1.primitive_type, static_cast<int>(PrimitiveType::Int64));
    EXPECT_EQ(elem1.dims.size(), 1);
    EXPECT_EQ(elem1.dims[0], 8);
    EXPECT_EQ(elem1.data->size(), 8);
    uint8_t ref_dat1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(memcmp(elem1.data->data(), ref_dat1, 8), 0);

    auto& elem2 = deser->get_captures().at("123").at(std::make_pair(3, true));
    EXPECT_EQ(elem2.arg_idx, 3);
    EXPECT_EQ(elem2.after, true);
    EXPECT_EQ(elem2.primitive_type, static_cast<int>(PrimitiveType::Double));
    EXPECT_EQ(elem2.dims.size(), 1);
    EXPECT_EQ(elem2.dims[0], 8);
    EXPECT_EQ(elem2.data->size(), 8);
    uint8_t ref_dat2[] = {9, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(memcmp(elem2.data->data(), ref_dat2, 8), 0);

    auto& elem3 = deser->get_captures().at("123").at(std::make_pair(4, false));
    EXPECT_EQ(elem3.arg_idx, 4);
    EXPECT_EQ(elem3.after, false);
    EXPECT_EQ(elem3.primitive_type, static_cast<int>(PrimitiveType::Int64));
    EXPECT_EQ(elem3.dims.size(), 3);
    EXPECT_EQ(elem3.dims[0], 1);
    EXPECT_EQ(elem3.dims[1], 2);
    EXPECT_EQ(elem3.dims[2], 4);
    EXPECT_FALSE(elem3.data);
    std::ifstream file(*elem3.ext_file, std::ios::binary);
    uint8_t file_contents[8];
    file.read(reinterpret_cast<char*>(file_contents), 8);
    char extra;
    file.read(&extra, 1);
    EXPECT_TRUE(file.eof());
    file.close();
    uint8_t ref_dat3[] = {9, 10, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(memcmp(file_contents, ref_dat3, 8), 0);
}

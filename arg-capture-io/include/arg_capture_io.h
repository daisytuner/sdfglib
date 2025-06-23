#pragma once

#include <sys/types.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <unordered_map>
#include <vector>

namespace arg_capture {

struct ArgCapture {
    int32_t arg_idx;
    bool after;
    /**
     * innermost dimension is always the element size
     */
    const std::vector<size_t> dims {0};
    int primitive_type;
    std::shared_ptr<const std::filesystem::path> ext_file;
    std::shared_ptr<const std::vector<uint8_t>> data;


    ArgCapture() = default;

    ArgCapture(
        int32_t idx,
        bool after,
        int primitive_type,
        const std::vector<size_t> dims
    ) :
        arg_idx(idx),
        after(after),
        primitive_type(primitive_type),
        dims(dims)
    {}

    ArgCapture(const ArgCapture& other)
        : arg_idx(other.arg_idx),
          after(other.after),
          dims(other.dims),
          primitive_type(other.primitive_type),
          ext_file(other.ext_file),
          data(other.data)
    {}

    ArgCapture(const ArgCapture&& other) noexcept
        : arg_idx(other.arg_idx),
          after(other.after),
          dims(std::move(other.dims)),
          primitive_type(other.primitive_type),
          ext_file(std::move(other.ext_file)),
          data(std::move(other.data))
    {}

    void serialize_into(nlohmann::json& j) const;
};

struct MyHash {
    std::size_t operator()(const std::pair<int32_t, bool>& p) const {
        return std::hash<int32_t>()(p.first) ^ p.second? 0x40000000 : 0;
    }
};


class ArgCaptureIO {
   protected:
    std::string name_;
    uint32_t invokes_ = -1;
    std::filesystem::path output_dir_;
    std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash> current_captures_;

   public:
    ArgCaptureIO(const char* name, std::filesystem::path base_dir = ".") : name_(name), output_dir_(base_dir) {}
    void clear();

    bool create_and_capture_inline(int arg_idx, bool after, int primitive_type, const std::vector<size_t>& dims, const void* data);
    bool create_and_capture_to_file(int arg_idx, bool after, int primitive_type, const std::vector<size_t>& dims, std::filesystem::path& file, const void* data);

    bool capture_inline(ArgCapture& capture, const void* data);
    bool write_capture_to_file(ArgCapture& capture, std::filesystem::path file, const void* data);

    void write_index(std::filesystem::path file);

};

}

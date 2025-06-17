
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "daisy_rtl.h"
#include "primitive_types.h"

#ifndef DEBUG_LOG
#define DEBUG_LOG false
#endif

struct ArgCapture {
    int32_t arg_idx;
    bool after;
    const std::vector<size_t> dims {0};
    int primitive_type;
    std::shared_ptr<const std::filesystem::path> ext_file;
    std::shared_ptr<const uint8_t[]> data;


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
};

struct MyHash {
    std::size_t operator()(const std::pair<int32_t, bool>& p) const {
        return std::hash<int32_t>()(p.first) ^ p.second? 0x40000000 : 0;
    }
};

class DaisyCapture {
   private:
    std::string name_;
    int32_t invokes_ = -1;
    bool debug_ = true;
    std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash> current_captures_;

   public:
    explicit DaisyCapture(const char* name): name_(name) {}
    bool enter();
    void capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after);
    void capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after);
    void capture_2d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after);
    void exit();

    std::filesystem::path generate_output_path(int arg_idx, bool after) const;
};


bool DaisyCapture::enter() {
    ++invokes_;

    if (debug_) {
        std::cout << "Invoking '" << name_ << "' (" << invokes_ << ")" << std::endl;
    }

    return true;
}

std::filesystem::path DaisyCapture::generate_output_path(int arg_idx, bool after) const {
    std::string capType = after? "out" : "in";
    return "arg_capture/" + name_ + "_inv" + std::to_string(invokes_) +  "_arg" + std::to_string(arg_idx) + "_" + capType + ".bin";
}

void DaisyCapture::capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after) {

    if (DEBUG_LOG) {
        auto capType = after? "result" : "input";

        std::cout << "Capturing scalar arg" << arg_idx << " as " << capType << ": type "
            << primitive_type_names[primitive_type] << ": 0x" << std::hex;
    

        int perGroup = 0;
        for (int i = 0; i < size; ++i) {
            if (perGroup == 4) {
                perGroup = 0;
                std::cout << " ";
            } else {
                perGroup += 1;
            }

            const uint8_t* ptr = static_cast<const uint8_t*>(data) + i;
            uint8_t byte = *ptr;
            std::cout << byte;
        }

        std::cout << std::dec << std::endl;
    }

    auto key = std::make_pair(arg_idx, after);

    current_captures_.emplace(
        key,
        ArgCapture(arg_idx, after, primitive_type, {size})
    );

    auto capturedData = std::make_shared<uint8_t[]>(size);
    std::memcpy(capturedData.get(), data, size);

    current_captures_[key].data = std::move(capturedData);
}

void DaisyCapture::capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after) {
    auto capType = after? "result" : "input";

    if (DEBUG_LOG) {
        std::cout << "Capturing 1D arg" << arg_idx << " as " << capType << ": type "
            << primitive_type_names[primitive_type] << ": 0x" << std::hex << data << std::dec << " "
            << num_elements << "*" << size << " bytes" << std::endl;
    }

    auto key = std::make_pair(arg_idx, after);

    auto dims = std::vector<size_t>{size, num_elements};

    current_captures_.emplace(
        std::make_pair(key, ArgCapture(arg_idx, after, primitive_type, dims))
    );

    auto path = std::make_shared<std::filesystem::path>(generate_output_path(arg_idx, after));

    std::ofstream ofs(*path, std::ofstream::binary | std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for dumping arg" + std::to_string(arg_idx) + ": " + path->string());
    }
    
    ofs.write(reinterpret_cast<const char*>(data), size * num_elements);

    ofs.close();

    current_captures_[key].ext_file = std::move(path);
}


void DaisyCapture::capture_2d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after) {
    auto capType = after? "result" : "input";

    if (DEBUG_LOG) {
        std::cout << "Capturing 2D arg" << arg_idx << " as " << capType << ": type "
            << primitive_type_names[primitive_type] << ": 0x" << std::hex << data << std::dec << " "
            << num_rows << "*" << num_cols << "*" << size << " bytes" << std::endl;
    }

    auto key = std::make_pair(arg_idx, after);

    current_captures_.emplace(
        key,
        ArgCapture(arg_idx, after, primitive_type, {size, num_rows, num_cols})
    );

    auto path = std::make_shared<std::filesystem::path>(generate_output_path(arg_idx, after));

    std::ofstream ofs(*path, std::ofstream::binary | std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for dumping arg" + std::to_string(arg_idx) + ": " + path->string());
    }
    
    ofs.write(reinterpret_cast<const char*>(data), size * num_rows * num_cols);

    ofs.close();

    current_captures_[key].ext_file = std::move(path);
}

#ifdef __cplusplus
extern "C" {
#endif

struct __daisy_capture* __daisy_capture_init(const char* name) {
    DaisyCapture* ctx = new DaisyCapture(name);
    return (__daisy_capture_t*)ctx;
}

bool __daisy_capture_enter(__daisy_capture_t* context) {
    if (context) {
        return ((DaisyCapture*)context)->enter();
    } else {
        return false;
    }
}

void __daisy_capture_raw(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type, bool after) {
    if (context) {
        ((DaisyCapture*)context)->capture_raw(arg_idx, data, size, primitive_type, after);
    }
}

void __daisy_capture_1d(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type,
                            size_t num_elements, bool after) {
    if (context) {
        ((DaisyCapture*)context)->capture_1d(arg_idx, data, size, primitive_type, num_elements, after);
    }
}

void __daisy_capture_2d(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type,
                            size_t num_rows, size_t num_cols, bool after) {
    if (context) {
        ((DaisyCapture*)context)->capture_2d(arg_idx, data, size, primitive_type, num_rows, num_cols, after);
    }
}

void __daisy_capture_exit(__daisy_capture_t* context) {
    if (context) {
        ((DaisyCapture*)context)->exit();
    }
}


#ifdef __cplusplus
}
#endif



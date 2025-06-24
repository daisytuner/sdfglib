#include "arg_capture_io.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <utility>

#include "primitive_types.h"
#include "base64.h"

#ifndef DEBUG_LOG
#define DEBUG_LOG false
#endif

namespace arg_capture {

void ArgCapture::serialize_into(nlohmann::json& j) const {
    nlohmann::json entry;

    entry["arg_idx"] = arg_idx;
    entry["after"] = after;
    entry["primitive_type"] = primitive_type;

    auto jDims = nlohmann::json::array();
    for (const auto& dim : dims) {
        jDims.push_back(dim);
    }
    entry["dims"] = jDims;

    if (data) {
        std::string base64_data = base64_encode(data.get()->data(), data.get()->size());
        entry["data"] = base64_data;
    } else if (ext_file) {
        entry["ext_file"] = ext_file->string();
    }

    j.push_back(entry);
}

void ArgCapture::parse_from(const nlohmann::json& entry, std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>& map) {
    ArgCapture capture(
        entry["arg_idx"].get<int>(),
        entry["after"].get<bool>(),
        entry["primitive_type"].get<int>(),
        entry["dims"].get<std::vector<size_t>>()
    );

    if (entry.contains("data")) {
        auto data = base64_decode(entry["data"].get<std::string>());

        capture.data = std::make_shared<std::vector<uint8_t>>(std::move(data));
    } else if (entry.contains("ext_file")) {
        capture.ext_file = std::make_shared<std::filesystem::path>(entry["ext_file"].get<std::string>());
    }

    map.emplace(std::make_pair(capture.arg_idx, capture.after), std::move(capture));
}

const std::string& ArgCaptureIO::get_name() const {
    return name_;
}

uint32_t ArgCaptureIO::get_current_invocation() const {
    return invokes_;
}

void ArgCaptureIO::invocation() {
    ++invokes_;

    if (DEBUG_LOG) {
        std::cout << "Invoking '" << name_ << "' (" << invokes_ << ")" << std::endl;
    }
}

void ArgCaptureIO::clear() {
    current_captures_.clear();
}

const std::unordered_map<std::pair<int32_t, bool>, ArgCapture, MyHash>& ArgCaptureIO::get_captures() const {
    return current_captures_;
}

bool ArgCaptureIO::create_and_capture_inline(int arg_idx, bool after, int primitive_type, const std::vector<size_t>& dims, const void* data) {
    auto key = std::make_pair(arg_idx, after);

    auto it = current_captures_.emplace(
        key,
        ArgCapture(arg_idx, after, primitive_type, dims)
    );

    return capture_inline(it.first->second, data);
}

bool ArgCaptureIO::create_and_capture_to_file(int arg_idx, bool after, int primitive_type, const std::vector<size_t>& dims, std::filesystem::path& file, const void* data) {
    auto key = std::make_pair(arg_idx, after);

    auto it = current_captures_.emplace(
        key,
        ArgCapture(arg_idx, after, primitive_type, dims)
    );

    return write_capture_to_file(it.first->second, file, data);
}

bool ArgCaptureIO::capture_inline(ArgCapture& capture, const void* data) {
    auto size = std::accumulate(capture.dims.begin(), capture.dims.end(), 1, std::multiplies<size_t>());

    if (DEBUG_LOG) {
        auto capType = capture.after? "result" : "input";

        std::cout << "Capturing " << (capture.dims.size()-1) << "D arg" << capture.arg_idx << " as " << capType << ": type "
            << primitive_type_names[capture.primitive_type] << "(" << size << " bytes): 0x" << std::hex;

        int perGroup = 0;
        for (int i = 0; i < size; ++i) {
            const uint8_t* ptr = static_cast<const uint8_t*>(data) + i;
            uint32_t byte = *ptr;
            std::cout << byte;

            ++perGroup;
            if (perGroup == 16) {
                perGroup = 0;
                std::cout << "\n\t";
            } else if (perGroup % 4 == 0) {
                std::cout << " ";
            }
        }

        std::cout << std::dec << std::endl;
    }

    auto capturedData = std::make_shared<std::vector<uint8_t>>(size);
    std::memcpy(capturedData.get()->data(), data, size);

    current_captures_[std::make_pair(capture.arg_idx, capture.after)].data = std::move(capturedData);

    return true;
}

bool ArgCaptureIO::write_capture_to_file(ArgCapture& capture, std::filesystem::path file, const void* data) {

    if (DEBUG_LOG) {
        auto capType = capture.after? "result" : "input";

        std::cout << "Capturing " << (capture.dims.size()-1) << "D arg" << capture.arg_idx << " as " << capType << ": type "
            << primitive_type_names[capture.primitive_type] << ": 0x" << std::hex << data << std::dec << ", ";
        for (int i=0; i < capture.dims.size(); ++i) {
            auto dim = capture.dims[i];
            std::cout << dim;

            if (i < capture.dims.size() - 1) {
                std::cout << " * ";
            }
        }
        std::cout << " bytes" << std::endl;
    }

    std::filesystem::create_directories(file.parent_path());
    

    std::ofstream ofs(file, std::ofstream::binary | std::ofstream::out);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for dumping arg" + std::to_string(capture.arg_idx) + ": " + file.string());
    }

    auto totalSize = std::accumulate(capture.dims.begin(), capture.dims.end(), 1, std::multiplies<size_t>());
    
    ofs.write(reinterpret_cast<const char*>(data), totalSize);

    ofs.close();

    capture.ext_file = std::make_shared<std::filesystem::path>(file);

    return !ofs.bad();
}

void ArgCaptureIO::write_index(std::filesystem::path file) {    

    std::ofstream ofs(file);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open index file for writing: " + file.string());
    }

    nlohmann::json j;
    j["format"] = INDEX_FORMAT_VERSION;
    j["target"] = name_;
    j["invocation"] = invokes_;

    auto arr = nlohmann::json::array();

    for (const auto& [key, capture] : current_captures_) {

        capture.serialize_into(arr);
    }

    j["captures"] = arr;

    ofs << j;

    ofs.close();

    if (DEBUG_LOG) {
        std::cout << "Wrote capture index to " << file.string() << std::endl;
    }
}


}
#include "arg_capture_io.h"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <utility>

#include "primitive_types.h"

#ifndef DEBUG_LOG
#define DEBUG_LOG true
#endif

namespace arg_capture {

// copilot generated
std::string base64_encode(const uint8_t* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t octet_a = i < len ? data[i] : 0;
        uint32_t octet_b = (i + 1) < len ? data[i + 1] : 0;
        uint32_t octet_c = (i + 2) < len ? data[i + 2] : 0;

        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(i + 1 < len ? table[(triple >> 6) & 0x3F] : '=');
        encoded.push_back(i + 2 < len ? table[triple & 0x3F] : '=');
    }
    return encoded;
}

// copilot generated
std::vector<uint8_t> base64_decode(const std::string& input) {
    static const uint8_t table[] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64, 0,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64
    };
    std::vector<uint8_t> decoded;
    size_t len = input.size();
    int val = 0, valb = -8;
    for (size_t i = 0; i < len; ++i) {
        uint8_t c = input[i];
        if (c > 127) break;
        uint8_t d = table[c];
        if (d == 64) continue;
        val = (val << 6) | d;
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}

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

void ArgCaptureIO::clear() {
    current_captures_.clear();
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

        std::cout << "Capturing arg" << capture.arg_idx << " as " << capType << ": type "
            << primitive_type_names[capture.primitive_type] << "(" << size << " bytes): 0x" << std::hex;

        int perGroup = 0;
        for (int i = 0; i < size; ++i) {
            if (perGroup == 4) {
                perGroup = 0;
                std::cout << " ";
            } else {
                perGroup += 1;
            }

            const uint8_t* ptr = static_cast<const uint8_t*>(data) + i;
            uint32_t byte = *ptr;
            std::cout << byte;
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
    j["format"] = 0x00000001;
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
#include "daisy_rtl/arg_capture_io.h"
#include "daisy_rtl/daisy_rtl.h"
#include "daisy_rtl/primitive_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>

#ifndef DEBUG_LOG
#define DEBUG_LOG false
#endif

using namespace arg_capture;

constexpr uint32_t ALL_INVOCATIONS = -1;

class DaisyRtlCapture : public ArgCaptureIO {
protected:
    uint32_t invocation_to_capture_ = ALL_INVOCATIONS;
    std::filesystem::path output_dir_;

public:
    explicit DaisyRtlCapture(
        const char* name,
        std::filesystem::path base_dir,
        uint32_t invocation_to_capture = ALL_INVOCATIONS
    )
        : ArgCaptureIO(name), output_dir_(std::move(base_dir)), invocation_to_capture_(invocation_to_capture) {}

    const std::filesystem::path& get_output_dir() const;

    bool enter();

    void capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after);
    void capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after);
    void capture_2d(
        int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after
    );
    void capture_3d(
        int arg_idx,
        const void* data,
        size_t size,
        int primitive_type,
        size_t num_x,
        size_t num_y,
        size_t num_z,
        bool after
    );

    void exit();

protected:
    bool write_capture_to_file(ArgCapture& capture, const void* data);
    std::filesystem::path generate_arg_capture_output_filename(int arg_idx, bool after) const;
};


const std::filesystem::path& DaisyRtlCapture::get_output_dir() const { return output_dir_; }

bool DaisyRtlCapture::enter() {
    clear();

    invocation();

    auto specific_invocation_only = this->invocation_to_capture_;
    return (specific_invocation_only == ALL_INVOCATIONS) || (invokes_ == specific_invocation_only);
}

void DaisyRtlCapture::capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after) {
    create_and_capture_inline(arg_idx, after, primitive_type, {size}, data);
}

void DaisyRtlCapture::
    capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after) {
    auto file = generate_arg_capture_output_filename(arg_idx, after);

    if (!create_and_capture_to_file(arg_idx, after, primitive_type, {size, num_elements}, file, data)) {
        throw std::runtime_error("Failed to write capture for arg" + std::to_string(arg_idx) + " to file");
    }
}


void DaisyRtlCapture::capture_2d(
    int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after
) {
    auto file = generate_arg_capture_output_filename(arg_idx, after);

    if (!create_and_capture_to_file(arg_idx, after, primitive_type, {size, num_rows, num_cols}, file, data)) {
        throw std::runtime_error("Failed to write capture for arg" + std::to_string(arg_idx) + " to file");
    }
}

void DaisyRtlCapture::capture_3d(
    int arg_idx, const void* data, size_t size, int primitive_type, size_t num_x, size_t num_y, size_t num_z, bool after
) {
    auto file = generate_arg_capture_output_filename(arg_idx, after);

    if (!create_and_capture_to_file(arg_idx, after, primitive_type, {size, num_x, num_y, num_z}, file, data)) {
        throw std::runtime_error("Failed to write capture for arg" + std::to_string(arg_idx) + " to file");
    }
}

bool DaisyRtlCapture::write_capture_to_file(arg_capture::ArgCapture& capture, const void* data) {
    auto file = generate_arg_capture_output_filename(capture.arg_idx, capture.after);
    return ArgCaptureIO::write_capture_to_file(capture, file, data);
}

void DaisyRtlCapture::exit() {
    if (DEBUG_LOG) {
        std::cout << "Finalizing capture of '" << name_ << std::endl;
    }
    if (!current_captures_.empty()) {
        auto path = output_dir_ / (name_ + "_inv" + std::to_string(invokes_) + ".index.json");
        write_index(path);
    }
}

std::filesystem::path DaisyRtlCapture::generate_arg_capture_output_filename(int arg_idx, bool after) const {
    std::string capType = after ? "out" : "in";
    return output_dir_ /
           (name_ + "_inv" + std::to_string(invokes_) + "_arg" + std::to_string(arg_idx) + "_" + capType + ".bin");
}


#ifdef __cplusplus
extern "C" {
#endif

struct __daisy_capture* __daisy_capture_init(const char* name, const char* base_dir) {
    auto* default_strat = getenv("__DAISY_CAPTURE_STRATEGY_DEFAULT");

    DaisyRtlCapture* ctx;

    std::filesystem::path base_dir_path = "arg_captures";
    if (base_dir) {
        base_dir_path = base_dir;
    }

    if (default_strat && std::strcmp(default_strat, "all") == 0) {
        ctx = new DaisyRtlCapture(name, base_dir_path, ALL_INVOCATIONS);
    } else if (default_strat && std::strcmp(default_strat, "never") == 0) {
        return nullptr;
    } else if (!default_strat || std::strcmp(default_strat, "once") == 0) {
        ctx = nullptr;
    } else {
        fprintf(stderr, "Unknown capture strategy: '%s' for ctx '%s'\n", default_strat, name);
        ctx = nullptr;
    }

    if (!ctx) { // default
        ctx = new DaisyRtlCapture(name, base_dir_path, 0);
    }

    return (__daisy_capture_t*) ctx;
}

bool __daisy_capture_enter(__daisy_capture_t* context) {
    if (context) {
        return ((DaisyRtlCapture*) context)->enter();
    } else {
        return false;
    }
}

void __daisy_capture_raw(
    __daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type, bool after
) {
    if (context) {
        ((DaisyRtlCapture*) context)->capture_raw(arg_idx, data, size, primitive_type, after);
    }
}

void __daisy_capture_1d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_elements,
    bool after
) {
    if (context) {
        ((DaisyRtlCapture*) context)->capture_1d(arg_idx, data, size, primitive_type, num_elements, after);
    }
}

void __daisy_capture_2d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_rows,
    size_t num_cols,
    bool after
) {
    if (context) {
        ((DaisyRtlCapture*) context)->capture_2d(arg_idx, data, size, primitive_type, num_rows, num_cols, after);
    }
}

void __daisy_capture_3d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_x,
    size_t num_y,
    size_t num_z,
    bool after
) {
    if (context) {
        ((DaisyRtlCapture*) context)->capture_3d(arg_idx, data, size, primitive_type, num_x, num_y, num_z, after);
    }
}

void __daisy_capture_end(__daisy_capture_t* context) {
    if (context) {
        ((DaisyRtlCapture*) context)->exit();
    }
}


#ifdef __cplusplus
}
#endif

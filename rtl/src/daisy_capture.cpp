
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <utility>

#include "daisy_rtl.h"
#include "primitive_types.h"

class DaisyCapture {
   private:
    std::string name;
    int32_t invokes = 0;
    bool debug = true;
   public:
    explicit DaisyCapture(const char* name): name(name) {}
    bool enter();
    void capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after);
    void capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after);
    void capture_2d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after);
};


bool DaisyCapture::enter() {
    ++invokes;

    if (debug) {
        std::cout << "Invoking '" << name << "' (" << invokes << ")" << std::endl;
    }

    return true;
}

void DaisyCapture::capture_raw(int arg_idx, const void* data, size_t size, int primitive_type, bool after) {
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

void DaisyCapture::capture_1d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_elements, bool after) {
    auto capType = after? "result" : "input";

    std::cout << "Capturing 1D arg" << arg_idx << " as " << capType << ": type "
        << primitive_type_names[primitive_type] << ": 0x" << std::hex << data << std::dec << " "
        << num_elements << "*" << size << " bytes" << std::endl;
}


void DaisyCapture::capture_2d(int arg_idx, const void* data, size_t size, int primitive_type, size_t num_rows, size_t num_cols, bool after) {
    auto capType = after? "result" : "input";

    std::cout << "Capturing 2D arg" << arg_idx << " as " << capType << ": type "
        << primitive_type_names[primitive_type] << ": 0x" << std::hex << data << std::dec << " "
        << num_rows << "*" << num_cols << "*" << size << " bytes" << std::endl;
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


#ifdef __cplusplus
}
#endif



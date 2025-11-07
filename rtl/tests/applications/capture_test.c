#include <stdint.h>
#include <stdio.h>

#include <daisy_rtl/daisy_rtl.h>

static void* __capture_ctx;
static void __attribute__((constructor(1000))) __capture_ctx_init(void) {
    __capture_ctx = __daisy_capture_init("__daisy_capture_test_function", NULL);
}


void __daisy_capture_test_function(const int32_t* arg_a, const int32_t* arg_b, int64_t arg_scalar, int32_t* arg_out) {
    const bool __daisy_cap_en = __daisy_capture_enter(__capture_ctx, 123);
    if (__daisy_cap_en) {
        __daisy_capture_1d(__capture_ctx, 0, arg_a, 4, 4, 10, false, 123);
        __daisy_capture_1d(__capture_ctx, 1, arg_b, 4, 4, 10, false, 123);
        __daisy_capture_raw(__capture_ctx, 2, &arg_scalar, 8, 5, false, 123);
    }

    for (int i = 0; i < 10; i++) {
        uint32_t mask_a = arg_scalar & 0xFFFFFFFFL;
        uint32_t mask_b = arg_scalar >> 32 & 0xFFFFFFFFL;
        arg_out[i] = (arg_a[i] & mask_a) + (arg_b[i] & mask_b);
    }

    if (__daisy_cap_en) {
        __daisy_capture_1d(__capture_ctx, 3, arg_out, 4, 4, 10, true, 123);
        __daisy_capture_end(__capture_ctx);
    }
}

void main(int argc, char** argv) {
    int32_t a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t b[10] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    int32_t out[10];
    int32_t out2[10];

    __daisy_capture_test_function(a, b, -1L, out);
    __daisy_capture_test_function(a, b, 0L, out2);

    for (int i = 0; i < 10; i++) {
        printf("%d: first: %d second: %d\n", i, out[i], out2[i]);
    }
}

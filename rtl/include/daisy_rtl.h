#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct __daisy_metadata {
    const char* file_name;
    const char* function_name;
    long line_begin;
    long line_end;
    long column_begin;
    long column_end;
    const char* region_name;
};

struct __daisy_instrumentation;

typedef struct __daisy_instrumentation __daisy_instrumentation_t;

__daisy_instrumentation_t* __daisy_instrumentation_init();
void __daisy_instrumentation_finalize(__daisy_instrumentation_t* context);
void __daisy_instrumentation_enter(__daisy_instrumentation_t* context, __daisy_metadata* metadata);
void __daisy_instrumentation_exit(__daisy_instrumentation_t* context, __daisy_metadata* metadata);

typedef struct __daisy_capture __daisy_capture_t;

__daisy_capture_t* __daisy_capture_init(const char* name);

bool __daisy_capture_enter(__daisy_capture_t* context);
void __daisy_capture_end(__daisy_capture_t* context);

void __daisy_capture_raw(
    __daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type, bool after
);

void __daisy_capture_1d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_elements,
    bool after
);

void __daisy_capture_2d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_rows,
    size_t num_cols,
    bool after
);

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
);

#ifdef __cplusplus
}
#endif

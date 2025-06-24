#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void __daisy_instrument_init();
void __daisy_instrument_finalize();
void __daisy_instrument_enter();

// Deprecated
void __daisy_instrument_exit(const char* region_name, const char* file_name,
                             const char* function_name, long line_begin, long line_end,
                             long column_begin, long column_end);

void __daisy_instrument_exit_with_metadata(const char* region_name, const char* dbg_file_name,
                                           const char* dbg_function_name, long dbg_line_begin,
                                           long dbg_line_end, long dbg_column_begin,
                                           long dbg_column_end, const char* source_file,
                                           const char* features_file);

typedef struct __daisy_capture __daisy_capture_t;

__daisy_capture_t* __daisy_capture_init(const char* name);

bool __daisy_capture_enter(__daisy_capture_t* context);
void __daisy_capture_end(__daisy_capture_t* context);

void __daisy_capture_raw(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type, bool after);

void __daisy_capture_1d(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type,
                            size_t num_elements, bool after);

void __daisy_capture_2d(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type,
                            size_t num_rows, size_t num_cols, bool after);

void __daisy_capture_3d(__daisy_capture_t* context, int arg_idx, const void* data, size_t size, int primitive_type,
                            size_t num_x, size_t num_y, size_t num_z, bool after);



#ifdef __cplusplus
}
#endif


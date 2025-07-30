#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum __daisy_event_set {
    __DAISY_EVENT_SET_CPU = 0,
    __DAISY_EVENT_SET_CUDA = 1,
};

typedef struct __daisy_metadata {
    const char* file_name;
    const char* function_name;
    long line_begin;
    long line_end;
    long column_begin;
    long column_end;
    const char* region_name;
} __daisy_metadata_t;

typedef struct __daisy_instrumentation __daisy_instrumentation_t;

__daisy_instrumentation_t* __daisy_instrumentation_init();
void __daisy_instrumentation_finalize(__daisy_instrumentation_t* context);
void __daisy_instrumentation_enter(
    __daisy_instrumentation_t* context, __daisy_metadata_t* metadata, enum __daisy_event_set event_set
);
void __daisy_instrumentation_exit(
    __daisy_instrumentation_t* context, __daisy_metadata_t* metadata, enum __daisy_event_set event_set
);

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

#define __daisy_min(a, b) ((a) < (b) ? (a) : (b))
#define __daisy_max(a, b) ((a) > (b) ? (a) : (b))
#define __daisy_fma(a, b, c) a* b + c

#ifdef __DAISY_NVVM__

// type conversion
#define __daisy_d2i_hi __nvvm_d2i_hi
#define __daisy_d2i_lo __nvvm_d2i_lo
#define __daisy_lohi_i2d __nvvm_lohi_i2d

#define __daisy_d2i_rn __nvvm_d2i_rn
#define __daisy_d2i_rm __nvvm_d2i_rm
#define __daisy_d2i_rp __nvvm_d2i_rp
#define __daisy_d2i_rz __nvvm_d2i_rz

#define __daisy_i2d_rn __nvvm_i2d_rn
#define __daisy_i2d_rm __nvvm_i2d_rm
#define __daisy_i2d_rp __nvvm_i2d_rp
#define __daisy_i2d_rz __nvvm_i2d_rz

#define __daisy_d2f_rn __nvvm_d2f_rn
#define __daisy_d2f_rm __nvvm_d2f_rm
#define __daisy_d2f_rp __nvvm_d2f_rp
#define __daisy_d2f_rz __nvvm_d2f_rz

#define __daisy_d2ui_rn __nvvm_d2ui_rn
#define __daisy_d2ui_rm __nvvm_d2ui_rm
#define __daisy_d2ui_rp __nvvm_d2ui_rp
#define __daisy_d2ui_rz __nvvm_d2ui_rz

#define __daisy_ui2d_rn __nvvm_ui2d_rn
#define __daisy_ui2d_rm __nvvm_ui2d_rm
#define __daisy_ui2d_rp __nvvm_ui2d_rp
#define __daisy_ui2d_rz __nvvm_ui2d_rz

#define __daisy_d2ll_rn __nvvm_d2ll_rn
#define __daisy_d2ll_rm __nvvm_d2ll_rm
#define __daisy_d2ll_rp __nvvm_d2ll_rp
#define __daisy_d2ll_rz __nvvm_d2ll_rz

#define __daisy_ll2d_rn __nvvm_ll2d_rn
#define __daisy_ll2d_rm __nvvm_ll2d_rm
#define __daisy_ll2d_rp __nvvm_ll2d_rp
#define __daisy_ll2d_rz __nvvm_ll2d_rz

#define __daisy_d2ull_rn __nvvm_d2ull_rn
#define __daisy_d2ull_rm __nvvm_d2ull_rm
#define __daisy_d2ull_rp __nvvm_d2ull_rp
#define __daisy_d2ull_rz __nvvm_d2ull_rz

#define __daisy_ull2d_rn __nvvm_ull2d_rn
#define __daisy_ull2d_rm __nvvm_ull2d_rm
#define __daisy_ull2d_rp __nvvm_ull2d_rp
#define __daisy_ull2d_rz __nvvm_ull2d_rz

#define __daisy_f2i_rn __nvvm_f2i_rn
#define __daisy_f2i_rm __nvvm_f2i_rm
#define __daisy_f2i_rp __nvvm_f2i_rp
#define __daisy_f2i_rz __nvvm_f2i_rz

#define __daisy_i2f_rn __nvvm_i2f_rn
#define __daisy_i2f_rm __nvvm_i2f_rm
#define __daisy_i2f_rp __nvvm_i2f_rp
#define __daisy_i2f_rz __nvvm_i2f_rz

#define __daisy_f2ui_rn __nvvm_f2ui_rn
#define __daisy_f2ui_rm __nvvm_f2ui_rm
#define __daisy_f2ui_rp __nvvm_f2ui_rp
#define __daisy_f2ui_rz __nvvm_f2ui_rz

#define __daisy_ui2f_rn __nvvm_ui2f_rn
#define __daisy_ui2f_rm __nvvm_ui2f_rm
#define __daisy_ui2f_rp __nvvm_ui2f_rp
#define __daisy_ui2f_rz __nvvm_ui2f_rz

#define __daisy_f2ll_rn __nvvm_f2ll_rn
#define __daisy_f2ll_rm __nvvm_f2ll_rm
#define __daisy_f2ll_rp __nvvm_f2ll_rp
#define __daisy_f2ll_rz __nvvm_f2ll_rz

#define __daisy_ll2f_rn __nvvm_ll2f_rn
#define __daisy_ll2f_rm __nvvm_ll2f_rm
#define __daisy_ll2f_rp __nvvm_ll2f_rp
#define __daisy_ll2f_rz __nvvm_ll2f_rz

#define __daisy_f2ull_rn __nvvm_f2ull_rn
#define __daisy_f2ull_rm __nvvm_f2ull_rm
#define __daisy_f2ull_rp __nvvm_f2ull_rp
#define __daisy_f2ull_rz __nvvm_f2ull_rz

#define __daisy_ull2f_rn __nvvm_ull2f_rn
#define __daisy_ull2f_rm __nvvm_ull2f_rm
#define __daisy_ull2f_rp __nvvm_ull2f_rp
#define __daisy_ull2f_rz __nvvm_ull2f_rz

#define __daisy_f2bf16_rn __nvvm_f2bf16_rn
#define __daisy_f2bf16_rz __nvvm_f2bf16_rz

#define __daisy_f2h_rn __nvvm_f2h_rn

// saturate
#define __daisy_saturate_f __nvvm_saturate_f
#define __daisy_saturate_d __nvvm_saturate_d

// fma instructions
#define __daisy_fma_rn_f __nvvm_fma_rn_f
#define __daisy_fma_rn_d __nvvm_fma_rn_d

#define __daisy_fma_rm_f __nvvm_fma_rm_f
#define __daisy_fma_rm_d __nvvm_fma_rm_d

#define __daisy_fma_rp_f __nvvm_fma_rp_f
#define __daisy_fma_rp_d __nvvm_fma_rp_d

#define __daisy_fma_rz_f __nvvm_fma_rz_f
#define __daisy_fma_rz_d __nvvm_fma_rz_d

#define __daisy_fma_rn_ftz_f __nvvm_fma_rn_ftz_f

#define __daisy_fma_rm_ftz_f __nvvm_fma_rm_ftz_f

#define __daisy_fma_rp_ftz_f __nvvm_fma_rp_ftz_f

#define __daisy_fma_rz_ftz_f __nvvm_fma_rz_ftz_f

#endif // __DAISY_NVVM__

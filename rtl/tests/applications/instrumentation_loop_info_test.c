#include <stdio.h>

#include <daisy_rtl/daisy_rtl.h>

void main(int argc, char** argv) {
    double A[128][128];
    double B[128][128];
    double C[128][128];

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0;
        }
    }

    __daisy_metadata_t metadata = {
        .file_name = "instrumentation_test.c",
        .function_name = "main",
        .line_begin = 18,
        .line_end = 31,
        .column_begin = 4,
        .column_end = 5,
        .sdfg_name = "__daisy_instrumentation_test_0",
        .sdfg_file = "/tmp/DOCC/0000-0000/123456789/sdfg_0.json",
        .element_id = 10,
        .element_type = "for",
        .target_type = "sequential",
        .loopnest_index = 0,
        .num_loops = 3,
        .num_maps = 2,
        .num_fors = 1,
        .num_whiles = 0,
        .max_depth = 3,
        .is_perfectly_nested = true,
        .is_perfectly_parallel = false,
        .is_elementwise = false,
        .has_side_effects = false,
        .region_uuid = "__daisy_instrumentation_test_0_10"
    };

    unsigned long long region_id = __daisy_instrumentation_init(&metadata, __DAISY_EVENT_SET_CPU);
    __daisy_instrumentation_enter(region_id);

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            for (int k = 0; k < 128; k++) {
                C[i][j] = A[i][k] * B[k][j] + C[i][j];
            }
        }
    }

    __daisy_instrumentation_exit(region_id);
    __daisy_instrumentation_finalize(region_id);

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            printf("%f\n", C[i][j]);
        }
    }
}

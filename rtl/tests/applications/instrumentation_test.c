#include <stdio.h>

#include <daisy_rtl/daisy_rtl.h>

void main(int argc, char** argv) {
    __daisy_metadata_t metadata = {
        .file_name = "instrumentation_test.c",
        .function_name = "main",
        .line_begin = 18,
        .line_end = 31,
        .column_begin = 4,
        .column_end = 5,
        .sdfg_name = "__daisy_instrumentation_test_0",
        .sdfg_file = "/tmp/DOCC/0000-0000/123456789/sdfg_0.json",
        .arg_capture_path = "",
        .features_file = "",
        .element_id = 10,
        .loopnest_index = 0,
        .region_uuid = "__daisy_instrumentation_test_0_10"
    };
    unsigned long long region_id = __daisy_instrumentation_init(&metadata, __DAISY_EVENT_SET_CPU);

    for (size_t rep = 0; rep < 10; rep++) {
        __daisy_instrumentation_enter(region_id);

        double A[1000];
        double B[1000];
        double C[1000];

        for (int i = 0; i < 1000; i++) {
            A[i] = i;
            B[i] = 1000 - i;
            C[i] = 0;
        }

        for (int i = 0; i < 1000; i++) {
            C[i] = A[i] + B[i];
        }

        __daisy_instrumentation_exit(region_id);

        for (int i = 0; i < 1000; i++) {
            printf("%f\n", C[i]);
        }
    }

    __daisy_instrumentation_finalize(region_id);
}

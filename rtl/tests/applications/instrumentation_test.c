#include <stdio.h>

#include <daisy_rtl/daisy_rtl.h>

void main(int argc, char** argv) {
    __daisy_instrumentation_t* context = __daisy_instrumentation_init();

    __daisy_metadata_t metadata = {
        .file_name = "instrumentation_test.c",
        .function_name = "main",
        .line_begin = 18,
        .line_end = 31,
        .column_begin = 4,
        .column_end = 5,
        .region_name = "instrumentation_test_main",
    };

    for (size_t rep = 0; rep < 10; rep++) {
        __daisy_instrumentation_enter(context, &metadata);

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

        __daisy_instrumentation_exit(context, &metadata);

        for (int i = 0; i < 1000; i++) {
            printf("%f\n", C[i]);
        }
    }

    __daisy_instrumentation_finalize(context);
}

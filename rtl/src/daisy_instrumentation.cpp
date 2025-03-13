#include "daisy_instrumentation.h"

#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>

#include <cstdlib>
#include <cstring>
#include <ctime>

Instrumentation_PAPI instrumentation;

void __daisy_instrument_init() { instrumentation = Instrumentation_PAPI(); }

void __daisy_instrument_finalize() {}

void __daisy_instrument_enter() { instrumentation.__daisy_instrument_enter(); }

void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                             long line_end, long column_begin, long column_end) {
    instrumentation.__daisy_instrument_exit(region_name, file_name, line_begin, line_end,
                                            column_begin, column_end);
}

Instrumentation_PAPI::Instrumentation_PAPI() {
    output_file = getenv("__DAISY_INSTRUMENTATION_FILE");
    if (output_file == NULL) {
        return;
    }

    int retval;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error initializing PAPI! %s\n", PAPI_strerror(retval));
        exit(1);
    }

    retval = PAPI_create_eventset(&eventset);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Error creating eventset! %s\n", PAPI_strerror(retval));
        exit(1);
    }

    runtime = false;
    event_name = getenv("__DAISY_INSTRUMENTATION_EVENTS");
    if (event_name == NULL) {
        return;
    } else if (strcmp(event_name, "DURATION_TIME") == 0) {
        runtime = true;
    } else {
        retval = PAPI_add_named_event(eventset, event_name);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error converting event name to code: %s\n", event_name);
            exit(1);
        }
    }
}

void Instrumentation_PAPI::__daisy_instrument_enter() {
    if (output_file == NULL || event_name == NULL) {
        return;
    }

    if (runtime) {
        runtime_start = PAPI_get_real_nsec();
        return;
    }

    int retval;
    PAPI_reset(eventset);
    retval = PAPI_start(eventset);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Error starting PAPI: %s\n", PAPI_strerror(retval));
    }
}

void Instrumentation_PAPI::__daisy_instrument_exit(const char* region_name, const char* file_name,
                                                   long line_begin, long line_end,
                                                   long column_begin, long column_end) {
    if (output_file == NULL || event_name == NULL) {
        return;
    }

    long long count;
    if (runtime) {
        count = PAPI_get_real_nsec() - runtime_start;
    } else {
        int retval;
        retval = PAPI_stop(eventset, &count);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error stopping PAPI:  %s\n", PAPI_strerror(retval));
            return;
        }
    }

    // concatentate the string to file and create file if it does not exist
    FILE* f = fopen(output_file, "a");
    if (f == NULL) {
        fprintf(stderr, "Error opening file %s\n", output_file);
        f = fopen(output_file, "w");
    }

    fprintf(f, "%s,%s,%ld,%ld,%ld,%ld,%s,%lld,%ld\n", region_name, file_name, line_begin, line_end,
            column_begin, column_end, event_name, count, std::time(nullptr));
}

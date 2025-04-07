#include "daisy_instrumentation.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>

Instrumentation_PAPI instrumentation;

extern "C" void __daisy_instrument_init() {
    instrumentation = Instrumentation_PAPI();
}

extern "C" void __daisy_instrument_finalize() {}

extern "C" void __daisy_instrument_enter() {
    instrumentation.__daisy_instrument_enter();
}

extern "C" void __daisy_instrument_exit(const char* region_name, const char* file_name,
                                        long line_begin, long line_end, long column_begin,
                                        long column_end) {
    instrumentation.__daisy_instrument_exit(region_name, file_name, line_begin, line_end,
                                            column_begin, column_end);
}

// Function pointers for PAPI functions
static int (*_PAPI_library_init)(int) = nullptr;
static int (*_PAPI_create_eventset)(int*) = nullptr;
static int (*_PAPI_add_named_event)(int, const char*) = nullptr;
static int (*_PAPI_start)(int) = nullptr;
static int (*_PAPI_stop)(int, long long*) = nullptr;
static int (*_PAPI_reset)(int) = nullptr;
static const char* (*_PAPI_strerror)(int) = nullptr;
static long long (*_PAPI_get_real_nsec)(void) = nullptr;

void Instrumentation_PAPI::load_papi_symbols() {
    const char* papi_lib = getenv("DAISY_PAPI_PATH");
    if (papi_lib == NULL) papi_lib = "libpapi-dev.so";

    void* handle = dlopen(papi_lib, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Could not load %s: %s\n", papi_lib, dlerror());
        return;
    }

    _PAPI_library_init = (int (*)(int))dlsym(handle, "PAPI_library_init");
    _PAPI_create_eventset = (int (*)(int*))dlsym(handle, "PAPI_create_eventset");
    _PAPI_add_named_event = (int (*)(int, const char*))dlsym(handle, "PAPI_add_named_event");
    _PAPI_start = (int (*)(int))dlsym(handle, "PAPI_start");
    _PAPI_stop = (int (*)(int, long long*))dlsym(handle, "PAPI_stop");
    _PAPI_reset = (int (*)(int))dlsym(handle, "PAPI_reset");
    _PAPI_strerror = (const char* (*)(int))dlsym(handle, "PAPI_strerror");
    _PAPI_get_real_nsec = (long long (*)(void))dlsym(handle, "PAPI_get_real_nsec");

    if (!_PAPI_library_init || !_PAPI_create_eventset || !_PAPI_add_named_event ||
        !_PAPI_start || !_PAPI_stop || !_PAPI_reset || !_PAPI_strerror || !_PAPI_get_real_nsec) {
        fprintf(stderr, "Failed to load one or more PAPI symbols. Please install papi.\n");
        dlclose(handle);
        exit(1);
    }
}

Instrumentation_PAPI::Instrumentation_PAPI() {
    output_file = getenv("__DAISY_INSTRUMENTATION_FILE");
    if (output_file == nullptr) {
        return;
    }

    load_papi_symbols();

    int retval = _PAPI_library_init(0x00000005);  // PAPI_VER_CURRENT
    if (retval != 0x00000005) {
        fprintf(stderr, "Error initializing PAPI! %s\n", _PAPI_strerror(retval));
        exit(1);
    }

    retval = _PAPI_create_eventset(&eventset);
    if (retval != 0) {
        fprintf(stderr, "Error creating eventset! %s\n", _PAPI_strerror(retval));
        exit(1);
    }

    event_name = getenv("__DAISY_INSTRUMENTATION_EVENTS");
    if (event_name == nullptr) return;

    if (strcmp(event_name, "DURATION_TIME") == 0) {
        runtime = true;
    } else {
        retval = _PAPI_add_named_event(eventset, event_name);
        if (retval != 0) {
            fprintf(stderr, "Error converting event name to code: %s\n", event_name);
            exit(1);
        }
    }
}

void Instrumentation_PAPI::__daisy_instrument_enter() {
    if (output_file == nullptr || event_name == nullptr) return;

    if (runtime) {
        runtime_start = _PAPI_get_real_nsec();
        return;
    }

    _PAPI_reset(eventset);
    int retval = _PAPI_start(eventset);
    if (retval != 0) {
        fprintf(stderr, "Error starting PAPI: %s\n", _PAPI_strerror(retval));
    }
}

void Instrumentation_PAPI::__daisy_instrument_exit(const char* region_name, const char* file_name,
                                                   long line_begin, long line_end,
                                                   long column_begin, long column_end) {
    if (output_file == nullptr || event_name == nullptr) return;

    long long count;
    if (runtime) {
        count = _PAPI_get_real_nsec() - runtime_start;
    } else {
        int retval = _PAPI_stop(eventset, &count);
        if (retval != 0) {
            fprintf(stderr, "Error stopping PAPI:  %s\n", _PAPI_strerror(retval));
            return;
        }
    }

    FILE* f = fopen(output_file, "a");
    if (f == nullptr) {
        fprintf(stderr, "Error opening file %s\n", output_file);
        f = fopen(output_file, "w");
    }

    fprintf(f, "%s,%s,%ld,%ld,%ld,%ld,%s,%lld,%ld\n", region_name, file_name, line_begin, line_end,
            column_begin, column_end, event_name, count, std::time(nullptr));
    fclose(f);
}

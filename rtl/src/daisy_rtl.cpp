#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "daisy_rtl_internal.h"

Instrumentation_PAPI instrumentation;

#ifdef __cplusplus
extern "C" {
#endif

void __daisy_instrument_init() { instrumentation = Instrumentation_PAPI(); }
void __daisy_instrument_finalize() {}
void __daisy_instrument_enter() { instrumentation.__daisy_instrument_enter(); }
void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                             long line_end, long column_begin, long column_end) {
    instrumentation.__daisy_instrument_exit(region_name, file_name, line_begin, line_end,
                                            column_begin, column_end);
}

#ifdef __cplusplus
}
#endif

// Function pointers for PAPI functions
static int (*_PAPI_library_init)(int) = nullptr;
static int (*_PAPI_create_eventset)(int*) = nullptr;
static int (*_PAPI_add_named_event)(int, const char*) = nullptr;
static int (*_PAPI_start)(int) = nullptr;
static int (*_PAPI_stop)(int, long long*) = nullptr;
static int (*_PAPI_reset)(int) = nullptr;
static const char* (*_PAPI_strerror)(int) = nullptr;
static long long (*_PAPI_get_real_nsec)(void) = nullptr;

void split(std::vector<std::string>& result, std::string s, std::string del = " ") {
    int start, end = -1 * del.size();
    do {
        start = end + del.size();
        end = s.find(del, start);
        result.push_back(s.substr(start, end - start));
    } while (end != -1);
};

void Instrumentation_PAPI::load_papi_symbols() {
    const char* papi_lib = getenv("DAISY_PAPI_PATH");
    if (papi_lib == NULL) papi_lib = "libpapi.so";

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

    if (!_PAPI_library_init || !_PAPI_create_eventset || !_PAPI_add_named_event || !_PAPI_start ||
        !_PAPI_stop || !_PAPI_reset || !_PAPI_strerror || !_PAPI_get_real_nsec) {
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

    char* ver_str = getenv("__DAISY_PAPI_VERSION");
    if (ver_str == NULL) {
        fprintf(stderr, "Environment variable __DAISY_PAPI_VERSION is not set.\n");
        exit(1);
    }

    char* endptr;
    int ver = (int)strtol(ver_str, &endptr, 0);
    if (*endptr != '\0') {
        fprintf(stderr, "Invalid PAPI version: %s\n", ver_str);
        exit(1);
    }

    int retval = _PAPI_library_init(ver);  // PAPI_VER_CURRENT
    if (retval != ver) {
        fprintf(stderr, "Error initializing PAPI! %s\n", _PAPI_strerror(retval));
        exit(1);
    }

    retval = _PAPI_create_eventset(&eventset);
    if (retval != 0) {
        fprintf(stderr, "Error creating eventset! %s\n", _PAPI_strerror(retval));
        exit(1);
    }

    char* events = getenv("__DAISY_INSTRUMENTATION_EVENTS");
    if (events == nullptr) return;

    if (strcmp(events, "DURATION_TIME") == 0) {
        runtime = true;
        event_names.push_back("DURATION_TIME");
    } else {
        // Split the events string by commas and store in event_names
        split(event_names, events, ",");

        for (const auto& event_name : event_names) {
            if (event_name == "DURATION_TIME") {
                fprintf(stderr, "Cannot use DURATION_TIME in list of events\n");
                exit(1);
            }

            retval = _PAPI_add_named_event(eventset, event_name.c_str());
            if (retval != 0) {
                fprintf(stderr, "Error converting event name to code: %s\n", event_name.c_str());
                exit(1);
            }
        }
    }
}

void Instrumentation_PAPI::__daisy_instrument_enter() {
    if (output_file == nullptr || event_names.empty()) return;

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
    if (output_file == nullptr || event_names.empty()) return;

    long long count[event_names.size()];
    if (runtime) {
        count[0] = _PAPI_get_real_nsec() - runtime_start;
    } else {
        int retval = _PAPI_stop(eventset, count);
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

    for (size_t i = 0; i < event_names.size(); ++i) {
        fprintf(f, "%s,%s,%ld,%ld,%ld,%ld,%s,%lld,%ld\n", region_name, file_name, line_begin, line_end,
            column_begin, column_end, event_names.at(i).c_str(), count[i], std::time(nullptr));
    }

    fclose(f);
}

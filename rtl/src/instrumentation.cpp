#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <iterator>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>
#include "daisy_rtl.h"

// -----------------------------------------------------------------------------
//  Dynamic PAPI loading – we avoid a hard dependency on the library at link time
// -----------------------------------------------------------------------------
static int (*_PAPI_library_init)(int) = nullptr;
static int (*_PAPI_create_eventset)(int*) = nullptr;
static int (*_PAPI_add_named_event)(int, const char*) = nullptr;
static int (*_PAPI_start)(int) = nullptr;
static int (*_PAPI_stop)(int, long long*) = nullptr;
static int (*_PAPI_reset)(int) = nullptr;
static const char* (*_PAPI_strerror)(int) = nullptr;
static long long (*_PAPI_get_real_nsec)(void) = nullptr;

static void load_papi_symbols() {
    const char* papi_lib = std::getenv("DAISY_PAPI_PATH");
    if (!papi_lib) papi_lib = "libpapi.so";
    void* handle = dlopen(papi_lib, RTLD_LAZY);
    if (!handle) {
        std::fprintf(stderr, "[daisy-rtl] Failed to load %s: %s\n", papi_lib, dlerror());
        return; // PAPI not available – we will fall back to time based measurement
    }
    _PAPI_library_init = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_library_init"));
    _PAPI_create_eventset = reinterpret_cast<int (*)(int*)>(dlsym(handle, "PAPI_create_eventset"));
    _PAPI_add_named_event = reinterpret_cast<int (*)(int, const char*)>(dlsym(handle, "PAPI_add_named_event"));
    _PAPI_start = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_start"));
    _PAPI_stop = reinterpret_cast<int (*)(int, long long*)>(dlsym(handle, "PAPI_stop"));
    _PAPI_reset = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_reset"));
    _PAPI_strerror = reinterpret_cast<const char* (*) (int)>(dlsym(handle, "PAPI_strerror"));
    _PAPI_get_real_nsec = reinterpret_cast<long long (*)(void)>(dlsym(handle, "PAPI_get_real_nsec"));
}

// -----------------------------------------------------------------------------
//  Helpers / globals
// -----------------------------------------------------------------------------
namespace {
std::vector<std::string> g_event_names;
int g_eventset = -1;
bool g_papi_available = false;
bool g_runtime_only = false; // true when only time measurement is requested
const char* g_output_file = nullptr;
bool g_header_written = false;
bool g_trace_closed = false;
static std::once_flag g_init_once;
std::mutex g_trace_mutex;

long long ns_to_us(long long ns) { return ns / 1000; }

void split_env_list(const char* str, std::vector<std::string>& out) {
    if (!str) return;
    std::string s(str);
    size_t start = 0, end = 0;
    while ((end = s.find(',', start)) != std::string::npos) {
        out.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    out.push_back(s.substr(start));
}

void ensure_global_init() {
    std::call_once(g_init_once, []() {
        load_papi_symbols();
        g_papi_available = _PAPI_library_init && _PAPI_create_eventset && _PAPI_add_named_event && _PAPI_start &&
                           _PAPI_stop && _PAPI_reset && _PAPI_get_real_nsec;

        // Output file
        g_output_file = std::getenv("__DAISY_INSTRUMENTATION_FILE");
        if (!g_output_file) g_output_file = "daisy_trace.json";

        // Events
        const char* env_events = std::getenv("__DAISY_INSTRUMENTATION_EVENTS");
        if (!env_events) env_events = "DURATION_TIME"; // default
        split_env_list(env_events, g_event_names);

        // Check for runtime only mode
        if (g_event_names.size() == 1 && g_event_names[0] == "DURATION_TIME") {
            g_runtime_only = true;
        }

        if (g_papi_available && !g_runtime_only) {
            // Initialise PAPI
            const char* ver_str = std::getenv("__DAISY_PAPI_VERSION");
            if (!ver_str) {
                std::fprintf(stderr, "[daisy-rtl] __DAISY_PAPI_VERSION not set – using PAPI for time only\n");
                g_runtime_only = true;
            } else {
                int ver = std::strtol(ver_str, nullptr, 0);
                int retval = _PAPI_library_init(ver);
                if (retval != ver) {
                    std::fprintf(
                        stderr,
                        "[daisy-rtl] PAPI init failed: %s – falling back to runtime mode\n",
                        _PAPI_strerror ? _PAPI_strerror(retval) : "unknown"
                    );
                    g_runtime_only = true;
                }
            }
        }

        if (g_papi_available && !g_runtime_only) {
            if (_PAPI_create_eventset(&g_eventset) != 0) {
                std::fprintf(stderr, "[daisy-rtl] Failed to create PAPI eventset – falling back to runtime mode\n");
                g_runtime_only = true;
            } else {
                for (const auto& ev : g_event_names) {
                    if (ev == "DURATION_TIME") continue; // handled separately
                    if (_PAPI_add_named_event(g_eventset, ev.c_str()) != 0) {
                        std::fprintf(stderr, "[daisy-rtl] Could not add event %s – ignoring\n", ev.c_str());
                    }
                }
            }
        }
    });
}

void write_event_json(
    const __daisy_metadata* md, long long start_ns, long long dur_ns, const std::vector<long long>& counts
) {
    std::lock_guard<std::mutex> guard(g_trace_mutex);
    FILE* f = std::fopen(g_output_file, g_header_written ? "a" : "w");
    if (!f) {
        std::perror("[daisy-rtl] fopen");
        return;
    }
    if (!g_header_written) {
        std::fprintf(f, "{\"traceEvents\":[\n");
        g_header_written = true;
    } else {
        std::fprintf(f, ",\n");
    }

    std::fprintf(
        f,
        "{\"name\":\"%s\",\"cat\":\"DAISY\",\"ph\":\"X\",\"ts\":%lld,\"dur\":%lld,\"pid\":%d,\"tid\":0,\"args\":{",
        md->region_name,
        ns_to_us(start_ns),
        ns_to_us(dur_ns),
        getpid()
    );
    // Source location metadata
    std::fprintf(
        f,
        "\"file\":\"%s\",\"function\":\"%s\",\"line_begin\":%ld,\"line_end\":%ld,\"column_begin\":%ld,\"column_end\":%"
        "ld",
        md->file_name ? md->file_name : "<unknown>",
        md->function_name ? md->function_name : "<unknown>",
        md->line_begin,
        md->line_end,
        md->column_begin,
        md->column_end
    );
    // PAPI counters
    for (size_t i = 0; i < g_event_names.size() && i < counts.size(); ++i) {
        std::fprintf(f, ",\"%s\":%lld", g_event_names[i].c_str(), counts[i]);
    }
    std::fprintf(f, "}}" /* end args */ "}" /* end event obj */);
    std::fclose(f);
}
} // anonymous namespace

// -----------------------------------------------------------------------------
//  Daisy RTL instrumentation C interface implementation
// -----------------------------------------------------------------------------

struct DaisyInstrumentationContext {
    long long start_ns;
    std::string region_name;
};

extern "C" {

__daisy_instrumentation_t* __daisy_instrumentation_init() {
    ensure_global_init();
    auto* ctx = reinterpret_cast<DaisyInstrumentationContext*>(std::calloc(1, sizeof(DaisyInstrumentationContext)));
    return reinterpret_cast<__daisy_instrumentation_t*>(ctx);
}

void __daisy_instrumentation_enter(__daisy_instrumentation_t* context, __daisy_metadata* metadata) {
    auto* ctx = reinterpret_cast<DaisyInstrumentationContext*>(context);
    if (!ctx || !metadata) return;

    long long start_ns = _PAPI_get_real_nsec ? _PAPI_get_real_nsec()
                                             : (long long) (std::clock()) * (1000000000LL / CLOCKS_PER_SEC);
    ctx->start_ns = start_ns;
    ctx->region_name = metadata->region_name;

    if (!g_runtime_only && g_papi_available) {
        _PAPI_reset(g_eventset);
        _PAPI_start(g_eventset);
    }
}

void __daisy_instrumentation_exit(__daisy_instrumentation_t* context, __daisy_metadata* metadata) {
    auto* ctx = reinterpret_cast<DaisyInstrumentationContext*>(context);
    if (!ctx || !metadata) return;

    long long end_ns = _PAPI_get_real_nsec ? _PAPI_get_real_nsec()
                                           : (long long) (std::clock()) * (1000000000LL / CLOCKS_PER_SEC);

    std::vector<long long> counts(g_event_names.size(), 0);
    if (!g_runtime_only && g_papi_available) {
        _PAPI_stop(g_eventset, counts.data());
    } else {
        counts[0] = end_ns - ctx->start_ns;
    }

    write_event_json(metadata, ctx->start_ns, end_ns - ctx->start_ns, counts);
}

void __daisy_instrumentation_finalize(__daisy_instrumentation_t* context) {
    std::free(context);
    std::lock_guard<std::mutex> guard(g_trace_mutex);
    if (!g_trace_closed && g_header_written) {
        FILE* f = std::fopen(g_output_file, "a");
        if (f) {
            std::fprintf(f, "\n]}\n");
            std::fclose(f);
        }
        g_trace_closed = true;
    }
}

} // extern "C"

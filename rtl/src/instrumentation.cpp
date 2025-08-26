#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include "daisy_rtl/daisy_rtl.h"

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
std::vector<std::string> g_event_names_cpu;
std::vector<std::string> g_event_names_cuda;
int g_eventset_cpu = -1;
int g_eventset_cuda = -1;
bool g_papi_available = false;
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
    if (s.empty()) return;
    out.push_back(s.substr(start));
}

void ensure_global_init() {
    std::call_once(g_init_once, []() {
        load_papi_symbols();
        g_papi_available = _PAPI_library_init && _PAPI_create_eventset && _PAPI_add_named_event && _PAPI_start &&
                           _PAPI_stop && _PAPI_reset && _PAPI_get_real_nsec;
        if (!g_papi_available) {
            std::fprintf(stderr, "[daisy-rtl] PAPI not available.\n");
            exit(EXIT_FAILURE);
        }

        // Initialise PAPI
        const char* ver_str = std::getenv("__DAISY_PAPI_VERSION");
        if (!ver_str) {
            std::fprintf(stderr, "[daisy-rtl] __DAISY_PAPI_VERSION not set.\n");
            exit(EXIT_FAILURE);
        } else {
            int ver = std::strtol(ver_str, nullptr, 0);
            int retval = _PAPI_library_init(ver);
            if (retval != ver) {
                std::fprintf(
                    stderr, "[daisy-rtl] PAPI init failed: %s.\n", _PAPI_strerror ? _PAPI_strerror(retval) : "unknown"
                );
                exit(EXIT_FAILURE);
            }
        }

        // Output file
        g_output_file = std::getenv("__DAISY_INSTRUMENTATION_FILE");
        if (!g_output_file) g_output_file = "daisy_trace.json";

        // Events - CPU
        const char* env_events_cpu = std::getenv("__DAISY_INSTRUMENTATION_EVENTS");
        if (env_events_cpu) {
            split_env_list(env_events_cpu, g_event_names_cpu);
        }

        // Events - CUDA
        const char* env_events_cuda = std::getenv("__DAISY_INSTRUMENTATION_EVENTS_CUDA");
        if (env_events_cuda) {
            split_env_list(env_events_cuda, g_event_names_cuda);
        }

        if (_PAPI_create_eventset(&g_eventset_cpu) != 0 || _PAPI_create_eventset(&g_eventset_cuda) != 0) {
            std::fprintf(stderr, "[daisy-rtl] Failed to create PAPI eventset.\n");
            exit(EXIT_FAILURE);
        }

        if (g_event_names_cpu.size() > 0) {
            for (const auto& ev : g_event_names_cpu) {
                if (_PAPI_add_named_event(g_eventset_cpu, ev.c_str()) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                    exit(EXIT_FAILURE);
                }
            }
        }

        if (g_event_names_cuda.size() > 0) {
            for (const auto& ev : g_event_names_cuda) {
                if (_PAPI_add_named_event(g_eventset_cuda, ev.c_str()) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                    exit(EXIT_FAILURE);
                }
            }
        }
    });
}

void write_event_json(
    const __daisy_metadata* md,
    long long start_ns,
    long long dur_ns,
    const std::vector<long long>& counts,
    enum __daisy_event_set event_set
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

    // Target format:
    // "ph": "X",
    // "cat": "region,daisy",
    // "name": "main  [L110–118]",
    // "pid": 9535,
    // "tid": 0,
    // "ts": 11657,
    // "dur": 613913,
    // "args": {
    // "region_id": "__daisy_correlation18122842100848744318_0_0_1396",
    // "function": "main",
    // "loopnest_index": 0,
    // "module": "correlation",
    // "build_id": "",
    // "source_ranges": [
    //     {
    //     "file": "/home/lukas/repos/docc/tests/polybench/datamining/correlation/correlation.c",
    //     "from": { "line": 110, "col": 17 },
    //     "to":   { "line": 118, "col": 24 }
    //     }
    // ],
    // "metrics": { "CYCLES": 2834021434 } }

    std::stringstream entry;
    entry << "{\"ph\":\"X\",";
    entry << "\"cat\":\"region,daisy\",";
    entry << "\"name\":\"" << md->function_name << " [L" << md->line_begin << "-" << md->line_end << "]\",";
    entry << "\"pid\":" << getpid() << ",";
    entry << "\"tid\":" << gettid() << ",";
    entry << "\"ts\":" << ns_to_us(start_ns) << ",";
    entry << "\"dur\":" << ns_to_us(dur_ns) << ",";
    entry << "\"args\":{";
    entry << "\"region_id\":\"" << md->region_name << "\",";
    entry << "\"function\":\"" << md->function_name << "\",";
    entry << "\"loopnest_index\":\"" << std::to_string(md->loopnest_index) << "\",";
    entry << "\"module\":\"" << std::filesystem::path(md->file_name).filename().string() << "\",";
    entry << "\"build_id\":\"\",";
    entry << "\"source_ranges\":[";
    entry << "{\"file\":\"" << md->file_name << "\",";
    entry << "\"from\":{\"line\":" << md->line_begin << ",\"col\":" << md->column_begin << "},";
    entry << "\"to\":{\"line\":" << md->line_end << ",\"col\":" << md->column_end << "}";
    entry << "}";
    entry << "],\"metrics\":{";
    // PAPI counters
    size_t num_events = event_set == __DAISY_EVENT_SET_CPU ? g_event_names_cpu.size() : g_event_names_cuda.size();
    for (size_t i = 0; i < num_events; ++i) {
        if (event_set == __DAISY_EVENT_SET_CPU) {
            entry << "\"" << g_event_names_cpu[i] << "\":" << counts[i];
        } else {
            entry << "\"" << g_event_names_cuda[i] << "\":" << counts[i];
        }
        if (i < num_events - 1) {
            entry << ",";
        }
    }
    entry << "}";
    entry << "}";
    entry << "}";

    std::fprintf(f, "%s", entry.str().c_str());
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

void __daisy_instrumentation_enter(
    __daisy_instrumentation_t* context, __daisy_metadata* metadata, enum __daisy_event_set event_set
) {
    auto* ctx = reinterpret_cast<DaisyInstrumentationContext*>(context);
    if (!ctx || !metadata) return;

    ctx->start_ns = _PAPI_get_real_nsec();
    ctx->region_name = metadata->region_name;

    if (event_set == __DAISY_EVENT_SET_CPU && g_event_names_cpu.size() > 0) {
        _PAPI_reset(g_eventset_cpu);
        _PAPI_start(g_eventset_cpu);
    } else if (event_set == __DAISY_EVENT_SET_CUDA && g_event_names_cuda.size() > 0) {
        _PAPI_reset(g_eventset_cuda);
        _PAPI_start(g_eventset_cuda);
    }
}

void __daisy_instrumentation_exit(
    __daisy_instrumentation_t* context, __daisy_metadata* metadata, enum __daisy_event_set event_set
) {
    auto* ctx = reinterpret_cast<DaisyInstrumentationContext*>(context);
    if (!ctx || !metadata) return;

    long long end_ns = _PAPI_get_real_nsec();
    long long duration_ns = end_ns - ctx->start_ns;

    size_t num_events = event_set == __DAISY_EVENT_SET_CPU ? g_event_names_cpu.size() : g_event_names_cuda.size();
    std::vector<long long> counts(num_events, 0);
    if (event_set == __DAISY_EVENT_SET_CPU && g_event_names_cpu.size() > 0) {
        _PAPI_stop(g_eventset_cpu, counts.data());
    } else if (event_set == __DAISY_EVENT_SET_CUDA && g_event_names_cuda.size() > 0) {
        _PAPI_stop(g_eventset_cuda, counts.data());
    }

    write_event_json(metadata, ctx->start_ns, duration_ns, counts, event_set);
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

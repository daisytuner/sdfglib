#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iterator>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "daisy_rtl/daisy_rtl.h"

// -----------------------------------------------------------------------------
//  Dynamic PAPI loading – we avoid a hard dependency on the library at link time
// -----------------------------------------------------------------------------
namespace {
static int (*_PAPI_library_init)(int) = nullptr;
static int (*_PAPI_create_eventset)(int*) = nullptr;
static int (*_PAPI_cleanup_eventset)(int) = nullptr;
static int (*_PAPI_destroy_eventset)(int*) = nullptr;
static int (*_PAPI_add_named_event)(int, const char*) = nullptr;
static int (*_PAPI_start)(int) = nullptr;
static int (*_PAPI_stop)(int, long long*) = nullptr;
static int (*_PAPI_reset)(int) = nullptr;
static int (*_PAPI_read)(int, long long*) = nullptr;
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
    _PAPI_cleanup_eventset = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_cleanup_eventset"));
    _PAPI_destroy_eventset = reinterpret_cast<int (*)(int*)>(dlsym(handle, "PAPI_destroy_eventset"));
    _PAPI_add_named_event = reinterpret_cast<int (*)(int, const char*)>(dlsym(handle, "PAPI_add_named_event"));
    _PAPI_start = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_start"));
    _PAPI_stop = reinterpret_cast<int (*)(int, long long*)>(dlsym(handle, "PAPI_stop"));
    _PAPI_reset = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_reset"));
    _PAPI_read = reinterpret_cast<int (*)(int, long long*)>(dlsym(handle, "PAPI_read"));
    _PAPI_strerror = reinterpret_cast<const char* (*) (int)>(dlsym(handle, "PAPI_strerror"));
    _PAPI_get_real_nsec = reinterpret_cast<long long (*)(void)>(dlsym(handle, "PAPI_get_real_nsec"));
}
} // namespace

struct DaisyRegion {
    __daisy_metadata metadata;
    enum __daisy_event_set event_set;

    int papi_eventset = -1;
    std::vector<long long> starts;
    std::vector<long long> durations;

    std::vector<long long> last_counts;
    std::vector<std::vector<long long>> counts;
};

class DaisyInstrumentationState {
private:
    size_t next_region_id = 1;
    std::unordered_map<size_t, DaisyRegion> regions;

    bool papi_available = false;
    std::string output_file;

    std::vector<std::string> event_names_cpu;
    std::vector<std::string> event_names_cuda;

    std::mutex mutex;

    long long ns_to_us(long long ns) { return ns / 1000; }

    void split_string(const char* str, std::vector<std::string>& out) {
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

    void append_event(
        FILE* f,
        const __daisy_metadata* md,
        long long start_ns,
        long long dur_ns,
        const std::vector<long long>& counts,
        const std::vector<std::string>& event_names
    ) {
        // Writes one event as a row in the Chrome trace format

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
        entry << "{";
        entry << "\"ph\":\"X\",";
        entry << "\"cat\":\"region,daisy\",";
        entry << "\"name\":\"" << md->function_name << " [L" << md->line_begin << "-" << md->line_end << "]\",";
        entry << "\"pid\":" << getpid() << ",";
        entry << "\"tid\":" << gettid() << ",";
        entry << "\"ts\":" << ns_to_us(start_ns) << ",";
        entry << "\"dur\":" << ns_to_us(dur_ns) << ",";
        entry << "\"args\":{";
        entry << "\"region_id\":\"" << md->region_name << "\",";
        entry << "\"function\":\"" << md->function_name << "\",";
        entry << "\"loopnest_index\":" << md->loopnest_index << ",";
        entry << "\"module\":\"" << std::filesystem::path(md->file_name).filename().string() << "\",";
        entry << "\"build_id\":\"\",";
        entry << "\"source_ranges\":[";
        entry << "{\"file\":\"" << md->file_name << "\",";
        entry << "\"from\":{\"line\":" << md->line_begin << ",\"col\":" << md->column_begin << "},";
        entry << "\"to\":{\"line\":" << md->line_end << ",\"col\":" << md->column_end << "}";
        entry << "}";
        entry << "],\"metrics\":{";
        // PAPI counters
        for (size_t i = 0; i < counts.size(); ++i) {
            entry << "\"" << event_names[i] << "\":" << counts[i];
            if (i < counts.size() - 1) {
                entry << ",";
            }
        }
        entry << "}";
        entry << "}";
        entry << "}";

        std::fprintf(f, "%s", entry.str().c_str());
    }

public:
    DaisyInstrumentationState() {
        load_papi_symbols();

        this->papi_available = _PAPI_library_init && _PAPI_create_eventset && _PAPI_add_named_event && _PAPI_start &&
                               _PAPI_stop && _PAPI_reset && _PAPI_read && _PAPI_get_real_nsec && _PAPI_strerror &&
                               _PAPI_cleanup_eventset && _PAPI_destroy_eventset;
        if (!this->papi_available) {
            std::fprintf(stderr, "[daisy-rtl] PAPI not available.\n");
            exit(EXIT_FAILURE);
        }

        // Initialise PAPI
        const char* ver_str = std::getenv("__DAISY_PAPI_VERSION");
        if (!ver_str) {
            std::fprintf(stderr, "[daisy-rtl] __DAISY_PAPI_VERSION not set.\n");
            exit(EXIT_FAILURE);
        }
        int ver = std::strtol(ver_str, nullptr, 0);
        int retval = _PAPI_library_init(ver);
        if (retval != ver) {
            std::fprintf(
                stderr, "[daisy-rtl] PAPI init failed: %s.\n", _PAPI_strerror ? _PAPI_strerror(retval) : "unknown"
            );
            exit(EXIT_FAILURE);
        }

        // Output file
        const char* output_file_env = std::getenv("__DAISY_INSTRUMENTATION_FILE");
        if (!output_file_env) {
            this->output_file = "daisy_trace.json";
        } else {
            this->output_file = output_file_env;
        }

        // Events - CPU
        const char* env_events_cpu = std::getenv("__DAISY_INSTRUMENTATION_EVENTS");
        if (env_events_cpu) {
            split_string(env_events_cpu, this->event_names_cpu);
        }

        // Events - CUDA
        const char* env_events_cuda = std::getenv("__DAISY_INSTRUMENTATION_EVENTS_CUDA");
        if (env_events_cuda) {
            split_string(env_events_cuda, this->event_names_cuda);
        }
    }

    ~DaisyInstrumentationState() {
        FILE* f = std::fopen(this->output_file.c_str(), "w");
        if (!f) {
            std::perror("[daisy-rtl] Failed to open output file");
            exit(EXIT_FAILURE);
        }

        // Output file header
        std::fprintf(f, "{\"traceEvents\":[\n");

        // Output all events
        for (auto iter = regions.begin(); iter != regions.end(); ++iter) {
            const auto& region = iter->second;
            for (size_t i = 0; i < region.starts.size(); ++i) {
                auto start = region.starts.at(i);
                auto dur = region.durations.at(i);

                std::vector<long long> counts;
                if (region.counts.size() > i) {
                    counts = region.counts.at(i);
                }
                append_event(
                    f,
                    &region.metadata,
                    start,
                    dur,
                    counts,
                    region.event_set == __DAISY_EVENT_SET_CPU ? event_names_cpu : event_names_cuda
                );

                if (i < region.starts.size() - 1) {
                    std::fprintf(f, ",\n");
                }
            }

            if (std::next(iter) != regions.end()) {
                std::fprintf(f, ",\n");
            }
        }

        // Output file footer
        std::fprintf(f, "\n]}\n");

        // Close output file
        std::fclose(f);
    }

    size_t register_region(const __daisy_metadata* metadata, enum __daisy_event_set event_set) {
        std::lock_guard<std::mutex> lock(mutex);

        DaisyRegion region;
        std::memcpy(&region.metadata, metadata, sizeof(__daisy_metadata));
        region.event_set = event_set;

        if (_PAPI_create_eventset(&region.papi_eventset) != 0) {
            std::fprintf(stderr, "[daisy-rtl] Failed to create PAPI eventset.\n");
            exit(EXIT_FAILURE);
        }
        if (region.event_set == __DAISY_EVENT_SET_CPU) {
            for (const auto& ev : this->event_names_cpu) {
                if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                    exit(EXIT_FAILURE);
                }
            }
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA) {
            for (const auto& ev : this->event_names_cuda) {
                if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                    exit(EXIT_FAILURE);
                }
            }
        }

        size_t region_id = next_region_id++;
        regions[region_id] = region;

        _PAPI_start(region.papi_eventset);
        return region_id;
    }

    void enter_region(size_t region_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: entering unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // Save start counters (before timing)
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());
            region.last_counts = counts;
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());
            region.last_counts = counts;
        }

        // Save start time
        long long start_ns = _PAPI_get_real_nsec();
        region.starts.push_back(start_ns);
    }

    void exit_region(size_t region_id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: exiting unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // Save duration (before counters)
        long long end_ns = _PAPI_get_real_nsec();
        long long duration_ns = end_ns - region.starts.back();
        region.durations.push_back(duration_ns);

        // Save end counters
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            region.counts.push_back(counts);
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            region.counts.push_back(counts);
        }
    }

    void finalize(size_t region_id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: finalizing unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // Stop counters
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_stop(region.papi_eventset, counts.data());
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_stop(region.papi_eventset, counts.data());
        }

        // Cleanup
        if (_PAPI_cleanup_eventset(region.papi_eventset) != 0) {
            std::fprintf(stderr, "[daisy-rtl] Failed to clean up PAPI eventset.\n");
            exit(EXIT_FAILURE);
        }
        if (_PAPI_destroy_eventset(&region.papi_eventset) != 0) {
            std::fprintf(stderr, "[daisy-rtl] Failed to destroy PAPI eventset.\n");
            exit(EXIT_FAILURE);
        }
        region.papi_eventset = -1;
    }
};

static DaisyInstrumentationState g_daisy_state;

#ifdef __cplusplus
extern "C" {
#endif

size_t __daisy_instrumentation_init(__daisy_metadata* metadata, enum __daisy_event_set event_set) {
    if (!metadata) {
        return 0;
    }
    return g_daisy_state.register_region(metadata, event_set);
}

void __daisy_instrumentation_enter(size_t region_id) { g_daisy_state.enter_region(region_id); }

void __daisy_instrumentation_exit(size_t region_id) { g_daisy_state.exit_region(region_id); }

void __daisy_instrumentation_finalize(size_t region_id) { g_daisy_state.finalize(region_id); }

#ifdef __cplusplus
} // extern "C"
#endif

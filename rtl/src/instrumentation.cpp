#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "daisy_rtl/daisy_rtl.h"

#define REGION_CACHE_SIZE 10

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

    long long first_start;
    std::vector<long long> last_counts;

    // Full results
    std::vector<std::vector<long long>> counts;

    // Aggregated results
    std::vector<long long> n_;
    std::vector<double> mean_;
    std::vector<double> variance_;
    std::vector<long long> min_;
    std::vector<long long> max_;
};

class DaisyInstrumentationState {
private:
    size_t next_region_id = 1;
    std::unordered_map<size_t, DaisyRegion> regions;

    // Keep a small cache of "sleeping" regions, i.e., after finalize we don't destroy the PAPI eventset immediately
    // This avoids re-creating PAPI eventsets for regions that are frequently re-entered
    std::list<std::string> sleeping_regions_by_name;
    std::unordered_map<std::string, size_t> sleeping_regions_id_lookup;

    bool aggregate_events = false;
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

    void append_event_aggregated(
        FILE* f,
        const __daisy_metadata* md,
        long long start_ns,
        long long dur_ns,
        const std::vector<double>& means,
        const std::vector<double>& variances,
        const std::vector<long long>& mins,
        const std::vector<long long>& maxs,
        const std::vector<long long>& ns,
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
        for (size_t i = 0; i < event_names.size(); ++i) {
            entry << "\"" << event_names[i] << "\":{";
            entry << "\"mean\":" << means[i] << ",";
            entry << "\"variance\":" << variances[i] << ",";
            entry << "\"count\":" << ns[i] << ",";
            entry << "\"min\":" << mins[i] << ",";
            entry << "\"max\":" << maxs[i];
            entry << "}";
            if (i < event_names.size() - 1) {
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

        // Aggregate events
        const char* aggregate_events_env = std::getenv("__DAISY_INSTRUMENTATION_MODE");
        if (aggregate_events_env && std::strcmp(aggregate_events_env, "aggregate") == 0) {
            this->aggregate_events = true;
        } else {
            this->aggregate_events = false;
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
        // Cleanup cached regions
        for (auto& region_name : this->sleeping_regions_by_name) {
            size_t region_id = this->sleeping_regions_id_lookup[region_name];
            auto& region = this->regions[region_id];

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
        this->sleeping_regions_id_lookup.clear();
        this->sleeping_regions_by_name.clear();

        // Write output file
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

                if (this->aggregate_events) {
                    append_event_aggregated(
                        f,
                        &region.metadata,
                        start,
                        dur,
                        region.mean_,
                        region.variance_,
                        region.n_,
                        region.min_,
                        region.max_,
                        region.event_set == __DAISY_EVENT_SET_CPU ? event_names_cpu : event_names_cuda
                    );
                } else {
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
                }

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

        // Check cache first
        if (this->sleeping_regions_id_lookup.find(metadata->region_name) != this->sleeping_regions_id_lookup.end()) {
            auto region_id = this->sleeping_regions_id_lookup[metadata->region_name];

            // Remove from sleeping regions
            this->sleeping_regions_by_name.remove(metadata->region_name);
            this->sleeping_regions_id_lookup.erase(metadata->region_name);

            // start region
            auto& region = this->regions[region_id];
            _PAPI_start(region.papi_eventset);

            return region_id;
        }

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
        if (this->aggregate_events) {
            if (region.starts.empty()) {
                region.first_start = start_ns;
                region.starts.push_back(start_ns);
            } else {
                region.starts[0] = start_ns;
            }
        } else {
            region.starts.push_back(start_ns);
        }
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
        if (this->aggregate_events) {
            if (region.durations.empty())
                region.durations.push_back(duration_ns);
            else
                region.durations[0] += duration_ns;
            region.starts[0] = region.first_start;
        } else {
            region.durations.push_back(duration_ns);
        }

        // Save end counters
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            if (!this->aggregate_events) {
                region.counts.push_back(counts);
                return;
            }

            // If aggregating, update mean/variance/min/max
            if (region.mean_.empty()) {
                region.n_.resize(counts.size(), 0);
                region.mean_.resize(counts.size(), 0.0);
                region.variance_.resize(counts.size(), 0.0);
                region.min_.resize(counts.size(), std::numeric_limits<long long>::max());
                region.max_.resize(counts.size(), std::numeric_limits<long long>::min());
            }
            for (size_t i = 0; i < counts.size(); ++i) {
                region.n_[i] += 1;

                // Mean/variance using Welford's algorithm

                // delta_1 = x_n - mean_{n-1}
                double delta1 = counts[i] - region.mean_[i];

                // mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                region.mean_[i] += delta1 / region.n_[i];

                // delta_2 = x_n - mean_n
                double delta2 = counts[i] - region.mean_[i];

                // variance_n = variance_{n-1} + (delta_1 * delta_2 - variance_{n-1}) / n
                region.variance_[i] += (delta1 * delta2 - region.variance_[i]) / region.n_[i];

                // Min/max
                if (counts[i] < region.min_[i]) region.min_[i] = counts[i];
                if (counts[i] > region.max_[i]) region.max_[i] = counts[i];
            }
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            if (!this->aggregate_events) {
                region.counts.push_back(counts);
                return;
            }

            // If aggregating, update mean/variance/min/max
            if (region.mean_.empty()) {
                region.n_.resize(counts.size(), 0);
                region.mean_.resize(counts.size(), 0.0);
                region.variance_.resize(counts.size(), 0.0);
                region.min_.resize(counts.size(), std::numeric_limits<long long>::max());
                region.max_.resize(counts.size(), std::numeric_limits<long long>::min());
            }
            for (size_t i = 0; i < counts.size(); ++i) {
                region.n_[i] += 1;

                // Mean/variance using Welford's algorithm

                // delta_1 = x_n - mean_{n-1}
                double delta1 = counts[i] - region.mean_[i];

                // mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                region.mean_[i] += delta1 / region.n_[i];

                // delta_2 = x_n - mean_n
                double delta2 = counts[i] - region.mean_[i];

                // variance_n = variance_{n-1} + (delta_1 * delta_2 - variance_{n-1}) / n
                region.variance_[i] += (delta1 * delta2 - region.variance_[i]) / region.n_[i];

                // Min/max
                if (counts[i] < region.min_[i]) region.min_[i] = counts[i];
                if (counts[i] > region.max_[i]) region.max_[i] = counts[i];
            }
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

        // Append to cache
        if (this->sleeping_regions_id_lookup.find(region.metadata.region_name) ==
            this->sleeping_regions_id_lookup.end()) {
            // Not in cache, add it
            this->sleeping_regions_id_lookup.insert({region.metadata.region_name, region_id});
            this->sleeping_regions_by_name.push_back(region.metadata.region_name);
        }

        // Check if cache has grown to large
        if (this->sleeping_regions_by_name.size() > REGION_CACHE_SIZE) {
            std::string evict_region_name = this->sleeping_regions_by_name.front();
            this->sleeping_regions_by_name.pop_front();

            size_t evict_region_id = this->sleeping_regions_id_lookup[evict_region_name];
            this->sleeping_regions_id_lookup.erase(evict_region_name);

            auto& evict_region = this->regions[evict_region_id];

            if (_PAPI_cleanup_eventset(evict_region.papi_eventset) != 0) {
                std::fprintf(stderr, "[daisy-rtl] EFailed to clean up PAPI eventset.\n");
                exit(EXIT_FAILURE);
            }
            if (_PAPI_destroy_eventset(&evict_region.papi_eventset) != 0) {
                std::fprintf(stderr, "[daisy-rtl] Failed to destroy PAPI eventset.\n");
                exit(EXIT_FAILURE);
            }
            evict_region.papi_eventset = -1;
        }
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

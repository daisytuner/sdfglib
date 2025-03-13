#ifndef __DAISY_INSTRUMENTATION__
#define __DAISY_INSTRUMENTATION__

#include <string>

#include "papi.h"

extern "C" void __daisy_instrument_init();

extern "C" void __daisy_instrument_finalize();

extern "C" void __daisy_instrument_enter();

extern "C" void __daisy_instrument_exit(const char* region_name, const char* file_name,
                                        long line_begin, long line_end, long column_begin,
                                        long column_end);

class Instrumentation_PAPI {
   private:
    int eventset = PAPI_NULL;
    char* event_name;
    char* output_file;

    bool runtime = false;
    long long runtime_start;

   public:
    Instrumentation_PAPI();

    void __daisy_instrument_enter();

    void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                                 long line_end, long column_begin, long column_end);
};

#endif  // __DAISY_INSTRUMENTATION__

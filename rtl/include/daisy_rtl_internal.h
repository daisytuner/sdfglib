#ifndef __DAISY_RTL_INTERNAL_
#define __DAISY_RTL_INTERNAL_

#include "daisy_rtl.h"

class Instrumentation_PAPI {
   private:
    int eventset = -1;
    char* event_name = nullptr;
    char* output_file = nullptr;

    bool runtime = false;
    long long runtime_start = 0;

    void load_papi_symbols();

   public:
    Instrumentation_PAPI();

    void __daisy_instrument_enter();

    void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                                 long line_end, long column_begin, long column_end);
};

#endif  // __DAISY_RTL_INTERNAL_

#pragma once

#include <string>
#include <vector>

#include "daisy_rtl.h"

class Instrumentation_PAPI {
   private:
    int eventset = -1;
    char* output_file = nullptr;
    std::vector<std::string> event_names;

    bool runtime = false;
    long long runtime_start = 0;

    void load_papi_symbols();

   public:
    Instrumentation_PAPI();

    void __daisy_instrument_enter();

    // Deprecated
    void __daisy_instrument_exit(const char* region_name, const char* file_name,
                                 const char* function_name, long line_begin, long line_end,
                                 long column_begin, long column_end);

    void __daisy_instrument_exit_with_metadata(const char* region_name, const char* dbg_file_name,
                                               const char* dbg_function_name, long dbg_line_begin,
                                               long dbg_line_end, long dbg_column_begin,
                                               long dbg_column_end, const char* source_file,
                                               const char* features_file);
};


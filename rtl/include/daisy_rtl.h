#ifndef __DAISY_RTL__
#define __DAISY_RTL__

#ifdef __cplusplus
extern "C" {
#endif

void __daisy_instrument_init();
void __daisy_instrument_finalize();
void __daisy_instrument_enter();

// Deprecated
void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                             long line_end, long column_begin, long column_end);

void __daisy_instrument_exit_with_metadata(
    const char* region_name,
    const char* dbg_file_name,
    long dbg_line_begin,
    long dbg_line_end,
    long dbg_column_begin,
    long dbg_column_end,
    const char* source_file,
    const char* features_file
);

#ifdef __cplusplus
}
#endif

#endif  // __DAISY_RTL__

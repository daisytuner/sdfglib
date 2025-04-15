#ifndef __DAISY_RTL__
#define __DAISY_RTL__

#ifdef __cplusplus
extern "C" {
#endif

void __daisy_instrument_init();
void __daisy_instrument_finalize();
void __daisy_instrument_enter();
void __daisy_instrument_exit(const char* region_name, const char* file_name, long line_begin,
                             long line_end, long column_begin, long column_end);

#ifdef __cplusplus
}
#endif

#endif  // __DAISY_RTL__

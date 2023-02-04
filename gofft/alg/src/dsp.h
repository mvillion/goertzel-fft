#define _USE_MATH_DEFINES	// for C
#include <math.h>

typedef void goertzel_fun_t(
    double *data, long data_len, double k, double *out);

typedef void goertzelf_fun_t(
    float *data, long data_len, float k, float *out);

// Goertzel algorithm (for single tone detection)
goertzel_fun_t goertzel;
goertzel_fun_t goertzel_cx;
goertzel_fun_t goertzel_rad2;
goertzel_fun_t goertzel_rad2_sse;
goertzel_fun_t goertzel_rad4;
goertzel_fun_t goertzel_rad4_avx;
goertzel_fun_t goertzel_rad4_cx_avx;
goertzel_fun_t goertzel_rad4u2_avx;
goertzel_fun_t goertzel_rad4u4_avx;
goertzel_fun_t goertzel_rad8_avx;
goertzel_fun_t goertzel_rad8_cx_avx;
goertzel_fun_t goertzel_rad12_avx;
goertzel_fun_t goertzel_rad12_cx_avx;
goertzel_fun_t goertzel_rad16_avx;
goertzel_fun_t goertzel_rad20_avx;
goertzel_fun_t goertzel_rad24_avx;
goertzel_fun_t goertzel_rad40_avx;
goertzel_fun_t goertzel_rad4x2_test;
goertzel_fun_t goertzel_rad4_fma;
goertzel_fun_t goertzel_rad8_fma;
goertzel_fun_t goertzel_rad20_fma;

goertzel_fun_t goertzel_dft;
goertzel_fun_t goertzel_cx_dft;
goertzel_fun_t goertzel_rad2_dft;
goertzel_fun_t goertzel_rad2_sse_dft;
goertzel_fun_t goertzel_rad4_avx_dft;
goertzel_fun_t goertzel_rad8_avx_dft;
goertzel_fun_t goertzel_rad12_avx_dft;
goertzel_fun_t goertzel_rad16_avx_dft;
goertzel_fun_t goertzel_rad20_avx_dft;
goertzel_fun_t goertzel_rad24_avx_dft;
goertzel_fun_t goertzel_rad40_avx_dft;
goertzel_fun_t goertzel_rad4_fma_dft;
goertzel_fun_t goertzel_rad8_fma_dft;
goertzel_fun_t goertzel_rad20_fma_dft;

goertzelf_fun_t goertzelf;
goertzelf_fun_t goertzelf_cx;
goertzelf_fun_t goertzelf_rad2;
goertzelf_fun_t goertzelf_rad4;
goertzelf_fun_t goertzelf_rad8_avx;
goertzelf_fun_t goertzelf_rad16_avx;
goertzelf_fun_t goertzelf_rad24_avx;
goertzelf_fun_t goertzelf_rad40_avx;

#define _USE_MATH_DEFINES	// for C
#include <math.h>

typedef void (goertzel_fun)(
    double *data, long data_len, double k, double *out);

// Goertzel algorithm (for single tone detection)
goertzel_fun goertzel;

goertzel_fun goertzel;
goertzel_fun goertzel_cx;
goertzel_fun goertzel_rad2;
goertzel_fun goertzel_rad2_sse;
goertzel_fun goertzel_rad4;
goertzel_fun goertzel_rad4_avx;
goertzel_fun goertzel_rad8_avx;
goertzel_fun goertzel_rad12_avx;

goertzel_fun goertzel_dft;
goertzel_fun goertzel_dft_cx;

// void goertzel(double *data, long data_len, double k, double *out);
// void goertzel_cx(double *data, long data_len, double k, double *out);
// void goertzel_rad2(double *data, long data_len, double k, double *out);
// void goertzel_rad2_sse(double *data, long data_len, double k, double *out);
// void goertzel_rad4(double *data, long data_len, double k, double *out);
// void goertzel_rad4_avx(double *data, long data_len, double k, double *out);
// void goertzel_rad8_avx(double *data, long data_len, double k, double *out);
// void goertzel_rad12_avx(double *data, long data_len, double k, double *out);

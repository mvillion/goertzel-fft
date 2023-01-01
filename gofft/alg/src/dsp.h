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

// void goertzel(double *data, long data_len, double k, double *out);
// void goertzel_cx(double *data, long data_len, double k, double *out);
// void goertzel_rad2(double *data, long data_len, double k, double *out);
// void goertzel_rad2_sse(double *data, long data_len, double k, double *out);
// void goertzel_rad4(double *data, long data_len, double k, double *out);
// void goertzel_rad4_avx(double *data, long data_len, double k, double *out);
// void goertzel_rad8_avx(double *data, long data_len, double k, double *out);
// void goertzel_rad12_avx(double *data, long data_len, double k, double *out);

double goertzel_mag(
    double *data, long data_len, int fs, double ft, int filter_size);
void goertzel_m(double* data, long int data_len, int fs, double* ft, int ft_num, int filter_size, double* mag);
double goertzel_rng(double* data, long data_len, int fs, double ft, int filter_size, double rng);

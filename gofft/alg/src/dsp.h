#define _USE_MATH_DEFINES	// for C
#include <math.h>

// Goertzel algorithm (for single tone detection)
void goertzel(double *data, long data_len, double k, double *out);
double goertzel_mag(
    double *data, long data_len, int fs, double ft, int filter_size);
void goertzel_m(double* data, long int data_len, int fs, double* ft, int ft_num, int filter_size, double* mag);
double goertzel_rng(double* data, long data_len, int fs, double ft, int filter_size, double rng);

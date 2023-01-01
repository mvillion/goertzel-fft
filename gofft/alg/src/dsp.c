#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include "dsp.h"


void goertzel(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    double coeff = 2.0*cw;

    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;

    long int i;
    for (i = 0; i < data_len/3*3; i += 3)
    {
        q0 = coeff*q1 - q2 + data[i+0];
        q2 = coeff*q0 - q1 + data[i+1];
        q1 = coeff*q2 - q0 + data[i+2];
    }
    for (; i < data_len; i++)
    {
        q0 = coeff*q1 - q2 + data[i];
        q2 = q1;
        q1 = q0;
    }

    // note: dm00446805-the-goertzel-algorithm-to-compute-individual-terms-of-the-discrete-fourier-transform-dft-stmicroelectronics-1.pdf
    // suggests for non-integer k:
//     w2 = 2*pi*k;
//     cw2 = cos(w2);
//     sw2 = sin(w2);
//     I = It*cw2 + Q*sw2;
//     Q = -It*sw2 + Q*cw2;

    out[0] = q1*cw-q2; // real
    out[1] = q1*sw; // imag
}

void goertzel_cx(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    double coeff = 2.0*cw;

    double q0a = 0.0; // for real part
    double q1a = 0.0;
    double q2a = 0.0;
    double q0b = 0.0; // for imaginary part
    double q1b = 0.0;
    double q2b = 0.0;

    long int i;
    for (i = 0; i < data_len/3*3; i += 3)
    {
        q0a = coeff*q1a - q2a + data[2*i+0];
        q0b = coeff*q1b - q2b + data[2*i+1];
        q2a = coeff*q0a - q1a + data[2*i+2];
        q2b = coeff*q0b - q1b + data[2*i+3];
        q1a = coeff*q2a - q0a + data[2*i+4];
        q1b = coeff*q2b - q0b + data[2*i+5];
    }
    for (; i < data_len; i++)
    {
        q0a = coeff*q1a - q2a + data[2*i+0];
        q0b = coeff*q1b - q2b + data[2*i+1];
        q2a = q1a;
        q2b = q1b;
        q1a = q0a;
        q1b = q0b;
    }

    out[0] = q1a*cw-q2a; // real
    out[1] = q1a*sw; // imag
    out[0] -= q1b*sw; // real
    out[1] += q1b*cw-q2b; // imag
}

double goertzel_mag(
    double *data, long data_len, int fs, double ft, int filter_size)
{
    double out_cx[2];
    double k = floor(0.5 + ((double)(filter_size*ft) / (double)fs));

    goertzel(data, data_len, k, out_cx);
    double sf = (double)data_len; // scale factor: for normalization
    double real = out_cx[0]/sf;
    double imag = out_cx[1]/sf;
    return sqrt(real*real + imag*imag);
}

void goertzel_rad2(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*2*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    double coeff = 2.0*cw;

    double q0a = 0.0; // 1st radix-2 state variables
    double q1a = 0.0;
    double q2a = 0.0;
    double q0b = 0.0; // 2nd radix-2 state variables
    double q1b = 0.0;
    double q2b = 0.0;

    long int i;
    for (i = 0; i < data_len/6*6; i += 6)
    {
        q0a = coeff*q1a - q2a + data[i+0];
        q0b = coeff*q1b - q2b + data[i+1];
        q2a = coeff*q0a - q1a + data[i+2];
        q2b = coeff*q0b - q1b + data[i+3];
        q1a = coeff*q2a - q0a + data[i+4];
        q1b = coeff*q2b - q0b + data[i+5];
    }
    for (; i < data_len/2*2; i += 2)
    {
        q0a = coeff*q1a - q2a + data[i+0];
        q0b = coeff*q1b - q2b + data[i+1];
        q2a = q1a;
        q2b = q1b;
        q1a = q0a;
        q1b = q0b;
    }
    for (; i < data_len; i++)
    {
        q0a = coeff*q1a - q2a + data[i];
        q2a = q1a;
        q1a = q0a;
    }
    double i0 = q1a*cw-q2a;
    double q0 = q1a*sw;
    double i1 = q1b*cw-q2b;
    double q1 = q1b*sw;

    omega = -2.0*M_PI*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    if ((data_len & 1) == 0)
    {
        double i_t = i1;
        // (cos+j*sin)*(i1+j*q1)
        i1 = i_t*cw-sw*q1;
        q1 = i_t*sw+q1*cw;
    }
    else
    {
        double i_t = i0;
        // (cos+j*sin)*(i0+j*q0)
        i0 = i_t*cw-sw*q0;
        q0 = i_t*sw+q0*cw;
    }

    out[0] = i0+i1; // real
    out[1] = q0+q1; // imag
}

void goertzel_rad2_sse(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*2*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    __m128d coeff = _mm_set1_pd(2.0*cw);

    __m128d q0 = _mm_setzero_pd(); // both radix-2 state variables
    __m128d q1 = _mm_setzero_pd();
    __m128d q2 = _mm_setzero_pd();
    __m128d *data_pd = (__m128d *)data;

    long int i;
    for (i = 0; i < data_len/6*6; i += 6)
    {
        q0 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q1), q2), *(data_pd++));
        q2 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q0), q1), *(data_pd++));
        q1 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q2), q0), *(data_pd++));
    }
    for (; i < data_len/2*2; i += 2)
    {
        q0 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q1), q2), *(data_pd++));
        q2 = q1;
        q2 = q1;
        q1 = q0;
        q1 = q0;
    }
    for (; i < data_len; i++)
    {
        __m128d data_i = _mm_set1_pd(data[i]);
        q0 = _mm_add_sd(_mm_sub_sd(_mm_mul_sd(coeff, q1), q2), data_i);
        q2 = _mm_move_sd(q2, q1);
        q1 = _mm_move_sd(q1, q0);
    }

    // back to non-SSE code
    // @todo: finish the work with SSE instructions
    double q1a = q1[0];
    double q1b = q1[1];
    double q2a = q2[0];
    double q2b = q2[1];
    double ia = q1a*cw-q2a;
    double qa = q1a*sw;
    double ib = q1b*cw-q2b;
    double qb = q1b*sw;

    omega = -2.0*M_PI*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    if ((data_len & 1) == 0)
    {
        double i_t = ib;
        // (cos+j*sin)*(ib+j*qb)
        ib = i_t*cw-sw*qb;
        qb = i_t*sw+qb*cw;
    }
    else
    {
        double i_t = ia;
        // (cos+j*sin)*(ia+j*qa)
        ia = i_t*cw-sw*qa;
        qa = i_t*sw+qa*cw;
    }

    out[0] = ia+ib; // real
    out[1] = qa+qb; // imag
}

void goertzel_rad4(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*4*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    double coeff = 2.0*cw;

    double q0a = 0.0; // 1st radix-4 state variables
    double q1a = 0.0;
    double q2a = 0.0;
    double q0b = 0.0; // 2nd radix-4 state variables
    double q1b = 0.0;
    double q2b = 0.0;
    double q0c = 0.0; // 3rd radix-4 state variables
    double q1c = 0.0;
    double q2c = 0.0;
    double q0d = 0.0; // 4th radix-4 state variables
    double q1d = 0.0;
    double q2d = 0.0;

    long int i;
    for (i = 0; i < data_len/12*12; i += 12)
    {
        q0a = coeff*q1a - q2a + data[i+0];
        q0b = coeff*q1b - q2b + data[i+1];
        q0c = coeff*q1c - q2c + data[i+2];
        q0d = coeff*q1d - q2d + data[i+3];
        q2a = coeff*q0a - q1a + data[i+4];
        q2b = coeff*q0b - q1b + data[i+5];
        q2c = coeff*q0c - q1c + data[i+6];
        q2d = coeff*q0d - q1d + data[i+7];
        q1a = coeff*q2a - q0a + data[i+8];
        q1b = coeff*q2b - q0b + data[i+9];
        q1c = coeff*q2c - q0c + data[i+10];
        q1d = coeff*q2d - q0d + data[i+11];
    }
    for (; i < data_len; i += 4)
    {
        // zero-pad values in range data_len/4*4:data_len
        double data_i[4] = {0, 0, 0, 0};
        for (long int j = 0; j < MIN(4, data_len-i); j++)
            data_i[j] = data[i+j];
        q0a = coeff*q1a - q2a + data_i[0];
        q0b = coeff*q1b - q2b + data_i[1];
        q0c = coeff*q1c - q2c + data_i[2];
        q0d = coeff*q1d - q2d + data_i[3];
        q2a = q1a;
        q2b = q1b;
        q2c = q1c;
        q2d = q1d;
        q1a = q0a;
        q1b = q0b;
        q1c = q0c;
        q1d = q0d;
    }
    double iq[8];
    iq[0] = q1a*cw-q2a;
    iq[1] = q1a*sw;
    iq[2] = q1b*cw-q2b;
    iq[3] = q1b*sw;
    iq[4] = q1c*cw-q2c;
    iq[5] = q1c*sw;
    iq[6] = q1d*cw-q2d;
    iq[7] = q1d*sw;

    goertzel_cx(iq, 4, k*4/data_len, out);

    long int n_pad = i-data_len;
    omega = -2.0*M_PI*(4+n_pad)*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    double i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}

void goertzel_rad4_avx(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*4*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    __m256d coeff = _mm256_set1_pd(2.0*cw);

    __m256d q0 = _mm256_setzero_pd(); // quad radix-4 state variables
    __m256d q1 = _mm256_setzero_pd();
    __m256d q2 = _mm256_setzero_pd();
    __m256d *data_pd = (__m256d *)data;

    long int i;
    for (i = 0; i < data_len/12*12; i += 12)
    {
        q0 = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(coeff, q1), q2), *(data_pd++));
        q2 = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(coeff, q0), q1), *(data_pd++));
        q1 = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(coeff, q2), q0), *(data_pd++));
    }
    for (; i < data_len; i += 4)
    {
        // zero-pad values in range data_len/4*4:data_len
        __m256d data_i = _mm256_setzero_pd();
        for (long int j = 0; j < MIN(4, data_len-i); j++)
            data_i[j] = data[i+j];
        q0 = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(coeff, q1), q2), data_i);
        q2 = q1;
        q1 = q0;
    }
    double iq[8];
    iq[0] = q1[0]*cw-q2[0];
    iq[1] = q1[0]*sw;
    iq[2] = q1[1]*cw-q2[1];
    iq[3] = q1[1]*sw;
    iq[4] = q1[2]*cw-q2[2];
    iq[5] = q1[2]*sw;
    iq[6] = q1[3]*cw-q2[3];
    iq[7] = q1[3]*sw;

    goertzel_cx(iq, 4, k*4/data_len, out);

    long int n_pad = i-data_len;
    omega = -2.0*M_PI*(4+n_pad)*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    double i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}

#define RADIX 8
#define GOERTZEL_AVX goertzel_rad8_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 12
#define GOERTZEL_AVX goertzel_rad12_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

void goertzel_rad4x2_avx(
    double *data, long data_len, double k[2], double *out[2])
{
    double omega;
    double sw;
    double cw;
    __m256d coeff[2];

    __m256d q0[2]; // quad radix-4 state variables
    __m256d q1[2];
    __m256d q2[2];
    __m256d *data_pd = (__m256d *)data;

    for (int m = 0; m < 2; m++)
    {
        omega = 2.0*M_PI*4*k[m]/data_len;
        sw = sin(omega);
        cw = cos(omega);
        coeff[m] = _mm256_set1_pd(2.0*cw);

        q0[m] = _mm256_setzero_pd(); // quad radix-4 state variables
        q2[m] = _mm256_setzero_pd();
        q1[m] = _mm256_setzero_pd();
    }

    long int i;
    for (i = 0; i < data_len/12*12; i += 12)
    {
        q0[0] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[0], q1[0]), q2[0]), *(data_pd));
        q0[1] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[1], q1[1]), q2[1]), *(data_pd++));

        q2[0] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[0], q0[0]), q1[0]), *(data_pd));
        q2[1] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[1], q0[1]), q1[1]), *(data_pd++));

        q1[0] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[0], q2[0]), q0[0]), *(data_pd));
        q1[1] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
            coeff[1], q2[1]), q0[1]), *(data_pd++));
    }
    for (; i < data_len; i += 4)
        for (int m = 0; m < 2; m++)
        {
            // zero-pad values in range data_len/4*4:data_len
            __m256d data_i = _mm256_setzero_pd();
            for (long int j = 0; j < MIN(4, data_len-i); j++)
                data_i[j] = data[i+j];
            q0[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff[m], q1[m]), q2[m]), data_i);
            q2[m] = q1[m];
            q1[m] = q0[m];
        }
    double iq[8];
    for (int m = 0; m < 2; m++)
    {
        iq[0] = q1[m][0]*cw-q2[m][0];
        iq[1] = q1[m][0]*sw;
        iq[2] = q1[m][1]*cw-q2[m][1];
        iq[3] = q1[m][1]*sw;
        iq[4] = q1[m][2]*cw-q2[m][2];
        iq[5] = q1[m][2]*sw;
        iq[6] = q1[m][3]*cw-q2[m][3];
        iq[7] = q1[m][3]*sw;

        goertzel_cx(iq, 4, k[m]*4/data_len, out[m]);

        long int n_pad = i-data_len;
        omega = -2.0*M_PI*(4+n_pad)*k[m]/data_len;
        sw = sin(omega);
        cw = cos(omega);

        double i_t = out[m][0];
        // (cw+j*sw)*(i1+j*q1)
        out[m][0] = i_t*cw-out[m][1]*sw;
        out[m][1] = i_t*sw+out[m][1]*cw;
    }
}

void goertzel_mag_m_dumb(
    double* data, long int data_len, int fs, double* ft,  int ft_num,
    int filter_size, double* mag)
{
    for (int cnt = 0; cnt < ft_num; cnt++)
        mag[cnt] = goertzel_mag(data, data_len, fs, ft[cnt], filter_size);
}

void goertzel_m(double* data, long int data_len, int fs, double* ft,
                int ft_num, int filter_size, double* mag)
{
    double k;
    double omega;
    double sine, cosine, coeff, sf;
    double q0, q1, q2, real, imag;
    long int i, dlen;
    int cnt;

    for (cnt = 0; cnt < 2+ft_num*0; cnt++)
    {
        k = floor(0.5 + ((double)(filter_size*ft[cnt]) / (double)fs));
        omega = 2.0*M_PI*k/(double)filter_size;
        sine = sin(omega);
        cosine = cos(omega);
        coeff = 2.0*cosine;
        sf = (double)data_len;

        q0 = 0.0;
        q1 = 0.0;
        q2 = 0.0;

        dlen = data_len - data_len%3;
        for (i = 0; i < dlen; i+=3)
        {
            q0 = coeff*q1 - q2 + data[i];
            q2 = coeff*q0 - q1 + data[i+1];
            q1 = coeff*q2 - q0 + data[i+2];
        }
        for (; i < data_len; i++)
        {
            q0 = coeff*q1 - q2 + data[i];
            q2 = q1;
            q1 = q0;
        }

        real = (q1 - q2*cosine)/sf;
        imag = (q2*sine)/sf;
        mag[cnt] = sqrt(real*real + imag*imag);
    }
}

double goertzel_rng(double* data, long data_len, int fs, double ft,
                    int filter_size, double rng)
{
    double f_step, f_step_normalized, k_s, k_e;
    double omega, sine, cosine, coeff, mag;
    double real, imag;
    double sf;
    double q0, q1, q2;
    double k;
    double f;
    long int i;
    long int dlen;

    f_step = (double)fs/(double)filter_size;
    f_step_normalized = 1.0/(double)filter_size;
    k_s = floor(0.5+ft/f_step);
    k_e = floor(0.5+(ft+rng)/f_step);
    sf = (double)data_len;

    mag = 0.0;

    for (k=k_s; k<k_e; k+=1.0)
    {
        f = k*f_step_normalized;
        omega = 2.0*M_PI*f;
        sine = sin(omega);
        cosine = cos(omega);
        coeff = 2.0*cosine;

        q0 = 0.0;
        q1 = 0.0;
        q2 = 0.0;

        dlen = data_len - data_len%3;
        for (i = 0; i < dlen; i+=3)
        {
            q0 = coeff*q1 - q2 + data[i];
            q2 = coeff*q0 - q1 + data[i+1];
            q1 = coeff*q2 - q0 + data[i+2];
        }
        for (; i < data_len; i++)
        {
            q0 = coeff*q1 - q2 + data[i];
            q2 = q1;
            q1 = q0;
        }

        real = (q1 - q2*cosine)/sf;
        imag = (q2*sine)/sf;
        mag += sqrt(real*real + imag*imag);

    }
    return mag;
}

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include "dsp.h"

#define FLT_TYPE double
#define GOERTZEL_PREFIX(a) goertzel ## a
#include "dsp_simple.c"
#undef GOERTZEL_PREFIX
#undef FLT_TYPE

#define FLT_TYPE float
#define GOERTZEL_PREFIX(a) goertzelf ## a
#include "dsp_simple.c"
#undef GOERTZEL_PREFIX
#undef FLT_TYPE


void goertzel_rad2_sse(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*2*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    __m128d coeff = _mm_set1_pd(2.0*cw);
    __m128d data2;

    __m128d q0 = _mm_setzero_pd(); // both radix-2 state variables
    __m128d q1 = _mm_setzero_pd();
    __m128d q2 = _mm_setzero_pd();
    double *data_ptr = data;

    long int i;
    for (i = 0; i < data_len/6*6; i += 6)
    {
        data2 = _mm_loadu_pd(data_ptr);
        data_ptr += 2;
        q0 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q1), q2), data2);
        data2 = _mm_loadu_pd(data_ptr);
        data_ptr += 2;
        q2 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q0), q1), data2);
        data2 = _mm_loadu_pd(data_ptr);
        data_ptr += 2;
        q1 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q2), q0), data2);
    }
    for (; i < data_len/2*2; i += 2)
    {
        data2 = _mm_loadu_pd(data_ptr);
        data_ptr += 2;
        q0 = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(coeff, q1), q2), data2);
        q2 = q1;
        q2 = q1;
        q1 = q0;
        q1 = q0;
    }
    for (; i < data_len; i++)
    {
        data2 = _mm_set1_pd(data[i]);
        q0 = _mm_add_sd(_mm_sub_sd(_mm_mul_sd(coeff, q1), q2), data2);
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

#define RADIX 4
#define GOERTZEL_AVX goertzel_rad4_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

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

#define RADIX 16
#define GOERTZEL_AVX goertzel_rad16_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 20
#define GOERTZEL_AVX goertzel_rad20_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 24
#define GOERTZEL_AVX goertzel_rad24_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 40
#define GOERTZEL_AVX goertzel_rad40_avx
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

//------------------------------------------------------------------------------
#define RADIX 8
#define GOERTZEL_AVX goertzel_rad4_cx_avx
#include "dsp_avx_cx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 16
#define GOERTZEL_AVX goertzel_rad8_cx_avx
#include "dsp_avx_cx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 24
#define GOERTZEL_AVX goertzel_rad12_cx_avx
#include "dsp_avx_cx.c"
#undef GOERTZEL_AVX
#undef RADIX

//------------------------------------------------------------------------------
#define RADIX 8
#define GOERTZEL_AVX goertzelf_rad8_avx
#include "dspf_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 16
#define GOERTZEL_AVX goertzelf_rad16_avx
#include "dspf_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 24
#define GOERTZEL_AVX goertzelf_rad24_avx
#include "dspf_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 40
#define GOERTZEL_AVX goertzelf_rad40_avx
#include "dspf_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#pragma GCC push_options //----------------------------------------------------
#pragma GCC target("fma")

#define RADIX 4
#define GOERTZEL_AVX goertzel_rad4_fma
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 8
#define GOERTZEL_AVX goertzel_rad8_fma
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#define RADIX 20
#define GOERTZEL_AVX goertzel_rad20_fma
#include "dsp_avx.c"
#undef GOERTZEL_AVX
#undef RADIX

#pragma GCC pop_options //-----------------------------------------------------

#define UNROLL_FACTOR 2
#define RADIX 4
#define GOERTZEL_AVX goertzel_rad4u2_avx
#include "dsp_avx_unroll.c"
#undef GOERTZEL_AVX
#undef RADIX
#undef UNROLL_FACTOR

#define UNROLL_FACTOR 4
#define RADIX 4
#define GOERTZEL_AVX goertzel_rad4u4_avx
#include "dsp_avx_unroll.c"
#undef GOERTZEL_AVX
#undef RADIX
#undef UNROLL_FACTOR

#define N_RADIX 2
#define GOERTZEL_AVX goertzel_rad4x2_avx
#include "dsp_rad4x_avx.c"
#undef GOERTZEL_AVX
#undef N_RADIX

void goertzel_rad4x2_test(double *data, long data_len, double k, double *out)
{
    double out2[4];
    double k2[2] = {k, k+1};
    goertzel_rad4x2_avx(data, data_len, k2, out2);
    out[0] = out2[0];
    out[1] = out2[1];
}

void goertzel_dft(double *data, long data_len, double k, double *out)
{
    double *out_end = out+2*data_len;

    goertzel(data, data_len, (double)0, out);
    out += 2;
    out_end -= 2;

    for (long i = 1; i < data_len/2+1; i++)
    {
        goertzel(data, data_len, (double)i, out);
        out_end[0] = out[0];
        out_end[1] = -out[1];
        out += 2;
        out_end -= 2;
    }
}

void goertzel_dft_cx(double *data, long data_len, double k, double *out)
{
    for (long i = 0; i < data_len; i++)
    {
        goertzel_cx(data, data_len, (double)i, out);
        out += 2;
    }
}

void goertzel_dft_rad2(double *data, long data_len, double k, double *out)
{
    double *out_end = out+2*data_len;

    goertzel(data, data_len, (double)0, out);
    out += 2;
    out_end -= 2;

    for (long i = 1; i < data_len/2+1; i++)
    {
        goertzel_rad2(data, data_len, (double)i, out);
        out_end[0] = out[0];
        out_end[1] = -out[1];
        out += 2;
        out_end -= 2;
    }
}

void goertzel_dft_rad2_sse(double *data, long data_len, double k, double *out)
{
    double *out_end = out+2*data_len;

    goertzel(data, data_len, (double)0, out);
    out += 2;
    out_end -= 2;

    for (long i = 1; i < data_len/2+1; i++)
    {
        goertzel_rad2_sse(data, data_len, (double)i, out);
        out_end[0] = out[0];
        out_end[1] = -out[1];
        out += 2;
        out_end -= 2;
    }
}

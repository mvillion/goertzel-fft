// code for radix-RADIX with single frequency
static void concat(GOERTZEL_AVX, _core)(
    double *data, long data_len, double k, double *iq, long int *n_pad)
{
    double omega = 2.0*M_PI*RADIX*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);
    __m256d coeff = _mm256_set1_pd(2.0*cw);
    __m256d data4;

    __m256d q0[RADIX/4]; // radix-RADIX state variables
    __m256d q1[RADIX/4];
    __m256d q2[RADIX/4];
    for (int m = 0; m < RADIX/4; m++)
    {
        q0[m] = _mm256_setzero_pd();
        q1[m] = _mm256_setzero_pd();
        q2[m] = _mm256_setzero_pd();
    }
    double *data_ptr = data;

    long int i;
    long int step = RADIX*3;
    for (i = 0; i < data_len/step*step; i += step)
    {
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/4; m++)
        {
            data4 = _mm256_loadu_pd(data_ptr);
            data_ptr += 4;
            q0[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff, q1[m]), q2[m]), data4);
        }
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/4; m++)
        {
            data4 = _mm256_loadu_pd(data_ptr);
            data_ptr += 4;
            q2[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff, q0[m]), q1[m]), data4);
        }
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/4; m++)
        {
            data4 = _mm256_loadu_pd(data_ptr);
            data_ptr += 4;
            q1[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff, q2[m]), q0[m]), data4);
        }
    }
    for (; i < data_len; i += RADIX)
        for (int m = 0; m < RADIX/4; m++)
        {
            // zero-pad values in range data_len/4*4:data_len
            data4 = _mm256_setzero_pd();
            for (long int j = 0; j < MAX(0, MIN(4, data_len-i-4*m)); j++)
                data4[j] = data[i+4*m+j];
            q0[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff, q1[m]), q2[m]), data4);
            q2[m] = q1[m];
            q1[m] = q0[m];
        }
    *n_pad = i-data_len;

    for (int m = 0; m < RADIX/4; m++)
    {
        __m256d datai4 = _mm256_mul_pd(q1[m], _mm256_set1_pd(cw));
        datai4 = _mm256_sub_pd(datai4, q2[m]);
        __m256d dataq4 = _mm256_mul_pd(q1[m], _mm256_set1_pd(sw));
        iq[8*m+0] = datai4[0];
        iq[8*m+1] = dataq4[0];
        iq[8*m+2] = datai4[1];
        iq[8*m+3] = dataq4[1];
        iq[8*m+4] = datai4[2];
        iq[8*m+5] = dataq4[2];
        iq[8*m+6] = datai4[3];
        iq[8*m+7] = dataq4[3];
    }
}

void GOERTZEL_AVX(double *data, long data_len, double k, double *out)
{
    double iq[RADIX*2];
    long int n_pad;
    concat(GOERTZEL_AVX, _core)(data, data_len, k, iq, &n_pad);

    goertzel_cx(iq, RADIX, k*RADIX/data_len, out);

    double omega = -2.0*M_PI*(RADIX+n_pad)*k/data_len;
    double sw = sin(omega);
    double cw = cos(omega);

    double i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}

static void concat(GOERTZEL_AVX, _dft_slow)(
    double *data, long data_len, double k, double *out)
{
    double *out_end = out+2*data_len;

    goertzel(data, data_len, (double)0, out);
    out += 2;
    out_end -= 2;

    for (long i = 1; i < data_len/2+1; i++)
    {
        GOERTZEL_AVX(data, data_len, (double)i, out);
        out_end[0] = out[0];
        out_end[1] = -out[1];
        out += 2;
        out_end -= 2;
    }
}

// static void concat(GOERTZEL_AVX, _dft_fast)(
//     double *data, long n_radix, double k, double *out)
// {
//     double iq[RADIX*2];
//     double out2[2];
//     long int n_pad;
//     for (long i_radix = 0; i_radix < n_radix; i_radix++)
//         for (long r = 0; r < RADIX; r++)
//         {
//             long i = n_radix*r+i_radix;
//             concat(GOERTZEL_AVX, _core)(
//                 data, n_radix*RADIX, (double)i, iq, &n_pad);
//
//             goertzel_cx(iq, RADIX, (double)i/n_radix, out2);
//
//             double omega = -2.0*M_PI*(RADIX+n_pad)*i/(n_radix*RADIX);
//             double sw = sin(omega);
//             double cw = cos(omega);
//
//             // (cw+j*sw)*(i1+j*q1)
//             out[2*i+0] = out2[0]*cw-out2[1]*sw;
//             out[2*i+1] = out2[0]*sw+out2[1]*cw;
//         }
// }

static void concat(GOERTZEL_AVX, _dft_fast)(
    double *data, long n_radix, double k, double *out)
{
    double iq[RADIX*2];
    double out2[2];
    long int n_pad;
    for (long i_radix = 0; i_radix < n_radix; i_radix++)
    {
        // omega = 2.0*M_PI*RADIX*(n_radix*r+i_radix)/(n_radix*RADIX)
        // omega = 2.0*M_PI*(n_radix*r+i_radix)/n_radix
        // omega = 2.0*M_PI*i_radix/n_radix
        // omega does not depend on r
        concat(GOERTZEL_AVX, _core)(
            data, n_radix*RADIX, (double)i_radix, iq, &n_pad);

        for (long r = 0; r < RADIX; r++)
        {
            long i = n_radix*r+i_radix;

            goertzel_cx(iq, RADIX, (double)i/n_radix, out2);

            double omega = -2.0*M_PI*(RADIX+n_pad)*i/(n_radix*RADIX);
            double sw = sin(omega);
            double cw = cos(omega);

            // (cw+j*sw)*(i1+j*q1)
            out[2*i+0] = out2[0]*cw-out2[1]*sw;
            out[2*i+1] = out2[0]*sw+out2[1]*cw;
        }
    }
}

void concat(GOERTZEL_AVX, _dft)(
    double *data, long data_len, double k, double *out)
{
    if (data_len % RADIX == 0)
        concat(GOERTZEL_AVX, _dft_fast)(data, data_len/RADIX, k, out);
    else
        concat(GOERTZEL_AVX, _dft_slow)(data, data_len, k, out);
}

// cos(2*w) = 1-2*sin(w)*sin(w)
// sin(2*w) = 2*cos(w)*sin(w)

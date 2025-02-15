// code for radix-RADIX with single frequency
void GOERTZEL_AVX(double *data, long data_len, double k, double *out)
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
    long int step = RADIX*3*UNROLL_FACTOR;
    for (i = 0; i < data_len/step*step; i += step)
    {
        #pragma GCC unroll 8
        for (int i_unroll = 0; i_unroll < UNROLL_FACTOR; i_unroll++)
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
    double iq[RADIX*2];
    for (int m = 0; m < RADIX/4; m++)
    {
        iq[8*m+0] = q1[m][0]*cw-q2[m][0];
        iq[8*m+1] = q1[m][0]*sw;
        iq[8*m+2] = q1[m][1]*cw-q2[m][1];
        iq[8*m+3] = q1[m][1]*sw;
        iq[8*m+4] = q1[m][2]*cw-q2[m][2];
        iq[8*m+5] = q1[m][2]*sw;
        iq[8*m+6] = q1[m][3]*cw-q2[m][3];
        iq[8*m+7] = q1[m][3]*sw;
    }

    goertzel_cx(iq, RADIX, k*RADIX/data_len, out);

    long int n_pad = i-data_len;
    omega = -2.0*M_PI*(RADIX+n_pad)*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    double i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}

// cos(2*w) = 1-2*sin(w)*sin(w)
// sin(2*w) = 2*cos(w)*sin(w)

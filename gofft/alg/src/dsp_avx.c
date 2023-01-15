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
    double iq[RADIX*2];
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

// code for radix-4 with N_RADIX frequencies
void GOERTZEL_AVX(double *data, long data_len, double *k, double *out)
{
    double omega;
    double sw[N_RADIX];
    double cw[N_RADIX];
    __m256d coeff[N_RADIX];
    __m256d data4;

    __m256d q0[N_RADIX]; // quad radix-4 state variables
    __m256d q1[N_RADIX];
    __m256d q2[N_RADIX];
    double *data_ptr = data;

    for (int m = 0; m < N_RADIX; m++)
    {
        omega = 2.0*M_PI*4*k[m]/data_len;
        sw[m] = sin(omega);
        cw[m] = cos(omega);
        coeff[m] = _mm256_set1_pd(2.0*cw[m]);

        q0[m] = _mm256_setzero_pd(); // quad radix-4 state variables
        q1[m] = _mm256_setzero_pd();
        q2[m] = _mm256_setzero_pd();
    }

    long int i;
    long int step = 12;
    for (i = 0; i < data_len/step*step; i += step)
    {
        data4 = _mm256_loadu_pd(data_ptr);
        data_ptr += 4;
        #pragma GCC unroll 8
        for (int m = 0; m < N_RADIX; m++)
        {
            q0[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff[m], q1[m]), q2[m]), data4);
        }
        data4 = _mm256_loadu_pd(data_ptr);
        data_ptr += 4;
        #pragma GCC unroll 8
        for (int m = 0; m < N_RADIX; m++)
        {
            q2[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff[m], q0[m]), q1[m]), data4);
        }
        data4 = _mm256_loadu_pd(data_ptr);
        data_ptr += 4;
        #pragma GCC unroll 8
        for (int m = 0; m < N_RADIX; m++)
        {
            q1[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff[m], q2[m]), q0[m]), data4);
        }
    }
    for (; i < data_len; i += 4)
    {
        // zero-pad values in range data_len/4*4:data_len
        data4 = _mm256_setzero_pd();
        for (long int j = 0; j < MIN(4, data_len-i); j++)
            data4[j] = data[i+j];
        #pragma GCC unroll 8
        for (int m = 0; m < N_RADIX; m++)
        {
            q0[m] = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(
                coeff[m], q1[m]), q2[m]), data4);
            q2[m] = q1[m];
            q1[m] = q0[m];
        }
    }
    double iq[8];
    for (int m = 0; m < N_RADIX; m++)
    {
//         iq[0] = q1[m][0]*cw[m]-q2[m][0];
//         iq[1] = q1[m][0]*sw[m];
//         iq[2] = q1[m][1]*cw[m]-q2[m][1];
//         iq[3] = q1[m][1]*sw[m];
//         iq[4] = q1[m][2]*cw[m]-q2[m][2];
//         iq[5] = q1[m][2]*sw[m];
//         iq[6] = q1[m][3]*cw[m]-q2[m][3];
//         iq[7] = q1[m][3]*sw[m];
        __m256d datai4 = _mm256_mul_pd(q1[m], _mm256_set1_pd(cw[m]));
        datai4 = _mm256_sub_pd(datai4, q2[m]);
        __m256d dataq4 = _mm256_mul_pd(q1[m], _mm256_set1_pd(sw[m]));
        iq[0] = datai4[0];
        iq[1] = dataq4[0];
        iq[2] = datai4[1];
        iq[3] = dataq4[1];
        iq[4] = datai4[2];
        iq[5] = dataq4[2];
        iq[6] = datai4[3];
        iq[7] = dataq4[3];

        goertzel_cx(iq, 4, k[m]*4/data_len, out+2*m);

        long int n_pad = i-data_len;
        omega = -2.0*M_PI*(4+n_pad)*k[m]/data_len;
        sw[m] = sin(omega);
        cw[m] = cos(omega);

        double i_t = out[2*m+0];
        // (cw+j*sw)*(i1+j*q1)
        out[2*m+0] = i_t*cw[m]-out[2*m+1]*sw[m];
        out[2*m+1] = i_t*sw[m]+out[2*m+1]*cw[m];
    }
}

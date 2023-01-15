// code for radix-RADIX with single frequency
void GOERTZEL_AVX(float *data, long data_len, float k, float *out)
{
    float omega = 2.0*M_PI*RADIX*k/data_len;
    float sw = sinf(omega);
    float cw = cosf(omega);
    __m256 coeff = _mm256_set1_ps(2.0*cw);
    __m256 data8;

    __m256 q0[RADIX/8]; // radix-RADIX state variables
    __m256 q1[RADIX/8];
    __m256 q2[RADIX/8];
    for (int m = 0; m < RADIX/8; m++)
    {
        q0[m] = _mm256_setzero_ps();
        q1[m] = _mm256_setzero_ps();
        q2[m] = _mm256_setzero_ps();
    }
    float *data_ptr = data;

    long int i;
    long int step = RADIX*3;
    for (i = 0; i < data_len/step*step; i += step)
    {
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/8; m++)
        {
            data8 = _mm256_loadu_ps(data_ptr);
            data_ptr += 8;
            q0[m] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(
                coeff, q1[m]), q2[m]), data8);
        }
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/8; m++)
        {
            data8 = _mm256_loadu_ps(data_ptr);
            data_ptr += 8;
            q2[m] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(
                coeff, q0[m]), q1[m]), data8);
        }
        #pragma GCC unroll 8
        for (int m = 0; m < RADIX/8; m++)
        {
            data8 = _mm256_loadu_ps(data_ptr);
            data_ptr += 8;
            q1[m] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(
                coeff, q2[m]), q0[m]), data8);
        }
    }
    for (; i < data_len; i += RADIX)
        for (int m = 0; m < RADIX/8; m++)
        {
            // zero-pad values in range data_len/8*8:data_len
            data8 = _mm256_setzero_ps();
            for (long int j = 0; j < MAX(0, MIN(8, data_len-i-8*m)); j++)
                data8[j] = data[i+8*m+j];
            q0[m] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(
                coeff, q1[m]), q2[m]), data8);
            q2[m] = q1[m];
            q1[m] = q0[m];
        }
    float iq[RADIX*2];
    for (int m = 0; m < RADIX/8; m++)
    {
        __m256 datai8 = _mm256_mul_ps(q1[m], _mm256_set1_ps(cw));
        datai8 = _mm256_sub_ps(datai8, q2[m]);
        __m256 dataq8 = _mm256_mul_ps(q1[m], _mm256_set1_ps(sw));
        iq[16*m+0] = datai8[0];
        iq[16*m+1] = dataq8[0];
        iq[16*m+2] = datai8[1];
        iq[16*m+3] = dataq8[1];
        iq[16*m+4] = datai8[2];
        iq[16*m+5] = dataq8[2];
        iq[16*m+6] = datai8[3];
        iq[16*m+7] = dataq8[3];
        iq[16*m+8] = datai8[4];
        iq[16*m+9] = dataq8[4];
        iq[16*m+10] = datai8[5];
        iq[16*m+11] = dataq8[5];
        iq[16*m+12] = datai8[6];
        iq[16*m+13] = dataq8[6];
        iq[16*m+14] = datai8[7];
        iq[16*m+15] = dataq8[7];
    }

    goertzelf_cx(iq, RADIX, k*RADIX/data_len, out);

    long int n_pad = i-data_len;
    omega = -2.0*M_PI*(RADIX+n_pad)*k/data_len;
    sw = sinf(omega);
    cw = cosf(omega);

    float i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}

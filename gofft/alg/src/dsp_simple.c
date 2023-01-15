void GOERTZEL_PREFIX()(FLT_TYPE *data, long data_len, FLT_TYPE k, FLT_TYPE *out)
{
    FLT_TYPE omega = 2.0*M_PI*k/data_len;
    FLT_TYPE sw = sin(omega);
    FLT_TYPE cw = cos(omega);
    FLT_TYPE coeff = 2.0*cw;

    FLT_TYPE q0 = 0.0;
    FLT_TYPE q1 = 0.0;
    FLT_TYPE q2 = 0.0;

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

void GOERTZEL_PREFIX(_cx)(
    FLT_TYPE *data, long data_len, FLT_TYPE k, FLT_TYPE *out)
{
    FLT_TYPE omega = 2.0*M_PI*k/data_len;
    FLT_TYPE sw = sin(omega);
    FLT_TYPE cw = cos(omega);
    FLT_TYPE coeff = 2.0*cw;

    FLT_TYPE q0a = 0.0; // for real part
    FLT_TYPE q1a = 0.0;
    FLT_TYPE q2a = 0.0;
    FLT_TYPE q0b = 0.0; // for imaginary part
    FLT_TYPE q1b = 0.0;
    FLT_TYPE q2b = 0.0;

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

FLT_TYPE GOERTZEL_PREFIX(_mag)(
    FLT_TYPE *data, long data_len, int fs, FLT_TYPE ft, int filter_size)
{
    FLT_TYPE out_cx[2];
    FLT_TYPE k = floor(0.5 + ((FLT_TYPE)(filter_size*ft) / (FLT_TYPE)fs));

    GOERTZEL_PREFIX()(data, data_len, k, out_cx);
    FLT_TYPE sf = (FLT_TYPE)data_len; // scale factor: for normalization
    FLT_TYPE real = out_cx[0]/sf;
    FLT_TYPE imag = out_cx[1]/sf;
    return sqrt(real*real + imag*imag);
}

void GOERTZEL_PREFIX(_rad2)(
    FLT_TYPE *data, long data_len, FLT_TYPE k, FLT_TYPE *out)
{
    FLT_TYPE omega = 2.0*M_PI*2*k/data_len;
    FLT_TYPE sw = sin(omega);
    FLT_TYPE cw = cos(omega);
    FLT_TYPE coeff = 2.0*cw;

    FLT_TYPE q0a = 0.0; // 1st radix-2 state variables
    FLT_TYPE q1a = 0.0;
    FLT_TYPE q2a = 0.0;
    FLT_TYPE q0b = 0.0; // 2nd radix-2 state variables
    FLT_TYPE q1b = 0.0;
    FLT_TYPE q2b = 0.0;

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
    FLT_TYPE i0 = q1a*cw-q2a;
    FLT_TYPE q0 = q1a*sw;
    FLT_TYPE i1 = q1b*cw-q2b;
    FLT_TYPE q1 = q1b*sw;

    omega = -2.0*M_PI*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    if ((data_len & 1) == 0)
    {
        FLT_TYPE i_t = i1;
        // (cos+j*sin)*(i1+j*q1)
        i1 = i_t*cw-sw*q1;
        q1 = i_t*sw+q1*cw;
    }
    else
    {
        FLT_TYPE i_t = i0;
        // (cos+j*sin)*(i0+j*q0)
        i0 = i_t*cw-sw*q0;
        q0 = i_t*sw+q0*cw;
    }

    out[0] = i0+i1; // real
    out[1] = q0+q1; // imag
}

void GOERTZEL_PREFIX(_rad4)(
    FLT_TYPE *data, long data_len, FLT_TYPE k, FLT_TYPE *out)
{
    FLT_TYPE omega = 2.0*M_PI*4*k/data_len;
    FLT_TYPE sw = sin(omega);
    FLT_TYPE cw = cos(omega);
    FLT_TYPE coeff = 2.0*cw;

    FLT_TYPE q0a = 0.0; // 1st radix-4 state variables
    FLT_TYPE q1a = 0.0;
    FLT_TYPE q2a = 0.0;
    FLT_TYPE q0b = 0.0; // 2nd radix-4 state variables
    FLT_TYPE q1b = 0.0;
    FLT_TYPE q2b = 0.0;
    FLT_TYPE q0c = 0.0; // 3rd radix-4 state variables
    FLT_TYPE q1c = 0.0;
    FLT_TYPE q2c = 0.0;
    FLT_TYPE q0d = 0.0; // 4th radix-4 state variables
    FLT_TYPE q1d = 0.0;
    FLT_TYPE q2d = 0.0;

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
        FLT_TYPE data_i[4] = {0, 0, 0, 0};
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
    FLT_TYPE iq[8];
    iq[0] = q1a*cw-q2a;
    iq[1] = q1a*sw;
    iq[2] = q1b*cw-q2b;
    iq[3] = q1b*sw;
    iq[4] = q1c*cw-q2c;
    iq[5] = q1c*sw;
    iq[6] = q1d*cw-q2d;
    iq[7] = q1d*sw;

    GOERTZEL_PREFIX(_cx)(iq, 4, k*4/data_len, out);

    long int n_pad = i-data_len;
    omega = -2.0*M_PI*(4+n_pad)*k/data_len;
    sw = sin(omega);
    cw = cos(omega);

    FLT_TYPE i_t = out[0];
    // (cw+j*sw)*(i1+j*q1)
    out[0] = i_t*cw-out[1]*sw;
    out[1] = i_t*sw+out[1]*cw;
}


#include <stdio.h>
#include <stdlib.h>
#include "dsp.h"

void goertzel(double *data, long data_len, double k, double *out)
{
    double omega = 2.0*M_PI*k/data_len;
    double sine = sin(omega);
    double cosine = cos(omega);
    double coeff = 2.0*cosine;

    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;

    long int i;
    long int dlen = data_len - data_len%3;
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

    // note: dm00446805-the-goertzel-algorithm-to-compute-individual-terms-of-the-discrete-fourier-transform-dft-stmicroelectronics-1.pdf
    // suggests for non-integer k:
//     w2 = 2*pi*k;
//     cw2 = cos(w2);
//     sw2 = sin(w2);
//     I = It*cw2 + Q*sw2;
//     Q = -It*sw2 + Q*cw2;

    out[0] = q1*cosine-q2; // real
    out[1] = q1*sine; // imag
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

double goertzel_avx(double* data, long data_len, int fs, double ft,
                int filter_size)
{
    double k;		// Related to frequency bins
    double omega;
    double sine, cosine, coeff, sf, mag;
    double q0, q1, q2, real, imag;
    long int i;
    long int dlen;

    k = floor(0.5 + ((double)(filter_size*ft) / (double)fs));

    omega = 2.0*M_PI*k/(double)filter_size;
    sine = sin(omega);
    cosine = cos(omega);
    coeff = 2.0*cosine;
    sf = (double)data_len;		// scale factor: for normalization

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
    mag = sqrt(real*real + imag*imag);

    return mag;
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

#!/usr/bin/python3
import numpy as np
import sys

from gofft_directory import dsp_ext
from time import time


def bench_goertzel(data_len, n_test=1000):
    rng = np.random.Generator(np.random.SFC64())
    in_data = rng.random((n_test, data_len), np.float64)

    cost = {}
    x = np.arange(data_len)
    F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
    t0 = time()
    out_dft = in_data @ F
    cost["dft"] = time()-t0

    t0 = time()
    out_fft = np.fft.fft(in_data)
    cost["fft"] = time()-t0

    max_diff = (out_fft-out_dft).std(axis=-1).max()
    print("maximal std diff between dft and fft %g" % max_diff)

    out_goertzel = np.empty_like(out_fft)
    cost["goertzel"] = 0
    for k in range(data_len):
        t0 = time()
        out_k = dsp_ext.goertzel(in_data, data_len, k, data_len)
        cost["goertzel"] += time()-t0
        out_goertzel[:, k] = out_k[:, 0]+1j*out_k[:, 1]

    cost["goertzel"] /= data_len

    max_diff = (out_goertzel-out_dft).std(axis=-1).max()
    print("maximal std diff between dft and goertzel %g" % max_diff)

    return cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("dummy exit")
        sys.exit()
    cost_dict = bench_goertzel(16)
    print(", ".join(["%s %f" % tup for tup in cost_dict.items()]))

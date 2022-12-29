#!/usr/bin/python3
import numpy as np
import sys

from enum import Enum
from gofft_directory import dsp_ext
from time import time


class BenchType(Enum):
    dft = 0
    fft = 1
    goertzel = 2


def bench_goertzel(data_len, n_test=10000):
    rng = np.random.Generator(np.random.SFC64())
    in_data = rng.random((n_test, data_len), np.float64)

    cost = np.empty(len(BenchType))
    x = np.arange(data_len)
    F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
    t0 = time()
    out_dft = in_data @ F
    cost[BenchType.dft.value] = time()-t0

    t0 = time()
    out_fft = np.fft.fft(in_data)
    cost[BenchType.fft.value] = time()-t0

    max_diff = (out_fft-out_dft).std(axis=-1).max()
    print("maximal std diff between dft and fft %g" % max_diff)

    out_goertzel = np.empty_like(out_fft)
    cost_goertzel = 0
    for k in range(data_len):
        t0 = time()
        out_k = dsp_ext.goertzel(in_data, k)
        cost_goertzel += time()-t0
        out_goertzel[:, k] = out_k[:, 0]+1j*out_k[:, 1]

    cost[BenchType.goertzel.value] = cost_goertzel/data_len

    max_diff = (out_goertzel-out_dft).std(axis=-1).max()
    print("maximal std diff between dft and goertzel %g" % max_diff)

    return cost


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("dummy exit")
        sys.exit()
    cost = bench_goertzel(16)
    print(", ".join(["%s %f" % (k.name, cost[k.value]) for k in BenchType]))

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
    # goertzel_radix2_py = 3


def goertzel_radix2(in_data, k):
    shape = in_data.shape
    in_data = in_data.reshape(-1, shape[-1])
    iq0 = dsp_ext.goertzel(in_data[:, 0::2], k)
    iq1 = dsp_ext.goertzel(in_data[:, 1::2], k)


def bench_goertzel(data_len, n_test=10000):
    rng = np.random.Generator(np.random.SFC64())
    in_data = rng.random((n_test, data_len), np.float64)

    cost = np.empty(len(BenchType))
    error = np.empty(len(BenchType))

    # protection against memory errors
    if data_len < 1024:
        x = np.arange(data_len)
        F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
        t0 = time()
        out_dft = in_data @ F
        cost[BenchType.dft.value] = time()-t0
    else:
        out_dft = np.nan
        cost[BenchType.dft.value] = np.nan

    t0 = time()
    out_fft = np.fft.fft(in_data)
    cost[BenchType.fft.value] = time()-t0
    error[BenchType.dft.value] = (out_fft-out_dft).std(axis=-1).max()
    error[BenchType.fft.value] = np.nan

    out_goertzel = np.empty_like(out_fft)
    cost_goertzel = 0
    for k in range(data_len):
        t0 = time()
        out_k = dsp_ext.goertzel(in_data, k)
        cost_goertzel += time()-t0
        out_goertzel[:, k] = out_k

    cost[BenchType.goertzel.value] = cost_goertzel/data_len
    error[BenchType.goertzel.value] = (out_goertzel-out_fft).std(axis=-1).max()

    return cost, error


def bench_range(len_range, n_test=10000):
    cost = np.empty((len(BenchType), len(len_range)))
    error = np.empty((len(BenchType), len(len_range)))
    for i, data_len in enumerate(len_range):
        print("data_len %d" % data_len)
        cost[:, i], error[:, i] = bench_goertzel(data_len, n_test=n_test)

    return cost, error


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("dummy exit")
        sys.exit()
    cost, _ = bench_goertzel(16)
    print(", ".join(["%s %f" % (k.name, cost[k.value]) for k in BenchType]))

    len_range = np.concatenate((
        np.arange(1, 64), np.arange(64, 1024, 64),
        2**np.arange(10, 15)))
    cost, error = bench_range(len_range, n_test=10)

    from matplotlib import pyplot as plt
    plt.figure(1)
    for k in BenchType:
        plt.plot(len_range, 10*np.log10(cost[k.value, :]), label=k.name)
    plt.legend()
    plt.ylabel("time (dBs)")
    plt.xlabel("length (samples)")

    plt.figure(2)
    for k in BenchType:
        if np.isnan(error[k.value, :]).all():
            continue
        plt.plot(len_range, error[k.value, :], label="%s vs fft" % k.name)
    plt.legend()
    plt.ylabel("error")
    plt.xlabel("length (samples)")
    plt.show()


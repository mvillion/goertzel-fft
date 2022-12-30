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
    goertzel_rad2_py = 3
    goertzel_rad2 = 4
    goertzel_rad2_sse = 5


def goertzel_rad2_py(data, k):
    # if data_len = 9, data_len0 = 5 and data_len1 = 4
    # for first part (i), k = 10/9 and internally k = 10/9*(0:5)/5
    # k = (0:10:2)/9
    shape = data.shape
    data_len = shape[-1]
    data = data.reshape(-1, data_len)

    data0 = data[:, 0::2]
    data1 = data[:, 1::2]
    data_len0 = data0.shape[-1]
    data_len1 = data1.shape[-1]
    iq0 = dsp_ext.goertzel(data0, 2*k*data_len0/data_len)
    iq1 = dsp_ext.goertzel(data1, 2*k*data_len1/data_len)

    if data_len % 2 == 0:
        iq1 *= np.exp(-2j*k*np.pi/data_len)
    else:
        iq0 *= np.exp(-2j*k*np.pi/data_len)

    # # useful for debug:
    # if k == 1:
    #     x = np.arange(data_len)
    #     F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
    #     iq0_debug = (data0 @ F[0::2, :])[:, k]
    #     iq1_debug = (data1 @ F[1::2, :])[:, k]
    #     print("stop")

    return (iq0+iq1).reshape(shape[:-1])


def bench_goertzel(data_len, n_test=10000):
    rng = np.random.Generator(np.random.SFC64())
    in_data = rng.random((n_test, data_len), np.float64)

    cost = np.empty(len(BenchType))
    error = np.empty(len(BenchType))

    # dft, protection against memory errors
    if data_len < 1024:
        x = np.arange(data_len)
        F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
        t0 = time()
        out = in_data @ F
        cost[BenchType.dft.value] = time()-t0
    else:
        out = np.nan
        cost[BenchType.dft.value] = np.nan

    # fft
    t0 = time()
    out_fft = np.fft.fft(in_data)
    cost[BenchType.fft.value] = time()-t0
    error[BenchType.dft.value] = (out-out_fft).std(axis=-1).max()
    error[BenchType.fft.value] = np.nan

    bench2fun = {
        BenchType.goertzel: dsp_ext.goertzel,
        BenchType.goertzel_rad2_py: goertzel_rad2_py,
        BenchType.goertzel_rad2: dsp_ext.goertzel_rad2,
        BenchType.goertzel_rad2_sse: dsp_ext.goertzel_rad2_sse,
    }
    for etype, fun in bench2fun.items():
        out = np.empty_like(out_fft)
        cost_goertzel = 0
        for k in range(data_len):
            t0 = time()
            out_k = fun(in_data, k)
            cost_goertzel += time()-t0
            out[:, k] = out_k
        cost[etype.value] = cost_goertzel/data_len
        error[etype.value] = (out-out_fft).std(axis=-1).max()

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
    cost, _ = bench_goertzel(13, n_test=10)
    print(", ".join(["%s %f" % (k.name, cost[k.value]) for k in BenchType]))

    len_range = np.concatenate((
        np.arange(1, 64), np.arange(64, 1024, 64),
        2**np.arange(10, 13)))
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

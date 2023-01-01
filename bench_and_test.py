#!/usr/bin/python3
import numpy as np
import sys

from enum import Enum
from gofft_directory import dsp_ext
from time import time

bench_list = [
    "dft",
    "fft",
    "goertzel",
    # # "goertzel_rad2_py",
    # "goertzel_rad2",
    # "goertzel_rad2_sse",
    # # "goertzel_rad4_py",
    # "goertzel_rad4",
    # "goertzel_rad4_avx",
    # "goertzel_rad8_py",
    # "goertzel_rad8_avx",
    # "goertzel_rad12_avx",
    "goertzel_dft",
    "goertzel_dft_rad2",
    "goertzel_dft_rad2_sse",
]
BenchType = Enum("BenchType", bench_list, start=0)


def goertzel_rad2_py(data, k):
    # if data_len = 9, data_len0 = 5 and data_len1 = 4
    # for 1sr part, k = 10/9 and internally k = 10/9*(0:5)/5
    # k = (0:10:2)/9 = [0, 2, 4, 6, 8]/9
    # for 2nd part, k = 8/9 and internally k = 8/9*(0:4)/4
    # k = (0:8:2)/9 = [1, 3, 5, 7]/9 (with an additional rotation)
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
        # this is obvious: for even lengths, odd values need an additional rot
        iq1 *= np.exp(-2j*k*np.pi/data_len)
    else:
        # this is less obvious: for odd lengths
        # both even and odd are probably already rotated...?
        iq0 *= np.exp(-2j*k*np.pi/data_len)

    # # useful for debug:
    # if k == 1:
    #     x = np.arange(data_len)
    #     F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
    #     iq0_debug = (data0 @ F[0::2, :])[:, k]
    #     iq1_debug = (data1 @ F[1::2, :])[:, k]
    #     print("stop")

    return (iq0+iq1).reshape(shape[:-1])


def goertzel_radix_py(data, k, radix=4):
    shape = data.shape
    data_len = shape[-1]
    data = data.reshape(-1, data_len)

    data_len_r = (data_len+radix-1)//radix*radix
    n_pad = data_len_r-data_len
    if 0 < n_pad:
        data = np.column_stack(
            (data, np.zeros_like(data, shape=(data.shape[0], n_pad))))

    iq = [None]*radix
    for m in range(radix):
        data_m = data[:, m::radix]
        iq[m] = dsp_ext.goertzel(data_m, k*data_len_r/data_len)

    # perform complex goertzel
    iq_out = dsp_ext.goertzel(np.array(iq).T, k*radix/data_len)
    iq_out *= np.exp(-2j*radix*k*np.pi/data_len)

    # for m in range(1, radix):
    #     iq[m] *= np.exp(-2j*k*m*np.pi/data_len)

    # # # useful for debug:
    # # if k == 1:
    # #     x = np.arange(data_len)
    # #     F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
    # #     iq_debug = [None]*radix
    # #     for m in range(radix):
    # #         F_r = F[m:data_len:radix, :]
    # #         iq_debug[m] = (data[:, m:data_len:radix] @ F_r)[:, k]
    # #         ang = np.angle((iq[m]/iq_debug[m]).mean())
    # #         num = data_len*ang/(2*np.pi)
    # #         print("k: %f, m: %d, %f/%d" % (k, m, num, data_len))
    # #     print("stop")

    # iq = np.sum(iq, axis=0)

    # # ang = np.angle((iq_out/iq).mean())
    # # num = data_len*ang/(2*np.pi)
    # # print("k: %f, %f/%d" % (k, num, data_len))
    # # if np.std(iq-iq2) > 1e-8:
    # #     print("oops")

    if 0 < n_pad:
        iq_out *= np.exp(-2j*k*n_pad*np.pi/data_len)

    return iq_out.reshape(shape[:-1])


def goertzel_rad8_py(data, k):
    return goertzel_radix_py(data, k, radix=8)


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
        "goertzel_dft": dsp_ext.goertzel_dft,
        "goertzel_dft_rad2": dsp_ext.goertzel_dft_rad2,
        "goertzel_dft_rad2_sse": dsp_ext.goertzel_dft_rad2_sse,
    }
    for type_str, fun in bench2fun.items():
        try:
            etype = BenchType[type_str]
        except KeyError:
            continue
        t0 = time()
        out = fun(in_data, np.nan)
        cost[etype.value] = time()-t0
        error[etype.value] = (out-out_fft).std(axis=-1).max()

    bench2fun = {
        "goertzel": dsp_ext.goertzel,
        "goertzel_rad2_py": goertzel_rad2_py,
        "goertzel_rad2": dsp_ext.goertzel_rad2,
        "goertzel_rad2_sse": dsp_ext.goertzel_rad2_sse,
        "goertzel_rad4_py": goertzel_radix_py,
        "goertzel_rad4": dsp_ext.goertzel_rad4,
        "goertzel_rad4_avx": dsp_ext.goertzel_rad4_avx,
        "goertzel_rad8_py": goertzel_rad8_py,
        "goertzel_rad8_avx": dsp_ext.goertzel_rad8_avx,
        "goertzel_rad12_avx": dsp_ext.goertzel_rad12_avx,
    }
    for type_str, fun in bench2fun.items():
        try:
            etype = BenchType[type_str]
        except KeyError:
            continue
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

    len_range = np.arange(25, 26)
    cost, error = bench_range(len_range, n_test=2)
    for m, data_len in enumerate(len_range):
        cost_str = ["%s %f" % (k.name, error[k.value, m]) for k in BenchType]
        print(", ".join(cost_str))
    # sys.exit()

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

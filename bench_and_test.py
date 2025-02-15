#!/usr/bin/python3
import numpy as np
import sys

from cpuinfo import get_cpu_info
from enum import Enum
from gofft_directory import dsp_ext
from pathlib import Path
from pyfftw.interfaces import numpy_fft
from time import time


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


def bench_goertzel(BenchType, data_len, cx=False, n_test=10000):
    rng = np.random.Generator(np.random.SFC64())
    dtype = np.float32
    in_data = rng.random((n_test, data_len), dtype)
    if cx:
        in_data = in_data+1j*rng.random((n_test, data_len), dtype)
    in_data = in_data.astype(np.float64)

    cost = np.empty(len(BenchType))
    error = np.empty(len(BenchType))

    # fft
    for name in ["fft", "fftf"]:
        try:
            etype = BenchType[name]
        except KeyError:
            continue
        is_f32 = name in ["fftf"]
        if is_f32:
            in_data2 = in_data.astype(np.float32)
            t0 = time()
            out = numpy_fft.fft(in_data2)
            cost[etype.value] = time()-t0
        else:
            t0 = time()
            out = np.fft.fft(in_data)
            cost[etype.value] = time()-t0
            out_fft = out

        if is_f32:
            error[etype.value] = (out-out_fft).std(axis=-1).max()
        else:
            error[etype.value] = np.nan

    # dft, protection against memory errors
    try:
        if data_len < 1024:
            x = np.arange(data_len)
            F = np.exp(-2j*np.pi/data_len*np.outer(x, x))
            t0 = time()
            out = in_data @ F
            cost[BenchType.dft.value] = time()-t0
        else:
            out = np.nan
            cost[BenchType.dft.value] = np.nan
        error[BenchType.dft.value] = (out-out_fft).std(axis=-1).max()
    except AttributeError:
        pass

    for etype in BenchType:
        if "_dft" not in etype.name:
            continue
        try:
            fun = getattr(dsp_ext, etype.name)
        except AttributeError:
            continue
        t0 = time()
        out = fun(in_data, np.nan)
        cost[etype.value] = time()-t0
        error[etype.value] = (out-out_fft).std(axis=-1).max()

    for etype in BenchType:
        name = etype.name
        if "_dft" in name or name in ["dft", "fft", "fftf"]:
            continue
        is_f32 = "goertzelf" in name
        if is_f32:
            name = name.replace("goertzelf", "goertzel")
            in_data2 = in_data.astype(np.float32)
        else:
            in_data2 = in_data
        try:
            fun = getattr(dsp_ext, name)
        except AttributeError:
            fun = globals()[name]
            # try:
            # except KeyError:
            #     continue
        out = np.empty_like(out_fft)
        cost_goertzel = 0
        for k in range(data_len):
            t0 = time()
            out_k = fun(in_data2, k)
            cost_goertzel += time()-t0
            out[:, k] = out_k
        cost[etype.value] = cost_goertzel/data_len
        error[etype.value] = (out-out_fft).std(axis=-1).max()

    return cost, error


def bench_range(BenchType, len_range, cx=False, n_test=10000):
    cost = np.empty((len(BenchType), len(len_range)))
    error = np.empty((len(BenchType), len(len_range)))
    for i, data_len in enumerate(len_range):
        print("data_len %d" % data_len)
        cost[:, i], error[:, i] = bench_goertzel(
            BenchType, data_len, cx=cx, n_test=n_test)

    return cost, error


def plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, plot_prefix,
        title_str):

    prefix = "/".join([str(media_path), plot_prefix])
    from matplotlib import pyplot as plt
    plt.close("all")
    fig_size = (12, 9)
    plt.figure(1, figsize=fig_size)
    for name in bench_list:
        k = BenchType[name]
        plt.plot(len_range, 10*np.log10(cost[k.value, :]), label=k.name)
    plt.legend()
    plt.ylabel("time (dBs)")
    plt.xlabel("length (samples)")
    plt.title(title_str)
    plt.savefig("%s_cost_db.png" % prefix, bbox_inches="tight")

    plt.figure(2, figsize=fig_size)
    for name in bench_list:
        k = BenchType[name]
        plt.plot(len_range, cost[k.value, :], label=k.name)
    plt.legend()
    plt.ylabel("time (s)")
    plt.xlabel("length (samples)")
    plt.title(title_str)
    plt.savefig("%s_cost.png" % prefix, bbox_inches="tight")

    plt.figure(3, figsize=fig_size)
    for name in bench_list:
        k = BenchType[name]
        if np.isnan(error[k.value, :]).all():
            plt.plot(np.arange(1), np.arange(1), label="%s vs fft" % k.name)
            continue
        error_k = error[k.value, :]
        error_k = 10*np.log10(error_k)
        plt.plot(len_range, error_k, label="%s vs fft" % k.name)
    plt.legend()
    plt.ylabel("error")
    plt.xlabel("length (samples)")
    plt.title(title_str)
    plt.savefig("%s_error.png" % prefix, bbox_inches="tight")
    # plt.show()
    plt.pause(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("dummy exit")
        sys.exit()

    cpu_flags = get_cpu_info()["flags"]

    # short test for debug
    bench_list = [
        # "dft",
        "fft",
        "goertzelf_rad8_avx",
        # "goertzel_rad4_avx",
        # "goertzel_rad4x2_test",
        # "goertzel_rad8_avx",
        # "goertzel_dft",
        # "goertzel_rad2_dft",
        "goertzel_rad2_sse_dft",
        "goertzel_rad4_avx_dft",
    ]
    BenchType = Enum("BenchType", bench_list, start=0)
    len_range = np.arange(1024, 1026)
    cost, error = bench_range(BenchType, len_range, n_test=2)
    for m, data_len in enumerate(len_range):
        cost_str = ["%s %f" % (k.name, error[k.value, m]) for k in BenchType]
        print(", ".join(cost_str))
    # sys.exit()

    media_path = Path(".") / "media_tmp"
    media_path.mkdir(exist_ok=True)
    n_test = 100

    len_range = np.concatenate((
        np.arange(1, 64), np.arange(64, 1024, 64),
        2**np.arange(10, 13)))

    # real-value tests---------------------------------------------------------
    bench_list = [
        "dft",
        "fft",
        "fftf",
        "goertzel",
        "goertzel_rad2_py",
        "goertzel_radix_py",
        "goertzel_rad8_py",
        "goertzel_rad2",
        "goertzel_rad2_sse",
        "goertzel_rad4",
        "goertzel_rad4_avx",
        "goertzel_rad8_avx",
        "goertzel_rad12_avx",
        "goertzel_rad16_avx",
        "goertzel_rad20_avx",
        "goertzel_rad24_avx",
        "goertzel_rad40_avx",
        "goertzel_rad4u2_avx",
        "goertzel_rad4u4_avx",
        "goertzel_rad4x2_test",
        "goertzelf",
        "goertzelf_rad2",
        "goertzelf_rad4",
        "goertzelf_rad8_avx",
        "goertzelf_rad16_avx",
        "goertzelf_rad24_avx",
        "goertzelf_rad40_avx",
        # "goertzel_dft",
        # "goertzel_rad2_dft",
        # "goertzel_rad2_sse_dft",
        # # "goertzel_rad4_avx_dft",
        # # "goertzel_rad8_avx_dft",
        # "goertzel_rad12_avx_dft",
        # "goertzel_rad16_avx_dft",
        # "goertzel_rad20_avx_dft",
        # # "goertzel_rad24_avx_dft",
        # "goertzel_rad40_avx_dft",
    ]
    if "fma" in cpu_flags:
        bench_list += [
            "goertzel_rad4_fma",
            "goertzel_rad8_fma",
            "goertzel_rad20_fma",
        ]
    BenchType = Enum("BenchType", bench_list, start=0)
    cost, error = bench_range(BenchType, len_range, n_test=n_test)

    # title_str = "Goertzel DFT vs FFT"
    # bench_list = [
    #     "fft",
    #     "goertzel",
    #     # "goertzel_rad2_dft",
    #     # "goertzel_rad2_sse_dft",
    #     # "goertzel_rad4_avx_dft",
    #     # "goertzel_rad8_avx_dft",
    #     "goertzel_rad12_avx_dft",
    #     "goertzel_rad16_avx_dft",
    #     "goertzel_rad20_avx_dft",
    #     # "goertzel_rad24_avx_dft",
    #     "goertzel_rad40_avx_dft",
    # ]
    # plot_bench(
    #     BenchType, len_range, cost, error, bench_list, media_path, "dft",
    #     title_str)

    title_str = "archived"
    bench_list = [
        "fft", "goertzel_rad2_py", "goertzel_radix_py", "goertzel_rad8_py"]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "archive",
        title_str)

    title_str = "Goertzel vs DFT vs FFT"
    bench_list = ["dft", "fft", "goertzel"]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "intro",
        title_str)

    title_str = "Faster Goertzel w/ radix"
    bench_list = [
        "fft",
        "goertzel",
        "goertzel_rad2",
        "goertzel_rad2_sse",
        "goertzel_rad4",
        "goertzel_rad4_avx",
        "goertzel_rad8_avx",
    ]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "radix",
        title_str)

    title_str = "More radix"
    bench_list = [
        "fft",
        "goertzel",
        "goertzel_rad4_avx",
        "goertzel_rad8_avx",
        "goertzel_rad12_avx",
        "goertzel_rad16_avx",
        "goertzel_rad20_avx",
        "goertzel_rad24_avx",
        "goertzel_rad40_avx",
    ]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path,
        "more_radix", title_str)

    title_str = "Influence of unrolling"
    bench_list = [
        "goertzel",
        "goertzel_rad4_avx",
        "goertzel_rad4u2_avx",
        "goertzel_rad4u4_avx",
        "goertzel_rad8_avx",
        "goertzel_rad16_avx",
    ]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "unroll",
        title_str)

    if "fma" in cpu_flags:
        title_str = "AVX vs FMA3"
        bench_list = [
            "goertzel",
            "goertzel_rad4_avx",
            "goertzel_rad8_avx",
            "goertzel_rad20_avx",
            "goertzel_rad4_fma",
            "goertzel_rad8_fma",
            "goertzel_rad20_fma",
        ]
        plot_bench(
            BenchType, len_range, cost, error, bench_list, media_path, "fma3",
            title_str)

    # float tests--------------------------------------------------------------
    title_str = "float32 input"
    bench_list = [
        "fft",
        "fftf",
        "goertzel",
        "goertzelf",
        # "goertzel_rad2",
        # "goertzelf_rad2",
        # "goertzel_rad4",
        # "goertzelf_rad4",
        "goertzelf_rad8_avx",
        "goertzelf_rad16_avx",
        "goertzelf_rad24_avx",
        "goertzelf_rad40_avx",
    ]
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "f32",
        title_str)

    # complex-value tests------------------------------------------------------
    bench_list = [
        "fft",
        "goertzel",
        "goertzel_rad4_avx",
        "goertzel_rad8_avx",
        "goertzel_rad12_avx",
    ]
    BenchType = Enum("BenchType", bench_list, start=0)
    cost, error = bench_range(BenchType, len_range, cx=True, n_test=n_test)

    title_str = "complex input"
    plot_bench(
        BenchType, len_range, cost, error, bench_list, media_path, "cx",
        title_str)

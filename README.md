# Benchmark for Goertzel algorithm

[![Build Status](https://travis-ci.com/NaleRaphael/goertzel-fft.svg?branch=master)](https://travis-ci.com/NaleRaphael/goertzel-fft)
[![Binder](https://mybinder.org/badge_logo.svg)][launch_on_binder]

## Overview

To evaluate the power of specific frequency component in signal, `Goertzel algorithm` will be a better solution than `fast Fourier transform (FFT)`. Because `Goertzel algorithm` allows us to evaluate a single `DFT (Discrete Fourier Transform)` term at a time.

As Goertzel algorithm computes a single output frequency, it is faster than FFT.

## First problem: How much faster Goertzel is? How can it be made faster?

### Introductory results

Goertzel is indeed faster than FFT.

![Alt text](media/intro_cost_db.png?raw=true "Goertzel vs DFT vs FFT (cost)")

Cost is represented in measured time with a logarithmic scale.

### Radix-Goerztel to use SSE and AVX instructions

SSE can process 2 double values at a time but for a simple implementation, it is practical to split even and odd values of the Goertzel computation to enable parallel computation.

With AVX, the problem is the same but with 4 double values at a time.

A radix-4 not using AVX intrinsics (goertzel_rad4) is compiled as:

    /-> vmulsd %xmm4,%xmm0,%xmm9
    |   add    $0xc,%r13
    |   add    $0x60,%rax
    |   vmulsd %xmm3,%xmm0,%xmm8
    |   vmulsd %xmm2,%xmm0,%xmm7
    |   vmulsd %xmm1,%xmm0,%xmm6
    |   vsubsd %xmm5,%xmm9,%xmm9
    |   vaddsd -0x60(%rax),%xmm9,%xmm9
    |   vsubsd %xmm10,%xmm8,%xmm8
    |   vaddsd -0x58(%rax),%xmm8,%xmm8
    |   vsubsd %xmm11,%xmm7,%xmm7
    |   vaddsd -0x50(%rax),%xmm7,%xmm7
    |   vmulsd %xmm9,%xmm0,%xmm5
    |   vsubsd %xmm12,%xmm6,%xmm6
    |   vaddsd -0x48(%rax),%xmm6,%xmm6
    |   vsubsd %xmm4,%xmm5,%xmm5
    |   vmulsd %xmm8,%xmm0,%xmm4
    |   vaddsd -0x40(%rax),%xmm5,%xmm5
    |   vsubsd %xmm3,%xmm4,%xmm3
    |   vaddsd -0x38(%rax),%xmm3,%xmm10
    |   vmulsd %xmm7,%xmm0,%xmm3
    |   vmulsd %xmm5,%xmm0,%xmm4
    |   vsubsd %xmm2,%xmm3,%xmm2
    |   vaddsd -0x30(%rax),%xmm2,%xmm11
    |   vmulsd %xmm6,%xmm0,%xmm2
    |   vmulsd %xmm10,%xmm0,%xmm3
    |   vsubsd %xmm9,%xmm4,%xmm4
    |   vaddsd -0x20(%rax),%xmm4,%xmm4
    |   vsubsd %xmm1,%xmm2,%xmm1
    |   vaddsd -0x28(%rax),%xmm1,%xmm12
    |   vmulsd %xmm11,%xmm0,%xmm2
    |   vsubsd %xmm8,%xmm3,%xmm3
    |   vaddsd -0x18(%rax),%xmm3,%xmm3
    |   vmulsd %xmm12,%xmm0,%xmm1
    |   vsubsd %xmm7,%xmm2,%xmm2
    |   vaddsd -0x10(%rax),%xmm2,%xmm2
    |   vsubsd %xmm6,%xmm1,%xmm1
    |   vaddsd -0x8(%rax),%xmm1,%xmm1
    |   cmp    %rdx,%r13
    \-- jl     <goertzel_rad4+0x100>

These 40 instructions are using only "sd" instructions with single double operations.

Equivalent AVX code (goertzel_rad4_avx) is using only 12 instructions.

    /-> vmulpd %ymm5,%ymm0,%ymm2
    |   vsubpd %ymm1,%ymm2,%ymm2
    |   vaddpd (%rcx,%r15,8),%ymm2,%ymm2
    |   vmulpd %ymm2,%ymm5,%ymm1
    |   vsubpd %ymm0,%ymm1,%ymm1
    |   vaddpd 0x20(%rcx,%r15,8),%ymm1,%ymm1
    |   vmulpd %ymm1,%ymm5,%ymm0
    |   vsubpd %ymm2,%ymm0,%ymm0
    |   vaddpd 0x40(%rcx,%r15,8),%ymm0,%ymm0
    |   add    $0xc,%r15
    |   cmp    %rax,%r15
    \-- jl     <goertzel_rad4_avx+0xe0>

Registers used are 0, 1, 2, 5; 4 out of 16. Many registers are available for e.g. processing other frequencies.
With a number of instructions reduced by a factor of almost 4, a substantial increase in performance could be expected.
Figures astonishingly prove the contrary:

![Alt text](media/radix_cost_db.png?raw=true "Goertzel non-radix vs radix (cost)")

Efforts for code optimization do not seem to be fruitful using AVX. Explanation will be given in the next section.

Radix-8 is possible because of the reduced number of registers used so in the end AVX optimisation gives a marginal gain.

Higher order radix are faster and also give better numerical precision.

![Alt text](media/radix_error.png?raw=true "Goertzel vs DFT vs FFT (error)")

### Is radix-8 faster because of loop-unrolling?

Core loop of radix-8 (goertzel_rad8_avx) is:

    /-> vmulpd %ymm3,%ymm1,%ymm2
    |   add    $0x18,%rbx
    |   add    $0xc0,%rax
    |   vmulpd %ymm3,%ymm0,%ymm8
    |   vsubpd %ymm6,%ymm2,%ymm2
    |   vaddpd -0xc0(%rax),%ymm2,%ymm2
    |   vsubpd %ymm7,%ymm8,%ymm7
    |   vaddpd -0xa0(%rax),%ymm7,%ymm8
    |   vmulpd %ymm2,%ymm3,%ymm6
    |   vsubpd %ymm1,%ymm6,%ymm1
    |   vaddpd -0x80(%rax),%ymm1,%ymm6
    |   vmulpd %ymm8,%ymm3,%ymm1
    |   vsubpd %ymm0,%ymm1,%ymm0
    |   vaddpd -0x60(%rax),%ymm0,%ymm7
    |   vmulpd %ymm6,%ymm3,%ymm1
    |   vmulpd %ymm7,%ymm3,%ymm0
    |   vsubpd %ymm2,%ymm1,%ymm1
    |   vmovupd -0x20(%rax),%ymm2
    |   vaddpd -0x40(%rax),%ymm1,%ymm1
    |   vsubpd %ymm8,%ymm0,%ymm0
    |   vaddpd %ymm2,%ymm0,%ymm0
    |   cmp    %rdx,%rbx
    \-- jl     <goertzel_rad8_avx+0x110>

12 operations from radix-4 are now 23 operations.

With a radix-4 unrolled 2 times (goertzel_rad4u2_avx), the code is:

    /-> vmulpd %ymm0,%ymm2,%ymm3
    |   add    $0x18,%r15
    |   vmovupd 0xa0(%rax),%ymm6
    |   add    $0xc0,%rax
    |   vsubpd %ymm1,%ymm3,%ymm3
    |   vaddpd -0xc0(%rax),%ymm3,%ymm3
    |   vmulpd %ymm3,%ymm2,%ymm1
    |   vsubpd %ymm0,%ymm1,%ymm1
    |   vaddpd -0xa0(%rax),%ymm1,%ymm1
    |   vmulpd %ymm1,%ymm2,%ymm0
    |   vsubpd %ymm3,%ymm0,%ymm0
    |   vaddpd -0x80(%rax),%ymm0,%ymm0
    |   vmulpd %ymm0,%ymm2,%ymm3
    |   vsubpd %ymm1,%ymm3,%ymm3
    |   vaddpd -0x60(%rax),%ymm3,%ymm3
    |   vmulpd %ymm3,%ymm2,%ymm1
    |   vsubpd %ymm0,%ymm1,%ymm1
    |   vaddpd -0x40(%rax),%ymm1,%ymm1
    |   vmulpd %ymm1,%ymm2,%ymm0
    |   vsubpd %ymm3,%ymm0,%ymm0
    |   vaddpd %ymm6,%ymm0,%ymm0
    |   cmp    %rdx,%r15
    \-- jl     <goertzel_rad4u2_avx+0xf0>

The code looks indentical but using 4 registers instead of 7.

For a slow processor:

![Alt text](media/unroll_cost_db.png?raw=true "Influence of loop unrolling (cost)")

and a faster one:

![Alt text](media/2950x/unroll_cost_db.png?raw=true "Influence of loop unrolling (cost) AMD 2950x")

Loop unrolling has no influence. Branch prediction is likely efficient and correctly predicts that this core loop is ... looping.

Radix-8 is still faster with identical operations!
PC processors are using out-of-order execution and as radix-8 instructions from the two sets of avx registers are not dependent on each other, many operations can be executed in a single cycle. This also explains why non-AVX code could execute as fast as AVX code.

### More radix: longer radix
On a slow AMD processor, higher order radix are faster up to 8.
radix-12 is not faster than 8.

![Alt text](media/more_radix_cost_db.png?raw=true "Longer radix (cost)")

As a register is used for the 2 x cos factor and 3 registers are needed for 4 values radix-20 is the limit (1+3x5) to avoid using stack memory access.
This can be confirmed on a fast processor:

![Alt text](media/2950x/more_radix_cost_db.png?raw=true "Longer radix (cost) AMD 2950x")

### Use of FMA3 instructions

The core loop using FMA3 instructions is shorter:

    /-> vfmsub231pd %ymm3,%ymm1,%ymm0
    |   vaddpd (%rcx,%r13,8),%ymm0,%ymm2
    |   vmovupd 0x20(%rcx,%r13,8),%ymm0
    |   vfmsub231pd %ymm2,%ymm3,%ymm1
    |   vaddpd %ymm1,%ymm0,%ymm0
    |   vfmsub231pd %ymm0,%ymm3,%ymm2
    |   vaddpd 0x40(%rcx,%r13,8),%ymm2,%ymm1
    |   add    $0xc,%r13
    |   cmp    %rax,%r13
    \-- jl     <goertzel_rad4_fma+0xe0>

AVX used 12 instructions, FMA2 needs only 10 (could be 9?).

![Alt text](media/2950x/fma3_cost_db.png?raw=true "AVX vs FMA3 (cost) AMD 2950x")

FMA loops are a little faster for shorter lengths.

## Second problem: If Goertzel is used to compute all frequencies, how much slower Goertzel is?
This problem does not fully make sense.
It is an exercise to process more than one frequency.


## Environments
### Machine
* OS: Ubuntu 22.04
* CPU: AMD A4-5000 APU with Radeon(TM) HD Graphics
* RAM: 16.00 GB Dual-Channel DDR3 @ 667 MT/s

### Version of Python and packages
* Python: Python 3.10.7
* Numpy: 1.23.4

## Installation

* To uninstall this package:

  ```bash
  $ ./setup build_ext --inplace --cpu-baseline="avx"
  ```

## Usage
* Evaluate the power of a single DFT term by `goertzel`

  ```python
  import gofft
  import numpy as np

  fs = 1000   # sampling frequency
  ft = 60     # target frequency to be evaluated (60 Hz)
  dur = 2     # duration of signal
  num = fs*dur  # sampling points
  t = np.linspace(0, dur, num)  # time series
  data = np.sin(2*np.pi*ft*t)   # signal to be evaluated (60 Hz)

  mag = gofft.alg.goertzel(data, fs, ft, fs)
  print(mag)  # 0.4969141358692001
  ```

## Implemented algorithms

1. `gofft.alg.goertzel`: Normal Goertzel algorithm.
2. `gofft.alg.goertzel_cx`: Goertzel with complex input algorithm.
3. `gofft.alg.goertzel_rad2`: Goertzel radix-2 algorithm.
4. `gofft.alg.goertzel_rad2_sse`: Goertzel radix-2 algorithm using SSE instructions.
5. `gofft.alg.goertzel_rad4`: Goertzel radix-4 algorithm.
6. `gofft.alg.goertzel_rad4_avx`: Goertzel radix-4 algorithm using AVX instructions.
7. `gofft.alg.goertzel_rad8_avx`: Goertzel radix-8 algorithm using AVX instructions.
8. `gofft.alg.goertzel_rad12_avx`: Goertzel radix-12 algorithm using AVX instructions.
9. `gofft.alg.goertzel_dft`: Goertzel algorithm to compute dft.
10. `gofft.alg.goertzel_dft`: Goertzel radix-2 algorithm to compute dft.
11. `gofft.alg.goertzel_dft_rad2_sse`: Goertzel radix-2 algorithm using SSE instructions to compute dft.

## Algorithm verification and benchmark

* Run all benchmark cases and plot result

  ```bash
  $ python3 bench_and_test.py
  ```

## Reference
[wikipedia - Goertzel](https://en.wikipedia.org/wiki/Goertzel_algorithm)
[stackoverflow - Implementation of Goertzel algorithm in C](http://stackoverflow.com/questions/11579367)

[STFT]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
[launch_on_binder]: https://mybinder.org/v2/gh/NaleRaphael/goertzel-fft/master?filepath=doc%2Fipynb%2Fdemo_simple_example.ipynb

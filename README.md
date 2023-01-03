# Benchmark for Goertzel algorithm

[![Build Status](https://travis-ci.com/NaleRaphael/goertzel-fft.svg?branch=master)](https://travis-ci.com/NaleRaphael/goertzel-fft)
[![Binder](https://mybinder.org/badge_logo.svg)][launch_on_binder]

## Overview

To evaluate the power of specific frequency component in signal, `Goertzel algorithm` will be a better solution than `fast Fourier transform (FFT)`. Because `Goertzel algorithm` allows us to evaluate a single `DFT (Discrete Fourier Transform)` term at a time.

As Goertzel algorithm computes a single output frequency, it is faster than FFT.

First problem:
How much master Goertzel is? How can it be made faster?

Second problem:
If Goertzel is used to compute all frequencies, how much slower Goertzel is?
This problem does not fully make sense.


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

## Performance

* Data type: float64 (fig_02 is a partial view of fig_01)

  ![Fig 01. Result of test (error)][dtype_float64_error]

  ![Fig 02. Result of benchmark (csot)][dtype_float64_cost]


## Reference
[wikipedia - Goertzel](https://en.wikipedia.org/wiki/Goertzel_algorithm)
[stackoverflow - Implementation of Goertzel algorithm in C](http://stackoverflow.com/questions/11579367)

[dtype_float64_error]: https://imgur.com/a/zbqqTzH
[dtype_float64_cost]: https://imgur.com/a/p9J1xNW

[STFT]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
[launch_on_binder]: https://mybinder.org/v2/gh/NaleRaphael/goertzel-fft/master?filepath=doc%2Fipynb%2Fdemo_simple_example.ipynb

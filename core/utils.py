import numpy as np
import math

def bit_reverse(x, width):
    b = '{:0{width}b}'.format(x, width=width)
    return int(b[::-1], 2)

def bit_reversal(x: np.ndarray) -> np.ndarray:
    N = len(x)

    # Bit width
    width = int(math.log2(N))
    for i in range(N):
        reverse_i = bit_reverse(i, width)
        if i < reverse_i:
            x[i], x[reverse_i] = x[reverse_i], x[i]

    return x

def pad_with_zeros(x: np.ndarray) -> np.ndarray:
    N = len(x)

    pad_len = (2 ** math.floor(math.log2(N) + 1)) - N
    if pad_len != N:
        x = np.concatenate((x, np.zeros(pad_len)))

    return x

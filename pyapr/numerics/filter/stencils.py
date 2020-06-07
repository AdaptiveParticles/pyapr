import numpy as np


def get_gaussian_stencil(size, sigma, normalize=False):

    x = np.arange(-size//2 + 1, size//2 + 1)

    vals = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    if normalize:
        vals = vals / vals.sum()

    stenc = np.empty((size, size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            for k in range(size):
                stenc[i, j, k] = vals[i] * vals[j] * vals[k]

    return stenc

import numpy as np


def get_gaussian_stencil(size, sigma, ndims=3, normalize=False):
    """Naively generate a Gaussian stencil."""
    x = np.arange(-(size//2), size//2 + 1)

    vals = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    if normalize:
        vals = vals / vals.sum()

    stenc = np.empty((size,)*ndims, dtype=np.float32)

    if ndims == 3:
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    stenc[i, j, k] = vals[i] * vals[j] * vals[k]
    elif ndims == 2:
        for i in range(size):
            for j in range(size):
                stenc[i, j] = vals[i] * vals[j]

        stenc = np.expand_dims(stenc, axis=0)

    return stenc

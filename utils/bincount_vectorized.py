import numpy as np


def bin_count_2d_vectorized(a):
    n = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:, None] * n
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*n).reshape(-1, n)

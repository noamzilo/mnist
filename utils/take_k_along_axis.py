import numpy as np


def take_largest_indices_along_axis(ar, n, axis):
    s = ar.ndim*[slice(None, None, None)]
    s[axis] = slice(-n, None, None)
    idx = np.argpartition(ar, kth=-n, axis=axis)[tuple(s)]
    sidx = np.take_along_axis(ar, idx, axis=axis).argsort(axis=axis)
    return np.flip(np.take_along_axis(idx, sidx, axis=axis), axis=axis)


def take_smallest_indices_along_axis(ar, n, axis):
    s = ar.ndim*[slice(None, None, None)]
    s[axis] = slice(None, n, None)
    idx = np.argpartition(ar, kth=n, axis=axis)[tuple(s)]
    sidx = np.take_along_axis(ar, idx, axis=axis).argsort(axis=axis)
    return np.take_along_axis(idx, sidx, axis=axis)


def take_largest_along_axis(ar, n, axis):
    idx = take_largest_indices_along_axis(ar, n, axis)
    return np.take_along_axis(ar, idx, axis=axis)


def take_smallest_along_axis(ar, n, axis):
    idx = take_smallest_indices_along_axis(ar, n, axis)
    return np.take_along_axis(ar, idx, axis=axis)

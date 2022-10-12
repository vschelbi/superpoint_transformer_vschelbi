import torch
import numpy as np
from numba import njit


__all__ = ['numba_randperm']


def torch_to_numba(func):
    """Decorator intended for numba functions to be fed and return
    torch.Tensor arguments.

    :param func:
    :return:
    """

    def numbafy(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def torchify(x):
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    def wrapper_torch_to_numba(*args, **kwargs):
        args_numba = [numbafy(x) for x in args]
        kwargs_numba = {k: numbafy(v) for k, v in kwargs.items()}
        out = func(*args_numba, **kwargs_numba)
        if isinstance(out, list):
            out = [torchify(x) for x in out]
        elif isinstance(out, tuple):
            out = tuple([torchify(x) for x in list(out)])
        elif isinstance(out, dict):
            out = {k: torchify(v) for k, v in out.items()}
        else:
            out = torchify(out)
        return out

    return wrapper_torch_to_numba

@torch_to_numba
@njit(cache=True, nogil=True)
def numba_randperm(n):
    """Same as torch.randperm but leveraing numba on CPU.

    NB: slightly faster than `np.random.permutation(np.arange(n))`
    """
    a = np.arange(n)
    np.random.shuffle(a)
    return a

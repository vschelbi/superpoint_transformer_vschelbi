import torch
import numpy as np


def tensor_idx(idx):
    """Convert an int, slice, list or numpy index to a torch.LongTensor."""
    if idx is None:
        idx = torch.LongTensor([])
    elif isinstance(idx, int):
        idx = torch.LongTensor([idx])
    elif isinstance(idx, list):
        idx = torch.LongTensor(idx)
    elif isinstance(idx, slice):
        idx = torch.arange(idx.stop)[idx]
    elif isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx)
    # elif not isinstance(idx, torch.LongTensor):
    #     raise NotImplementedError
    if isinstance(idx, torch.BoolTensor):
        idx = torch.where(idx)[0]
    assert idx.dtype is torch.int64, \
        "Expected LongTensor but got {idx.type} instead."
    # assert idx.shape[0] > 0, \
    #     "Expected non-empty indices. At least one index must be provided."
    return idx


def is_sorted(a: torch.LongTensor, increasing=True, strict=False):
    """Checks whether a 1D tensor of indices is sorted."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Floating point tensors are not supported"
    if increasing and strict:
        f = torch.gt
    if increasing and not strict:
        f = torch.ge
    if not increasing and strict:
        f = torch.lt
    if not increasing and not strict:
        f = torch.le
    return f(a[1:], a[:-1]).all()


def has_duplicates(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices contains duplicates."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Floating point tensors are not supported"
    return a.unique().numel() != a.numel()


def is_dense(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices contains dense indices.
    That is to say all values in [a.min(), a.max] appear at least once
    in a.
    """
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Floating point tensors are not supported"
    unique = a.unique()
    return unique[0] == 0 and unique[-1] == a.max()


def is_permutation(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices is a permutation."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Floating point tensors are not supported"
    return a.sort().values.equal(torch.arange(a.numel()))


def arange_interleave(sizes):
    """Vectorized equivalent of:
        ```torch.cat([torch.arange(x) for x in sizes])```
    """
    assert sizes.dim() == 1, 'Only supports 1D tensors'
    assert isinstance(sizes, torch.LongTensor), 'Only supports LongTensors'
    assert sizes.ge(0).all(), 'Only supports positive integers'

    a = torch.cat((torch.LongTensor([0]), sizes[:-1]))
    b = torch.cumsum(a, 0).long()
    return torch.arange(sizes.sum()) - torch.repeat_interleave(b, sizes)

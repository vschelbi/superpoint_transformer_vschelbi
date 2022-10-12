import torch
import numpy as np


__all__ = [
    'tensor_idx', 'is_sorted', 'has_duplicates', 'is_dense', 'is_permutation',
    'arange_interleave', 'print_tensor_info', 'numpyfy']


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
    assert not a.is_floating_point(), "Float tensors are not supported"
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
    assert not a.is_floating_point(), "Float tensors are not supported"
    return a.unique().numel() != a.numel()


def is_dense(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices contains dense indices.
    That is to say all values in [a.min(), a.max] appear at least once
    in a.
    """
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    unique = a.unique()
    return a.min() == 0 and unique.size(0) == a.max() + 1


def is_permutation(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices is a permutation."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    return a.sort().values.equal(torch.arange(a.numel()))


def arange_interleave(width, start=None):
    """Vectorized equivalent of:
        >>> torch.cat([torch.arange(s, s + w) for w, s in zip(width, start)])
    """
    assert width.dim() == 1, 'Only supports 1D tensors'
    assert isinstance(width, torch.Tensor), 'Only supports Tensors'
    assert not width.is_floating_point(), 'Only supports Tensors of integers'
    assert width.ge(0).all(), 'Only supports positive integers'
    start = start if start is not None else torch.zeros_like(width)
    assert width.shape == start.shape
    assert start.dim() == 1, 'Only supports 1D tensors'
    assert isinstance(start, torch.Tensor), 'Only supports Tensors'
    assert not start.is_floating_point(), 'Only supports Tensors of integers'
    width = width.long()
    start = start.long()
    device = width.device
    a = torch.cat((torch.zeros(1, device=device).long(), width[:-1]))
    offsets = (start - a.cumsum(0)).repeat_interleave(width)
    return torch.arange(width.sum(), device=device) + offsets

def print_tensor_info(a, name):
    """Print some info about a tensor. Used for debugging.
    """
    is_1d = a.dim() == 1
    is_int = not a.is_floating_point()

    msg = f'{name}:'
    msg += f'  shape={a.shape}'
    msg += f'  dtype={a.dtype}'
    msg += f'  min={a.min()}'
    msg += f'  max={a.max()}'

    if is_1d and is_int:
        msg += f'  duplicates={has_duplicates(a)}'
        msg += f'  sorted={is_sorted(a)}'
        msg += f'  dense={is_dense(a)}'
        msg += f'  permutation={is_permutation(a)}'

    print(msg)


def numpyfy(a, x32=False):
    """Convert torch.Tensor to numpy while respecting some constraints
    on output dtype.
    """
    assert isinstance(a, torch.Tensor)

    if x32 and a.dtype == torch.double:
        a = a.float()
    elif x32 and a.dtype == torch.long:
        assert a.abs().max() < torch.iinfo(torch.int).max, \
            "Can't convert int64 tensor to int32, largest value is to high"
        a = a.int()
    return a.numpy()

import torch
from superpoint_transformer.utils.tensor import is_dense, is_sorted, fast_repeat


__all__ = ['indices_to_pointers', 'dense_to_csr', 'csr_to_dense']


def indices_to_pointers(indices: torch.LongTensor):
    """Convert pre-sorted dense indices to CSR format."""
    device = indices.device
    assert len(indices.shape) == 1, "Only 1D indices are accepted."
    assert indices.shape[0] >= 1, "At least one group index is required."
    assert is_dense(indices), "Indices must be dense"

    # Sort indices if need be
    order = torch.arange(indices.shape[0], device=device)
    if not is_sorted(indices):
        indices, order = indices.sort()

    # Convert sorted indices to pointers
    pointers = torch.cat([
        torch.LongTensor([0]).to(device),
        torch.where(indices[1:] > indices[:-1])[0] + 1,
        torch.LongTensor([indices.shape[0]]).to(device)])

    return pointers, order


def dense_to_csr(a):
    """Convert a dense matrix to its CSR counterpart."""
    assert a.dim() == 2
    index = a.nonzero(as_tuple=True)
    values = a[index]
    columns = index[1]
    pointers = indices_to_pointers(index[0])[0]
    return pointers, columns, values


def csr_to_dense(pointers, columns, values, shape=None):
    """Convert a CSR matrix to its dense counterpart of a given shape.
    """
    assert pointers.dim() == 1
    assert columns.dim() == 1
    assert values.dim() == 1
    assert shape is None or len(shape) == 2
    assert pointers.device == columns.device == values.device

    device = pointers.device

    shape_guess = (pointers.shape[0] - 1, columns.max().item() + 1)
    if shape is None:
        shape = shape_guess
    else:
        shape = (max(shape[0], shape_guess[0]), max(shape[1], shape_guess[1]))

    n, m = shape
    a = torch.zeros(n, m, dtype=values.dtype, device=device)
    i = torch.arange(n, device=device)
    i = fast_repeat(i, pointers[1:] - pointers[:-1])
    j = columns

    a[i, j] = values

    return a

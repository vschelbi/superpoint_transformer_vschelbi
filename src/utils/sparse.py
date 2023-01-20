import torch
from src.utils.tensor import is_dense, is_sorted, fast_repeat
from torch_scatter import scatter_mean


__all__ = [
    'indices_to_pointers', 'sizes_to_pointers', 'dense_to_csr', 'csr_to_dense',
    'sparse_sort', 'sparse_sort_along_direction']


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


def sizes_to_pointers(sizes: torch.LongTensor):
    """Convert a tensor of sizes into the corresponding pointers. This
    is a trivial but often-required operation.
    """
    assert sizes.dim() == 1
    assert sizes.dtype == torch.long
    zero = torch.zeros(1, device=sizes.device, dtype=torch.long)
    return torch.cat((zero, sizes)).cumsum(dim=0)


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


def sparse_sort(src, index, dim=0, descending=False, eps=1e-6):
    """Lexicographic sort 1D src points based on index first and src
    values second.

    Credit: https://github.com/rusty1s/pytorch_scatter/issues/48
    """
    # NB: we use double precision here to make sure we can capture fine
    # grained src changes even with very large index values.
    f_src = src.double()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) + index.double()*(-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_sort_along_direction(src, index, direction, descending=False):
    """Lexicographic sort N-dimensional src points based on index first
    and the projection of the src values along a direction second.
    """
    assert src.dim() == 2
    assert index.dim() == 1
    assert src.shape[0] == index.shape[0]
    assert direction.dim() == 2 or direction.dim() == 1

    if direction.dim() == 1:
        direction = direction.view(1, -1)

    # If only 1 direction is provided, apply the same direction to all
    # points
    if direction.shape[0] == 1:
        direction = direction.repeat(src.shape[0], 1)

    # If the direction is provided group-wise, expand it to the points
    if direction.shape[0] != src.shape[0]:
        direction = direction[index]

    # Compute the centroid for each group. This is not mandatory, but
    # may help avoid precision errors if absolute src coordinates are
    # too large
    centroid = scatter_mean(src, index, dim=0)[index]

    # Project the points along the associated direction
    projection = torch.einsum('ed, ed -> e', src - centroid, direction)

    # Sort the projections
    _, perm = sparse_sort(projection, index, descending=descending)

    return src[perm], perm

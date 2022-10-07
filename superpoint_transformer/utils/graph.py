import torch


__all__ = ['is_pyg_edge_format', 'isolated_nodes']


def is_pyg_edge_format(edge_index):
    """Check whether edge_index follows pytorch geometric graph edge
    format: a [2, N] torch.LongTensor.
    """
    return \
        isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 \
        and edge_index.dtype == torch.long and edge_index.shape[0] == 2


def isolated_nodes(edge_index, num_nodes=None):
    """Return a boolean mask of size num_nodes indicating which node has
    no edge in edge_index.
    """
    assert is_pyg_edge_format(edge_index)
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    device = edge_index.device
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[edge_index.unique()] = False
    return mask

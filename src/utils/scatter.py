import torch
from torch_scatter import scatter_add


__all__ = ['scatter_mean_weighted']


def scatter_mean_weighted(x, idx, w, dim_size=None):
    """Helper for scatter_mean with weights"""
    assert w.ge(0).all(), "Only positive weights are accepted"
    assert w.dim() == idx.dim() == 1, "w and idx should be 1D Tensors"
    assert x.shape[0] == w.shape[0] == idx.shape[0], \
        "Only supports weighted mean along the first dimension"

    # Concatenate w and x in the same tensor to only call scatter once
    w = w.view(-1, 1).float()
    wx = torch.cat((w, x * w), dim=1)

    # Scatter sum the wx tensor to obtain
    wx_segment = scatter_add(wx, idx, dim=0, dim_size=dim_size)

    # Extract the weighted mean from the result
    w_segment = wx_segment[:, 0]
    x_segment = wx_segment[:, 1:]
    w_segment[w_segment == 0] = 1
    mean_segment = x_segment / w_segment.view(-1, 1)

    return mean_segment

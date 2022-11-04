import torch
from torch import nn
from torch_scatter import scatter, scatter_softmax


__all__ = [
    'ScatterPool', 'ScatterMaxPool', 'ScatterMinPool', 'ScatterSumPool',
    'ScatterMeanPool']


def append_zeros(x, n):
    """Helper to append to `x` `n` zero-tensors with the appropriate
    shape.
    """
    shape = list(x.shape)
    shape[0] = n
    zeros = torch.zeros(shape, device=x.device)
    return torch.cat((x, zeros), dim=0)


class ScatterPool(nn.Module):
    def __init__(self, reduce='sum'):
        super().__init__()
        self.reduce = reduce

    def forward(self, x, idx, dim_size=None):
        """
        :param x: [N, C] FloatTensor
            Features
        :param idx: [N] LongTensor
            Indices of clusters
        :param dim_size: int
            Specify the expected output size (ie the number of groups).
            By default this is inferred from `idx.max() + 1`, but we
            may need to specify `dim_size` if some elements are missing
            'at the end'
        :return:
        """
        # Compute the pooling operation
        out = scatter(x, idx, dim=0, reduce=self.reduce)

        # Append zero-tensors to the output tensor if need be
        if dim_size is not None and dim_size > out.shape[0]:
            out = append_zeros(out, dim_size - out.shape[0])

        return out


class ScatterMaxPool(ScatterPool):
    def __init__(self):
        super().__init__(reduce='max')


class ScatterMinPool(ScatterPool):
    def __init__(self):
        super().__init__(reduce='min')


class ScatterMeanPool(ScatterPool):
    def __init__(self):
        super().__init__(reduce='mean')


class ScatterSumPool(ScatterPool):
    def __init__(self):
        super().__init__(reduce='sum')


class ScatterSoftmaxPool(nn.Module):
    def __init__(self, scaled=True):
        super().__init__()
        self.scaled = scaled

    def forward(self, x, w, idx, dim_size=None):
        """
        :param x: [N, C] FloatTensor
            Features
        :param w: [N] FloatTensor
            Weights to compute Softmax attention scores from
        :param idx: [N] LongTensor
            Indices of clusters
        :param dim_size: int
            Specify the expected output size (ie the number of groups).
            By default this is inferred from `idx.max() + 1`, but we
            may need to specify `dim_size` if some elements are missing
            'at the end'
        :return:
        """
        # If scaled softmax, first compute the scales, based on the
        # number of elements in each group
        if self.scaled:
            ones = torch.ones_like(idx, dtype=torch.float)  # [N]
            scales = (scatter(ones, idx, reduce='sum') + 1e-3) ** -0.5
            w = w * scales[idx]

        # Compute the softmax weights
        a = scatter_softmax(w, idx, dim=0)

        # Apply the softmax weights and sum across each group
        out = scatter(x * a, idx, dim=0, reduce='sum')

        # Append zero-tensors to the output tensor if need be
        if dim_size is not None and dim_size > out.shape[0]:
            out = append_zeros(out, dim_size - out.shape[0])

        return out

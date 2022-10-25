from torch import nn
from torch_scatter import scatter


__all__ = ['FastBatchNorm1d', 'SegmentUnitNorm']


class FastBatchNorm1d(nn.Module):
    """Credits: https://github.com/torch-points3d/torch-points3d"""
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    def _forward_dense(self, x):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x):
        """Batch norm 1D is not optimised for 2D tensors. The first
        dimension is supposed to be the batch and therefore not very
        large. So we introduce a custom version that leverages
        BatchNorm1D in a more optimised way.
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError(
                "Non supported number of dimensions {}".format(x.dim()))


class SegmentUnitNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos, idx):
        min_segment = scatter(pos, idx, dim=0, reduce='min')
        max_segment = scatter(pos, idx, dim=0, reduce='max')

        mean_segment = scatter(pos, idx, dim=0, reduce='mean')
        diameter_segment = (max_segment - min_segment).max(dim=1).values

        mean = mean_segment[idx]
        diameter = diameter_segment[idx]

        pos = (pos - mean) / (diameter.view(-1, 1) + 1e-2)

        return pos, diameter_segment

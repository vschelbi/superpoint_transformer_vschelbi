import torch
from torch import nn
from torch_scatter import scatter
from src.utils import scatter_mean_weighted
from torch_geometric.nn.norm import LayerNorm


__all__ = ['FastBatchNorm1d', 'UnitSphereNorm', 'LayerNorm']


class FastBatchNorm1d(nn.Module):
    """Credits: https://github.com/torch-points3d/torch-points3d"""

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

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


class UnitSphereNorm(nn.Module):
    """Normalize positions of same-segment nodes in a unit sphere of
    diameter 1 (ie radius 1/2).

    :param log_diameter: bool
        Whether the returned diameter should be log-normalized. This may
        be useful if using the diameter as a feature in downstream
        learning tasks
    """

    def __init__(self, log_diameter=False):
        super().__init__()
        self.log_diameter = log_diameter

    def forward(self, pos, idx, w=None, num_super=None):
        if w is not None:
            assert w.ge(0).all() and w.sum() > 0, \
                "At least one node must had a strictly positive weights"

        # Normalization
        if idx is None:
            pos, diameter = self._forward(pos, w=w)
        else:
            pos, diameter = self._forward_scatter(pos, idx, w=w, num_super=num_super)

        # Log-normalize the diameter if required. This facilitates using
        # the diameter as a feature in downstream learning tasks
        if self.log_diameter:
            diameter = torch.log(diameter + 1)

        return pos, diameter

    def _forward(self, pos, w=None):
        """Forward without scatter operations, in case `idx` is not
        provided. Applies the sphere normalization on all pos
        coordinates together.
        """
        # Compute the diameter (ie the maximum span along the main axes
        # here)
        min_ = pos.min(dim=0).values
        max_ = pos.max(dim=0).values
        diameter = (max_ - min_).max()

        # Compute the center of the nodes. If node weights are provided,
        # the weighted mean is computed
        if w is None:
            center = pos.mean(dim=0)
        else:
            w_sum = w.float().sum()
            w_sum = 1 if w_sum == 0 else w_sum
            center = (pos * w.view(-1, 1).float()).sum(dim=0) / w_sum
        center = center.view(1, -1)

        # Unit-sphere normalization
        pos = (pos - center) / (diameter + 1e-2)

        return pos, diameter.view(1, 1)

    def _forward_scatter(self, pos, idx, w=None, num_super=None):
        """Forward with scatter operations, in case `idx` is provided.
        Applies the sphere normalization for each segment separately.
        """
        # Compute the diameter (ie the maximum span along the main axes
        # here)
        min_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce='min')
        max_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce='max')
        diameter_segment = (max_segment - min_segment).max(dim=1).values

        # Compute the center of the nodes. If node weights are provided,
        # the weighted mean is computed
        if w is None:
            center_segment = scatter(
                pos, idx, dim=0, dim_size=num_super, reduce='mean')
        else:
            center_segment = scatter_mean_weighted(
                pos, idx, w, dim_size=num_super)

        # Compute per-node center and diameter
        center = center_segment[idx]
        diameter = diameter_segment[idx]

        # Unit-sphere normalization
        pos = (pos - center) / (diameter.view(-1, 1) + 1e-2)

        return pos, diameter_segment.view(-1, 1)

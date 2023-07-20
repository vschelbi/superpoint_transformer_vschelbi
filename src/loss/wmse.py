import torch


__all__ = ['OffsetLoss']


class OffsetLoss(torch.nn.MSELoss):
    """Weighted mean squared L2 distance between predicted and target
    offsets. This is basically the MSELoss except that the mean is not
    computed across all items and dimensions if `spatial_mean=False`.
    This means the number of dimensions in the offsets will scale the
    loss. Besides, positive weights can optionally passed to give more
    importance to some points.
    """

    def __init__(self, *args, spatial_mean=False, **kwargs):
        super().__init__(*args, reduction='none' **kwargs)
        self.spatial_mean = spatial_mean

    def forward(self, input, target, weight=None):
        assert weight.ge(0).all(), "Weights must be positive."
        assert weight.gt(0).any(), "At least one weight must be non-zero."

        # Compute the loss, without reduction
        loss = super().forward(input, target)

        # Sum the loss across the spatial dimension
        if not self.spatial_mean:
            loss = loss.sum(dim=1).view(-1, 1)

        # Compute the weighted mean
        return (loss * (weight / weight.sum()).view(-1, 1)).mean()

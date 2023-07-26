import torch


__all__ = ['OffsetLoss']


class OffsetLoss(torch.nn.MSELoss):
    """Weighted mean squared error between predicted and target
    offsets. This is basically the MSELoss except that positive weights
    must be passed to give more importance to some points.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction='none', **kwargs)

    def forward(self, input, target, weight):
        assert weight.ge(0).all(), "Weights must be positive."
        assert weight.gt(0).any(), "At least one weight must be non-zero."

        # Compute the loss, without reduction
        loss = super().forward(input, target)

        # Sum the loss terms across the spatial dimension, so the
        # downstream averaging does not normalize by the number of
        # dimensions
        loss = loss.sum(dim=1).view(-1, 1)

        # Compute the weighted mean
        return (loss * (weight / weight.sum()).view(-1, 1)).sum()

from torch.nn import L1Loss as TorchL1Loss
from torch.nn import MSELoss as TorchL2Loss


__all__ = ['WeightedL2Loss', 'WeightedL1Loss', 'L2Loss', 'L1Loss']


class WeightedLossMixIn:
    """A mix-in for converting a torch loss into an item-weighted loss.
    """
    def forward(self, input, target, weight):
        if weight is not None:
            assert weight.ge(0).all(), "Weights must be positive."
            assert weight.gt(0).any(), "At least one weight must be non-zero."

        # Compute the loss, without reduction
        loss = super().forward(input, target)

        # Sum the loss terms across the spatial dimension, so the
        # downstream averaging does not normalize by the number of
        # dimensions
        loss = loss.sum(dim=1).view(-1, 1)

        # If weights are None, fallback to normal unweighted L2 loss
        if weight is None:
            return loss.mean()

        # Compute the weighted mean
        return (loss * (weight / weight.sum()).view(-1, 1)).sum()


class WeightedL2Loss(WeightedLossMixIn, TorchL2Loss):
    """Weighted mean squared error (ie L2 loss) between predicted and
    target offsets. This is basically the MSELoss except that positive
    weights must be passed to give more importance to some items.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction='none', **kwargs)


class WeightedL1Loss(WeightedLossMixIn, TorchL1Loss):
    """Weighted L1 loss between predicted and target offsets. This is
    basically the L1Loss except that positive weights must be passed to
    give more importance to some items.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction='none', **kwargs)


class L2Loss(WeightedL2Loss):
    """Mean squared error (ie L2 loss) between predicted and target
    offsets.

    The forward signature allows using this loss as a weighted loss,
    with input weights ignored.
    """

    def forward(self, input, target, weight):
        return super().forward(input, target, None)


class L1Loss(WeightedL1Loss):
    """L1 loss between predicted and target offsets.

    The forward signature allows using this loss as a weighted loss,
    with input weights ignored.
    """

    def forward(self, input, target, weight):
        return super().forward(input, target, None)

import logging
from torch import Tensor
from typing import Tuple
from torchmetrics import MeanSquaredError
from torchmetrics.utilities.checks import _check_same_shape


log = logging.getLogger(__name__)


__all__ = ['WeightedMeanSquaredError']


class WeightedMeanSquaredError(MeanSquaredError):
    """Simply torchmetrics' MeanSquaredError with weighted mean to give
    more importance to some points.
    """

    def update(
            self,
            preds: Tensor,
            target: Tensor,
            weight: Tensor
    ) -> None:
        """Update state with predictions, targets, and weights."""
        sum_squared_error, sum_weight = _weighted_mean_squared_error_update(
            preds, target, weight)

        self.sum_squared_error += sum_squared_error
        self.total += sum_weight


def _weighted_mean_squared_error_update(
        preds: Tensor,
        target: Tensor,
        weight: Tensor
) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute weighted Mean
    Squared Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        weight: weight tensor
    """
    assert weight.dim() == 1
    assert weight.numel() == preds.shape[0]
    _check_same_shape(preds, target)
    sum_squared_error = (weight.view(-1, 1) * (preds - target)**2).sum()
    sum_weight = weight.sum()
    return sum_squared_error, sum_weight

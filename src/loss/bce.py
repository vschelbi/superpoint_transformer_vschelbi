import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as TorchBCEWithLogitsLoss


__all__ = ['BCEWithLogitsLoss']


class BCEWithLogitsLoss(TorchBCEWithLogitsLoss):
    """Torch's BCEWithLogitsLoss without the constraint of passing
    `pos_weight` as a Tensor. This simplifies instantiation with hydra.
    """
    def __init__(self, *args, pos_weight=None, **kwargs):
        if pos_weight is not None and not isinstance(pos_weight, Tensor):
            pos_weight = torch.as_tensor(pos_weight)
        super().__init__(*args, pos_weight=pos_weight, **kwargs)

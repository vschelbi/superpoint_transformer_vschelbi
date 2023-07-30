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

    def load_state_dict(self, state_dict, strict=True):
        """Normal `load_state_dict` behavior, except for the shared
        `pos_weight`.
        """
        # Get the weight from the state_dict
        pos_weight = state_dict.get('pos_weight')
        state_dict.pop('pos_weight')

        # Normal load_state_dict, ignoring pos_weight
        out = super().load_state_dict(state_dict, strict=strict)

        # Set the pos_weight
        self.pos_weight = pos_weight

        return out

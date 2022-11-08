import torch
from torch import nn


__all__ = ['CatFusion', 'ResidualFusion', 'TakeFirstFusion', 'TakeSecondFusion']


class BaseFusion(nn.Module):
    def forward(self, x1, x2):
        if x1 is None and x2 is None:
            return None
        if x1 is None:
            return x2
        if x2 is None:
            return x1
        return self._func(x1, x2)

    def _func(self, x1, x2):
        raise NotImplementedError


class CatFusion(BaseFusion):
    def _func(self, x1, x2):
        return torch.cat((x1, x2), dim=1)


class ResidualFusion(BaseFusion):
    def _func(self, x1, x2):
        return x1 + x2


class TakeFirstFusion(BaseFusion):
    def _func(self, x1, x2):
        return x1


class TakeSecondFusion(BaseFusion):
    def _func(self, x1, x2):
        return x2


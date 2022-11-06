import torch
from torch import nn


__all__ = ['CatFusion', 'ResidualFusion']


class CatFusion(nn.Module):
    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)


class ResidualFusion(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2

from torch import nn
from torch_scatter import scatter


__all__ = [
    'SegmentMaxPool', 'SegmentMinPool', 'SegmentSumPool', 'SegmentMeanPool']


class SegmentPool(nn.Module):
    _REDUCE = 'sum'
    def __init__(self):
        super().__init__()

    def forward(self, x, idx):
        scatter(x, idx, reduce=self.reduce)


class SegmentMaxPool(SegmentPool):
    _REDUCE = 'max'


class SegmentMinPool(SegmentPool):
    _REDUCE = 'min'


class SegmentMeanPool(SegmentPool):
    _REDUCE = 'mean'


class SegmentSumPool(SegmentPool):
    _REDUCE = 'sum'

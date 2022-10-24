from torch import nn
from torch_scatter import scatter


__all__ = [
    'SegmentPool', 'SegmentMaxPool', 'SegmentMinPool', 'SegmentSumPool',
    'SegmentMeanPool']


class SegmentPool(nn.Module):
    def __init__(self, reduce='sum'):
        super().__init__()
        self.reduce = reduce

    def forward(self, x, idx):
        scatter(x, idx, dim=0, reduce=self.reduce)


class SegmentMaxPool(SegmentPool):
    def __init__(self):
        super().__init__(reduce='max')


class SegmentMinPool(SegmentPool):
    def __init__(self):
        super().__init__(reduce='min')


class SegmentMeanPool(SegmentPool):
    def __init__(self):
        super().__init__(reduce='mean')


class SegmentSumPool(SegmentPool):
    def __init__(self):
        super().__init__(reduce='sum')

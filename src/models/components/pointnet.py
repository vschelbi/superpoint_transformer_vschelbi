import torch
from torch import nn
from src.nn import PointMLP, PointClassifier, SegmentPool, SegmentUnitNorm


class PointNet(nn.Module):
    def __init__(
            self, local_channels, global_channels, num_classes=None,
            reduce='max'):
        super().__init__()

        assert local_channels is None or len(local_channels) > 1
        assert global_channels is None or len(global_channels) > 1
        assert global_channels is not None or local_channels is not None

        if local_channels is not None and len(local_channels) > 1:
            self.local_nn = PointMLP(local_channels)
            last_channel = local_channels[-1]
        else:
            self.local_nn = None

        if global_channels is not None and len(global_channels) > 1:
            self.global_nn = PointMLP(global_channels)
            last_channel = global_channels[-1]
        else:
            self.global_nn = None

        self.pool = SegmentPool(reduce=reduce)

        if num_classes is not None:
            self.head = PointClassifier(last_channel, num_classes)
        else:
            self.head = None

    def forward(self, pos, x, idx):
        # Normalize each segment to a unit sphere
        pos, diameter = SegmentUnitNorm(pos, idx)

        # Add normalized coordinates to the point features
        x = torch.cat((x, pos), dim=1)

        # Compute pointwise features
        x = self.local_nn(x) if self.local_nn else x

        # Segment pooling
        x_global = self.pool(x, idx)

        # Add initial diameter to segment features
        x_global = torch.cat((x_global, diameter.unsqueeze(-1)), dim=1)

        # Compute segment-level features
        x_global = self.global_nn(x_global) if self.global_nn else x_global

        # Classifying head
        x_out = self.head(x_global) if self.head else x_global

        return x_out

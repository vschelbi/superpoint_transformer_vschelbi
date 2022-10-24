from torch import nn
from src.nn import PointMLP, PointClassifier
from torch_scatter import scatter


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

        self.reduce = reduce

        if num_classes is not None:
            self.head = PointClassifier(last_channel, num_classes)
        else:
            self.head = None

    def forward(self, x, idx):
        x = self.local_nn(x) if self.local_nn else x
        x_global = scatter(x, idx, dim=0, reduce=self.reduce)
        x_global = self.global_nn(x_global) if self.global_nn else x_global
        x_out = self.head(x_global) if self.head else x_global
        return x_out

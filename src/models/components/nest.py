import torch
from torch import nn
from src.nn import ScatterUnitNorm
from src.utils import listify


def parse_list_args(args):
    if args is None or len(args) == 0:
        return

    # Check whether args contains list of lists
    is_list_list = hasattr(args[0], '__iter__')


def init_weights(module):
    """Manual weight initialization. Allows setting specific init modes
    for certain modules. In particular, it is recommended to initialize
    Linear layers with Xavier initialization:
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    see Romain's: https://github.com/romainloiseau/Helix4D/blob/87a9d01922e22ae692851233e8614d2b4f7baeed/helix4d/model/voxel/transformer/utils.py#3
    """
    pass


class NeST(nn.Module):
    """NeST is composed of
    - input spherenorm on points and stuff
    - nn.ModuleList of Stage DOWN
    - nn.ModuleList of Stage UP
    - skip connection fusion mode (concatenation / residual)

    """

class Stage(nn.Module):
    """Stage is composed of
    - nn.ModuleList of TransformerBlock
    - input MLP to change dim (optional)
    - output MLP to change dim  (optional)
    - input DOWN Pool (optional) see UNet: https://www.researchgate.net/publication/331406702/figure/fig2/AS:731276273262594@1551361258173/Illustration-of-the-U-net-architecture-The-figure-illustrates-the-U-net-architecture.png
    - output UP Pool (optional) see UNet: https://www.researchgate.net/publication/331406702/figure/fig2/AS:731276273262594@1551361258173/Illustration-of-the-U-net-architecture-The-figure-illustrates-the-U-net-architecture.png
    """
    pass


class NeST(nn.Module):
    def __init__(
            self, self_attention=None, down=None, up=None, num_classes=None,
            reduce='max'):
        super().__init__()

        # Convert input arguments to nested lists
        self_attention = listify(self_attention)
        down = listify(down)
        up = listify(up)

        assert self_attention is None or len(self_attention) > 0
        assert down is None or len(down) > 0
        assert up is None or len(up) > 0
        assert down is None and up is None or len(down) == len(up)





        assert local_channels is None or len(local_channels) > 1
        assert global_channels is None or len(global_channels) > 1
        assert global_channels is not None or local_channels is not None

        self.sphere_norm = ScatterUnitNorm()



    def forward(self, pos, x, idx):
        # Normalize each segment to a unit sphere
        pos, diameter = self.sphere_norm(pos, idx)

        # Add normalized coordinates to the point features
        x = torch.cat((x, pos), dim=1)

        # Compute pointwise features
        x = self.local_nn(x) if self.local_nn else x

        # Segment pooling
        x_global = self.pool(x, idx)

        # Add initial diameter to segment features
        x_global = torch.cat((x_global, diameter.view(-1, 1)), dim=1)

        # Compute segment-level features
        x_global = self.global_nn(x_global) if self.global_nn else x_global

        # Classifying head
        x_out = self.head(x_global) if self.head else x_global

        return x_out

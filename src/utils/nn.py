from torch import nn


__all__ = ['init_weights']


def init_weights(m):
    """Manual weight initialization. Allows setting specific init modes
    for certain modules. In particular, it is recommended to initialize
    Linear layers with Xavier initialization:
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

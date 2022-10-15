from superpoint_transformer.transforms import Transform
from superpoint_transformer.data import NAG


__all__ = ['DataTo', 'NAGTo']


class DataTo(Transform):
    """Move Data object to specified device."""

    def __init__(self, device):
        self.device = device

    def _process(self, data):
        return data.to(self.device)


class NAGTo(Transform):
    """Move Data object to specified device."""

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, device):
        self.device = device

    def _process(self, nag):
        return nag.to(self.device)

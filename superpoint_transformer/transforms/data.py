from superpoint_transformer.transforms import Transform
from superpoint_transformer.data import Data, NAG


__all__ = ['DataToNAG', 'NAGToData']


class DataToNAG(Transform):
    """Convert Data to a single-level NAG."""

    _IN_TYPE = Data
    _OUT_TYPE = NAG

    def _process(self, data):
        return NAG([Data])


class NAGToData(Transform):
    """Convert a single-level NAG to Data."""

    _IN_TYPE = NAG
    _OUT_TYPE = Data

    def _process(self, nag):
        assert len(nag) == 1
        return NAG[0]

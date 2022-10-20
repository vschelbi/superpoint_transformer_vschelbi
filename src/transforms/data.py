from src.transforms import Transform
from src.data import Data, NAG
from src.utils import fast_randperm


__all__ = [
    'DataToNAG', 'NAGToData', 'RemoveAttributes', 'DropoutSegments',
    'SampleSegments', 'NodeSize']


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
        assert nag.num_levels == 1
        return NAG[0]


class RemoveAttributes(Transform):
    """Remove attributes of a Data object based on their name.

    :param keys: list(str)
        List of attribute names
    :param strict: bool=False
        If True, will raise an exception if an attribute from key is
        not within the input Data keys
    """

    def __init__(self, keys=[], strict=False):
        self.keys = keys
        self.strict = strict

    def _process(self, data):
        keys = set(data.keys)
        for k in self.keys:
            if k not in keys and self.strict:
                raise Exception(f"key: {k} is not within Data keys: {keys}")
        for k in self.keys:
            delattr(data, k)
        return data


class DropoutSegments(Transform):
    """Remove randomly-picked nodes from each level 1+ of the NAG. This
    operation relies on `NAG.select()` to maintain index consistency
    across the NAG levels.

    Note: we do not directly prune level-0 points, see `SampleSegments`
    for that. For speed consideration, it is recommended to use
    `SampleSegments` first before `DropoutSegments`, to minimize the
    number of level-0 points to manipulate.

    :param p: float or list(float)
        Portion of nodes to be dropped. A list may be passed to prune
        NAG 1+ levels with different probabilities.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, p=0.2):
        assert isinstance(p, (float, list))
        assert 0 <= p < 1
        self.p = p

    def _process(self, nag):
        if not isinstance(self.p, list):
            p = [self.p] * (nag.num_levels - 1)
        else:
            p = self.p

        # Drop some nodes from each NAG level. Note that we start
        # dropping from the highest to the lowest level, to accelerate training
        device = nag.device
        for i_level in range(nag.num_levels - 1, 0, -1):
            if p[i_level - 1] <= 0:
                continue

            # Shuffle the order of points
            num_nodes = nag[i_level].num_nodes
            perm = fast_randperm(num_nodes, device=device)
            num_keep = num_nodes - int(num_nodes * p[i_level - 1])
            idx = perm[:num_keep]

            # Select the nodes and update the NAG structure accordingly
            nag = nag.select(i_level, idx)

        return nag


class SampleSegments(Transform):
    """Sample elements at `low`-level, based on which segment they
    belong to at `high`-level.

    The sampling operation is run without replacement and each segment
    is sampled at least `n_min` and at most `n_max` times, within the
    limits allowed by its actual size.

    Optionally, a `mask` can be passed to filter out some `low`-level
    points.

    :param high: int
        Partition level of the segments we want to sample. By default,
        `high=1` to sample the level-1 segments
    :param low: int
        Partition level we will sample from, guided by the `high`
        segments. By default, `high=0` to sample the level-0 points.
        `low=-1` is accepted when level-0 has a `sub` attribute (ie
        level-0 points are themselves segments of `-1` level absent
        from the NAG object).
    :param n_max: int
        Maximum number of `low`-level elements to sample in each
        `high`-level segment
    :param n_min: int
        Minimum number of `low`-level elements to sample in each
        `high`-level segment, within the limits of its size (ie no
        oversampling)
    :param mask: list, np.ndarray, torch.LongTensor, torch.BoolTensor
        Indicates a subset of `low`-level elements to consider. This
        allows ignoring
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, high=1, low=0, n_max=32, n_min=16, mask=None):
        assert isinstance(high, int)
        assert isinstance(low, int)
        assert isinstance(n_max, int)
        assert isinstance(n_min, int)
        self.high = high
        self.low = low
        self.n_max = n_max
        self.n_min = n_min
        self.mask = mask

    def _process(self, nag):
        idx = nag.get_sampling(
            high=self.high, low=self.low, n_max=self.n_max, n_min=self.n_min,
            return_pointers=False)
        return nag.select(self.low, idx)


class NodeSize(Transform):
    """Compute the number of `low`-level elements are contained in each
    segment, at each above-level. Results are save in the `node_size`
    attribute of the corresponding Data objects.

    Note: `low=-1` is accepted when level-0 has a `sub` attribute
    (ie level-0 points are themselves segments of `-1` level absent
    from the NAG object).

    :param low: int
        Level whose elements we want to count
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, low=0):
        assert isinstance(low, int) and low >= -1
        self.low = low

    def _process(self, nag):
        for i_level in range(self.low + 1, nag.num_levels):
            nag[i_level].node_size = nag.get_sub_size(i_level, low=self.low)
        return nag
